from ultralytics import YOLO
import pyrealsense2 as rs
import numpy as np
import cv2


model = YOLO("yolov8n.pt")


def median_depth_in_patch(depth_frame, cx, cy, r=4):
    """Median depth (meters) in a (2r+1)x(2r+1) patch around (cx,cy)."""
    vals = []
    for y in range(cy - r, cy + r + 1):
        for x in range(cx - r, cx + r + 1):
            d = depth_frame.get_distance(x, y)
            if d > 0:
                vals.append(d)
    if not vals:
        return 0.0
    return float(np.median(vals))


def estimate_approach_from_depth_gradient(depth_frame, cx, cy, step=3):
    """
    Very simple approach direction estimate from local depth gradient.
    Returns a unit vector in camera coordinates, roughly pointing "toward" the object surface.
    """
    # Sample nearby depths (meters)
    d_c = depth_frame.get_distance(cx, cy)
    if d_c <= 0:
        return None

    d_x1 = depth_frame.get_distance(cx - step, cy)
    d_x2 = depth_frame.get_distance(cx + step, cy)
    d_y1 = depth_frame.get_distance(cx, cy - step)
    d_y2 = depth_frame.get_distance(cx, cy + step)

    # If missing data, fall back to "straight ahead"
    if min(d_x1, d_x2, d_y1, d_y2) <= 0:
        v = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        return v

    # Depth increases with distance; gradient gives surface tilt cue (approx)
    dzdx = (d_x2 - d_x1) / (2.0 * step)
    dzdy = (d_y2 - d_y1) / (2.0 * step)

    # Construct a pseudo-normal; sign chosen so arrow points outward-ish
    v = np.array([-dzdx, -dzdy, 1.0], dtype=np.float32)
    n = np.linalg.norm(v)
    if n < 1e-6:
        return np.array([0.0, 0.0, 1.0], dtype=np.float32)
    return v / n


def deproject(intr, u, v, depth_m):
    """(u,v,depth)->(X,Y,Z) in meters."""
    X, Y, Z = rs.rs2_deproject_pixel_to_point(intr, [float(u), float(v)], float(depth_m))
    return float(X), float(Y), float(Z)


def main():
    pipeline = rs.pipeline()
    started = False

    try:
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

        pipeline.start(config)
        started = True

        print("Running YOLO grasp points. Press 'q' to quit.")

        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            color = np.asanyarray(color_frame.get_data())
            h, w = color.shape[:2]

            intr = depth_frame.profile.as_video_stream_profile().intrinsics

            # YOLO detect
            results = model(color, verbose=False)

            for r in results:
                for box in r.boxes:
                    cls = int(box.cls[0])
                    label = model.names[cls]
                    if label != "bottle":
                        continue

                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    # Clamp bounds
                    x1 = max(0, min(w - 1, x1))
                    x2 = max(0, min(w - 1, x2))
                    y1 = max(0, min(h - 1, y1))
                    y2 = max(0, min(h - 1, y2))
                    if x2 <= x1 or y2 <= y1:
                        continue

                    # --- Choose grasp pixel (good region: ~60% down from top) ---
                    gx = (x1 + x2) // 2
                    gy = y1 + int(0.60 * (y2 - y1))

                    gx = max(5, min(w - 6, gx))
                    gy = max(5, min(h - 6, gy))

                    # --- Stable depth via median patch ---
                    depth_m = median_depth_in_patch(depth_frame, gx, gy, r=4)
                    if depth_m <= 0:
                        continue

                    # --- 3D grasp position ---
                    X, Y, Z = deproject(intr, gx, gy, depth_m)

                    # --- Approach direction (approx) ---
                    approach = estimate_approach_from_depth_gradient(depth_frame, gx, gy, step=3)
                    if approach is None:
                        approach = np.array([0.0, 0.0, 1.0], dtype=np.float32)

                    # --- Gripper width estimate (meters) from pixel width + depth ---
                    pixel_width = max(1, x2 - x1)
                    gripper_width_m = (pixel_width * Z) / float(intr.fx)

                    # Draw bbox + grasp point
                    cv2.rectangle(color, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.circle(color, (gx, gy), 5, (0, 255, 0), -1)

                    # Draw a 2D arrow for approach direction (projected in image plane)
                    # Use the x/y components to draw an arrow (purely for visualization)
                    arrow_len = 60
                    ax = int(gx + (-approach[0]) * arrow_len)
                    ay = int(gy + (-approach[1]) * arrow_len)
                    ax = max(0, min(w - 1, ax))
                    ay = max(0, min(h - 1, ay))
                    cv2.arrowedLine(color, (gx, gy), (ax, ay), (255, 255, 255), 2, tipLength=0.2)

                    text1 = f"Grasp XYZ: {X:.2f}, {Y:.2f}, {Z:.2f} m"
                    text2 = f"Width ~ {gripper_width_m*1000:.0f} mm"
                    cv2.putText(color, text1, (x1, max(20, y1 - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    cv2.putText(color, text2, (x1, min(h - 10, y2 + 20)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    # Print grasp data (robot-ready basics)
                    print(
                        f"GRASP: pos=({X:.3f},{Y:.3f},{Z:.3f}) m  "
                        f"approach=({approach[0]:.3f},{approach[1]:.3f},{approach[2]:.3f})  "
                        f"width={gripper_width_m:.3f} m"
                    )

            cv2.imshow("YOLO Grasp Points", color)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("\nStopped by user")

    except Exception as e:
        print("Error:", e)

    finally:
        if started:
            pipeline.stop()
        cv2.destroyAllWindows()
        print("Camera stopped cleanly")


if __name__ == "__main__":
    main()
