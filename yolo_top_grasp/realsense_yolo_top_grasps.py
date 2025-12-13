from ultralytics import YOLO
import pyrealsense2 as rs
import numpy as np
import cv2


model = YOLO("yolov8n.pt")


def median_depth(depth_frame, cx, cy, r=5):
    vals = []
    for y in range(cy - r, cy + r + 1):
        for x in range(cx - r, cx + r + 1):
            d = depth_frame.get_distance(x, y)
            if d > 0:
                vals.append(d)
    if not vals:
        return 0.0
    return float(np.median(vals))


def main():
    pipeline = rs.pipeline()
    started = False

    # List to store top grasps
    grasp_list = []

    try:
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

        pipeline.start(config)
        started = True

        print("Top-grasp detection running. Press 'q' to quit.")

        while True:
            grasp_list.clear()  # refresh each frame

            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            color = np.asanyarray(color_frame.get_data())
            h, w = color.shape[:2]

            intr = depth_frame.profile.as_video_stream_profile().intrinsics

            results = model(color, verbose=False)

            for r in results:
                for box in r.boxes:
                    cls = int(box.cls[0])
                    if model.names[cls] != "bottle":
                        continue

                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    # Clamp
                    x1 = max(0, min(w - 1, x1))
                    x2 = max(0, min(w - 1, x2))
                    y1 = max(0, min(h - 1, y1))
                    y2 = max(0, min(h - 1, y2))
                    if x2 <= x1 or y2 <= y1:
                        continue

                    # ---- Top grasp pixel ----
                    gx = (x1 + x2) // 2
                    gy = y1 + int(0.15 * (y2 - y1))
                    gx = max(6, min(w - 7, gx))
                    gy = max(6, min(h - 7, gy))

                    # ---- Depth ----
                    Z = median_depth(depth_frame, gx, gy, r=5)
                    if Z <= 0:
                        continue

                    X, Y, Z = rs.rs2_deproject_pixel_to_point(
                        intr, [gx, gy], Z
                    )

                    # ---- Fixed top-down approach ----
                    approach = (0.0, 0.0, 1.0)

                    # ---- Gripper width estimate ----
                    pixel_width = x2 - x1
                    gripper_width = (pixel_width * Z) / intr.fx

                    grasp = {
                        "position": (X, Y, Z),
                        "approach": approach,
                        "width": gripper_width,
                        "bbox": (x1, y1, x2, y2)
                    }
                    grasp_list.append(grasp)

                    # ---- Visualization ----
                    cv2.rectangle(color, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.circle(color, (gx, gy), 6, (255, 255, 255), -1)
                    cv2.putText(
                        color,
                        f"Top grasp Z={Z:.2f}m",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2
                    )

            # ---- Show ----
            cv2.imshow("Top Grasps", color)

            # ---- Debug print grasp list ----
            if grasp_list:
                print("Grasps:")
                for i, g in enumerate(grasp_list):
                    print(
                        f"  {i}: pos={g['position']} "
                        f"width={g['width']:.3f}m"
                    )

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("\nStopped by user")

    finally:
        if started:
            pipeline.stop()
        cv2.destroyAllWindows()
        print("Camera stopped cleanly")


if __name__ == "__main__":
    main()
