import pyrealsense2 as rs
import numpy as np
import cv2


# -------------------- Bottle Detection --------------------

def detect_red_bottles(color_image):
    hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)

    # Red color ranges (HSV)
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = mask1 | mask2

    # Clean mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    boxes = []
    for cnt in contours:
        if cv2.contourArea(cnt) > 1500:  # filter small noise
            x, y, w, h = cv2.boundingRect(cnt)
            boxes.append((x, y, w, h))

    return boxes


# -------------------- Depth to 3D --------------------

def pixel_to_3d(depth_frame, intrinsics, x, y):
    depth = depth_frame.get_distance(x, y)
    if depth <= 0:
        return None

    point = rs.rs2_deproject_pixel_to_point(
        intrinsics, [x, y], depth
    )
    return point  # (X, Y, Z) in meters

# -------------------- Main --------------------

def main():
    pipeline = rs.pipeline()
    started = False

    try:
        # Configure streams
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

        pipeline.start(config)
        started = True

        print("Running... Press 'q' to quit")

        while True:
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()

            if not depth_frame or not color_frame:
                continue

            # Convert to numpy
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            # Depth colormap
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=0.03),
                cv2.COLORMAP_JET
            )

            # Camera intrinsics
            intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics

            # Detect bottles
            boxes = detect_red_bottles(color_image)

            for i, (x, y, w, h) in enumerate(boxes):
                cx = x + w // 2
                cy = y + h // 2

                point = pixel_to_3d(depth_frame, intrinsics, cx, cy)
                if point is None:
                    continue

                X, Y, Z = point

                # Draw bounding box
                cv2.rectangle(color_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.circle(color_image, (cx, cy), 5, (0, 255, 0), -1)

                # Display 3D position
                text = f"Bottle {i}: X={X:.2f} Y={Y:.2f} Z={Z:.2f} m"
                cv2.putText(
                    color_image, text, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
                )

                print(text)

            # Show windows
            cv2.imshow("RGB Image", color_image)
            cv2.imshow("Depth Image", depth_colormap)

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
