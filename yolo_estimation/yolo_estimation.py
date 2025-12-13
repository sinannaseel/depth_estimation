from ultralytics import YOLO
import pyrealsense2 as rs
import numpy as np
import cv2


# Load YOLO model
model = YOLO("yolov8n.pt")  # small & fast


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

        print("Running YOLO + Depth. Press q to quit")

        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()

            if not color_frame or not depth_frame:
                continue

            color = np.asanyarray(color_frame.get_data())

            # Run YOLO
            results = model(color, verbose=False)

            intr = depth_frame.profile.as_video_stream_profile().intrinsics

            for r in results:
                for box in r.boxes:
                    cls = int(box.cls[0])
                    label = model.names[cls]

                    # Only bottles
                    if label != "bottle":
                        continue

                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2

                    depth = depth_frame.get_distance(cx, cy)
                    if depth <= 0:
                        continue

                    X, Y, Z = rs.rs2_deproject_pixel_to_point(
                        intr, [cx, cy], depth
                    )

                    # Draw results
                    cv2.rectangle(color, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.circle(color, (cx, cy), 4, (0, 255, 0), -1)

                    text = f"X={X:.2f} Y={Y:.2f} Z={Z:.2f} m"
                    cv2.putText(
                        color,
                        text,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2
                    )

                    print(f"Bottle 3D pose: {text}")

            cv2.imshow("YOLO + Depth", color)

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
