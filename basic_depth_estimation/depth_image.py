import pyrealsense2 as rs
import numpy as np
import cv2


def main(args=None):
    pipeline = rs.pipeline()
    started = False

    try:
        # Configure streams
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

        pipeline.start(config)
        started = True

        print("Press 'q' to quit")

        while True:
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()

            if not depth_frame or not color_frame:
                continue

            # Convert to numpy
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            # ---- Depth (Colorized) ----
            depth_color = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=0.03),
                cv2.COLORMAP_JET
            )

            # ---- Depth (Grayscale) ----
            depth_gray = cv2.normalize(
                depth_image,
                None,
                0,
                255,
                cv2.NORM_MINMAX,
                dtype=cv2.CV_8U
            )

            # Show all 3
            cv2.imshow("RGB Image", color_image)
            cv2.imshow("Depth Image (Color)", depth_color)
            cv2.imshow("Depth Image (Gray)", depth_gray)

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
