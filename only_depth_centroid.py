import cv2
import mediapipe as mp
import numpy as np
import pyrealsense2 as rs
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)

# Initialize MediaPipe face mesh model
face_mesh = mp.solutions.face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Configure depth and color streams from RealSense
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
try:
    pipeline.start(config)
except Exception as e:
    logging.error(f"Failed to start pipeline: {e}")
    exit(1)

try:
    while True:
        try:
            # Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames(10000)  # Increased timeout to 10 seconds
        except RuntimeError as e:
            logging.warning(f"RuntimeError: {e}")
            continue  # Skip to the next loop iteration and try again

        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        if not depth_frame or not color_frame:
            logging.warning("No frames received, check your camera setup.")
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        image_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

        # Process the image and detect faces
        results = face_mesh.process(image_rgb)
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                sum_x, sum_y = 0, 0
                num_landmarks = 0
                for idx_tuple in mp.solutions.face_mesh.FACEMESH_LIPS:
                    for idx in idx_tuple:
                        landmark = face_landmarks.landmark[idx]
                        sum_x += landmark.x
                        sum_y += landmark.y
                        num_landmarks += 1

                if num_landmarks > 0:
                    centroid_x = sum_x / num_landmarks
                    centroid_y = sum_y / num_landmarks

                    # Convert centroid coordinates from normalized to pixel
                    h, w, _ = color_image.shape
                    cx, cy = int(centroid_x * w), int(centroid_y * h)

                    # Retrieve the depth from the depth image
                    depth = depth_image[cy, cx]

                    # Get the 3D coordinates
                    depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
                    x, y, z = rs.rs2_deproject_pixel_to_point(depth_intrin, [cx, cy], depth)

                    logging.info(f"Mouth Centroid Coordinates in 3D Space: ({x}, {y}, {z})")
                    
                    # Visualize the centroid on the image
                    cv2.circle(color_image, (cx, cy), 5, (0, 0, 255), -1)

        # Display the image
        cv2.imshow("Real-time Video", color_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    pipeline.stop()
    cv2.destroyAllWindows()

