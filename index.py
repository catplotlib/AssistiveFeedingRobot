import cv2
import mediapipe as mp
import numpy as np
import pyrealsense2 as rs
import logging
from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS

# Setup logging
logging.basicConfig(level=logging.INFO)

def apply_transformation(x_cam, y_cam, z_cam):
    # Define the translation from camera to robot base
    x_offset = 0.05  # camera is 0.05 meters above the robot base
    y_offset = 0.3   # camera is 0.3 meters in front of the robot base
    z_offset = 0     # assuming no horizontal offset

    # Define the rotation matrix to align camera coordinate system to the robot's coordinate system
    rotation_matrix = np.array([
        [0,  0, 1],
        [1,  0, 0],
        [0, -1, 0]
    ])

    # Camera coordinates vector
    cam_vector = np.array([x_cam, y_cam, z_cam])

    # Apply rotation
    rotated_vector = np.dot(rotation_matrix, cam_vector)

    # Apply translation
    robot_vector = rotated_vector + np.array([x_offset, y_offset, z_offset])

    return robot_vector

# Initialize MediaPipe face mesh model
face_mesh = mp.solutions.face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Configure depth and color streams from RealSense
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Initialize the robot
bot = InterbotixManipulatorXS(robot_model='wx250s', group_name='arm', gripper_name='gripper')

# Start streaming
try:
    pipeline.start(config)
    while True:
        try:
            # Increase the timeout to 10 seconds to handle potential delays
            frames = pipeline.wait_for_frames(10000)
        except RuntimeError as e:
            logging.error(f"Timeout error: {str(e)} - Retrying frame capture")
            continue  # Skip to the next iteration if a frame can't be obtained

        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        if not depth_frame or not color_frame:
            logging.warning("No frames received, check your camera setup.")
            continue

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
                    h, w, _ = color_image.shape
                    cx, cy = int(centroid_x * w), int(centroid_y * h)
                    depth = depth_image[cy, cx]

                    depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
                    x, y, z = rs.rs2_deproject_pixel_to_point(depth_intrin, [cx, cy], depth)
                    
                    # Apply transformation to robot coordinates
                    x_robot, y_robot, z_robot = apply_transformation(x, y, z + 0.2)  # Adjusting height from the base and assuming the camera is centered
                    
                    logging.info(f"Transformed Robot Coordinates in 3D Space: ({x_robot}, {y_robot}, {z_robot})")
                    
                    # Move the robot arm to the detected mouth position
                    bot.arm.set_ee_pose_components(x=x_robot/100, y=y_robot/100, z=z_robot/100)

                    # Visualize the centroid on the image
                    cv2.circle(color_image, (cx, cy), 5, (0, 0, 255), -1)

        # Display the image
        cv2.imshow("Real-time Video", color_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    pipeline.stop()
    cv2.destroyAllWindows()
    bot.shutdown()

