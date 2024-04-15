import cv2
import mediapipe as mp
import numpy as np
import pyrealsense2 as rs
from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS
import logging
# Setup logging
logging.basicConfig(level=logging.INFO)

# Initialize MediaPipe Face Mesh
face_mesh = mp.solutions.face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Initialize the robot
bot = InterbotixManipulatorXS(robot_model='wx250s', group_name='arm', gripper_name='gripper')

# Setup RealSense camera
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


def transform_coordinates(x, y, z):
    # Transformation parameters
    scale = 1  # Scale factor from normalized to meters
    x_offset = 0.4  # Offset in x
    y_offset = 0  # Offset in y
    z_offset = 0.03  # Offset in z
    # Identity rotation matrix
    rotation_matrix = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])
    point_camera = np.array([x * scale, y * scale, z * scale])
    point_robot = np.dot(rotation_matrix, point_camera) + np.array([x_offset, y_offset, z_offset])
    return point_robot

def get_depth_from_coordinates(depth_frame, x, y):
    depth = depth_frame.get_distance(x, y)
    return depth * 1000  # Convert meters to millimeters if necessary

try:
    print("Press spacebar to capture and process an image.")
    while True:
        frames = pipeline.wait_for_frames(10000000)
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        if not depth_frame or not color_frame:
            continue

        frame = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        cv2.imshow("Real-time Video", frame)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break

        if key & 0xFF == ord(' '):
            results = face_mesh.process(image_rgb)
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    sum_x, sum_y, sum_z = 0, 0, 0
                    num_landmarks = 0
                    for idx in mp.solutions.face_mesh.FACEMESH_LIPS:
                        landmark1 = face_landmarks.landmark[idx[0]]
                        landmark2 = face_landmarks.landmark[idx[1]]
                        sum_x += landmark1.x + landmark2.x
                        sum_y += landmark1.y + landmark2.y
                        sum_z += landmark1.z + landmark2.z
                        num_landmarks += 2

                    if num_landmarks > 0:
                        centroid_x = sum_x / num_landmarks
                        centroid_y = sum_y / num_landmarks
                        centroid_z = sum_z / num_landmarks

                        # Get depth at centroid
                        depth = get_depth_from_coordinates(depth_frame, int(centroid_x * frame.shape[1]), int(centroid_y * frame.shape[0]))

                        # Transform coordinates
                        x_robot, y_robot, z_robot = transform_coordinates(centroid_x, centroid_y, depth)
                        print(f"Transformed coordinates: {x_robot}, {y_robot}, {z_robot}")

                        # Move the robotic arm
                        bot.arm.set_ee_pose_components(x=x_robot, y=y_robot, z=z_robot)
                        print(f"Moved to: {x_robot}, {y_robot}, {z_robot}")

                        # Draw a red circle at the centroid
                        cv2.circle(frame, (int(centroid_x * frame.shape[1]), int(centroid_y * frame.shape[0])), 5, (0, 0, 255), -1)
                        cv2.imshow("Real-time Video", frame)
                        print("Displayed centroid on video.")
            else:
                print("No face landmarks detected.")
finally:
    pipeline.stop()
    cv2.destroyAllWindows()
    bot.shutdown()

