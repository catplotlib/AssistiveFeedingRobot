import cv2
import mediapipe as mp
import numpy as np
from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS

# Initialize MediaPipe Face Mesh
face_mesh = mp.solutions.face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Initialize the robot
bot = InterbotixManipulatorXS(robot_model='wx250s', group_name='arm', gripper_name='gripper')

def transform_coordinates(x, y, z):
    # Transformation parameters
    scale = 1 # Scale factor from normalized to meters
    x_offset = 0  # Example offsets
    y_offset = 0
    z_offset = 0.2
    # Rotation matrix for coordinate alignment
    rotation_matrix = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])
    point_camera = np.array([x * scale, y * scale, z * scale])
    point_robot = np.dot(rotation_matrix, point_camera)
    point_robot += np.array([x_offset, y_offset, z_offset])
    return point_robot

# Open video capture
cap = cv2.VideoCapture(0)
try:
    print("Press spacebar to capture and process an image.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image")
            continue
        cv2.imshow("Real-time Video", frame)
        key = cv2.waitKey(1)        if key & 0xFF == ord('q'):  # Press 'q' to quit
            break

        if key & 0xFF == ord(' '):  # Process the image when spacebar is pressed
            print("Processing image...")
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(image_rgb)
            print("Image processed.")

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
                        print(f"Centroid in image coordinates: {centroid_x}, {centroid_y}, {centroid_z}")

                        # Transform coordinates
                        x_robot, y_robot, z_robot = transform_coordinates(centroid_x, centroid_y, centroid_z)
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
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    bot.shutdown()
