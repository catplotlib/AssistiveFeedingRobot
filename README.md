# RSS Project - Robotic Assistive Feeding System

https://github.com/anvithaanchala/RSS/assets/61319491/f5aef311-ad0e-4e57-8658-c240852c96bf

## Overview
This project focuses on developing a robotic arm designed to help individuals with upper limb impairments. 
The goal is to improve the flexibility and accuracy of assistive feeding devices, ultimately helping users become more independent.

## Robot Manipulation Workflow
The robotic arm manipulation is a meticulously designed sequence aimed at performing precise tasks essential for assistive feeding. Here’s how the workflow is structured:

1. **Picking of Spoon from a Fixed Location:** The robot is programmed to start the feeding process by accurately retrieving a spoon from a predetermined position. This ensures that the starting point is consistent, which is crucial for the subsequent steps.
2. **Scooping of Food:** Once the spoon is secured, the robot proceeds to scoop food. This step involves precise control and movements to ensure that an adequate amount of food is gathered on the spoon without spillage.
3. **Moving the End Effector to the Mouth:** After scooping the food, the robot arm moves the spoon towards the user's mouth. The position of the mouth is determined through real-time facial landmark detection using the camera system. Coordinates are transformed and scaled to guide the spoon accurately to the mouth, ensuring a comfortable and safe feeding experience.

## Imaging Workflow
The imaging workflow is crucial for the effective operation of the assistive feeding system. It utilizes a combination of hardware and software technologies to detect and track the user's facial features, specifically the mouth, to guide the robotic arm accurately. Here's an overview of the workflow:

1. **Camera Setup:** Utilizes the Intel Realsense D455i, a depth camera that captures high-quality images and depth information crucial for precise tracking.
2. **Facial Landmark Detection:** The Google Mediapipe library is employed to detect facial landmarks in real-time. This library provides robust and accurate facial feature recognition capabilities.
3. **Detection of Mouth Coordinates:** The system specifically looks for the coordinates of the mouth in the camera frame. This step is vital as it determines the target location where the spoon will deliver the food.
4. **Coordinate Transformation:** Once the mouth coordinates are detected, they are calculated and transformed according to the base frame of the robotic system, which in this case is the WIdowX250s robot arm. This transformation is essential for accurately aligning the robot’s movements with the user's position.
5. **Application of Coordinates:** The transformed coordinates are then applied to the robotic arm, guiding it to move precisely to the user’s mouth with the scooped food.

## Installation Requirements
- **Hardware:** Intel Realsense D455i camera, WidowX250s robotic arm.
- **Software:** Python 3.8 or later, Mediapipe, other dependencies listed in `requirements.txt`.
