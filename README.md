# Vine tomato grasping
This repository contains the code used for the paper [Geometry-Based Grasping of Vine Tomatoes](https://arxiv.org/pdf/2103.01272.pdf). It consists of two parts:

1. **Truss detection**: extract truss features from a given image and determined an optimal grasping pose. This is done in the ROS independent [`flex_vision`](/flex_vision) package.
2. **Truss manipulation**: Manipulator control to grasp vine tomatoes. This is done in the [`flex_grasp`](/flex_grasp) package.


## Contents
The contents of this repository are listed below:

### ROS packages
- **[flex_calibrate](/flex_calibrate)**: Manipulator calibration
- **[flex_gazebo](/flex_gazebo)**: General Gazebo simulation, containing files such as camera and marker.
- **[flex_gazebo_iiwa](/flex_gazebo_iiwa)**: KUKA LBR iiwa Gazebo simulation.
- **[flex_gazebo_interbotix](/flex_gazebo_interbotix)**: InterbotiX Gazebo simulation.
- **[flex_grasp](/flex_grasp)**: Manipulator control to grasp vine tomatoes. Contains all the ROS nodes.
- **[flex_sdh_moveit](/flex_sdh_moveit)**: MoveIt support for SCHUNK Dextrous Hand (SDH) manipulator control.
- **[rqt_user_interface](/rqt_user_interface)**: rqt GUI to command the robot, and change settings.
- **[flex_shared_resources](/flex_shared_resources)**: Shared utility resources, used by various packages

### Others

- **[flex_vision](/flex_vision)**: Computer vision pipeline for Python (ROS independent)
