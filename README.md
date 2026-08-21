# 3-DOF Arm Imitation Learning via Diffusion Policy


[![Dataset Converter](https://img.shields.io/badge/Kaggle-Dataset_Converter-20BEFF?logo=Kaggle)](https://www.kaggle.com/code/kaziabrarmahmud/ros-to-lerobot-port)
[![Inference Server](https://img.shields.io/badge/Kaggle-Inference_Server-20BEFF?logo=Kaggle)](https://www.kaggle.com/code/kaziabrarmahmud/infer-server-lerobot-v1)

[![Project Demo](https://img.youtube.com/vi/DUMMY_VIDEO_ID/maxresdefault.jpg)](https://www.youtube.com/watch?v=DUMMY_VIDEO_ID)
*(Click the thumbnail above to watch the video demonstration. Replace `DUMMY_VIDEO_ID` with your actual YouTube video ID.)*


## Overview
This repository contains the pipeline for an imitation learning project featuring a custom 3-DOF linear arm. The robotic arm is simulated in Gazebo (built using custom URDF), where demonstration data is collected for behavioral cloning. A Diffusion Policy is subsequently trained on this data utilizing the Hugging Face [LeRobot](https://github.com/huggingface/lerobot) framework. Finally, the trained policy is deployed via an inference server to control the simulated robot autonomously.

## Architecture & Workflow

1. **Simulation (Gazebo/ROS):** 
   - A custom 3-DOF linear arm is designed and simulated.
   - Expert demonstrations are recorded to generate the initial imitation learning dataset.
2. **Data Processing:** 
   - The raw ROS/Gazebo data is parsed and converted into a standard dataset format required by the LeRobot framework.
3. **Diffusion Policy Training:** 
   - LeRobot is utilized to train a robust Diffusion Policy on the converted dataset, enabling the system to learn complex, multi-modal action distributions.
4. **Inference Deployment:** 
   - A dedicated inference server runs the trained policy, receiving state observations and outputting action trajectories to drive the simulation.

## How to Use

Follow these steps to replicate the data collection, training, and inference pipeline:

### 1. Build and Setup the ROS Workspace
Navigate to the root of your ROS workspace and build the packages:
```bash
catkin build
source devel/setup.bash
```

### 2. Data Collection
Launch the leader bot in the Gazebo simulated environment to collect expert demonstration data:
```bash
roslaunch push_t_env hardware_push_t_datagen.launch
```

### 3. Dataset Conversion
Use the Kaggle converter script to format the raw Gazebo dataset into a LeRobot-compatible format.
* **Tool:** [ros-to-lerobot-port](https://www.kaggle.com/code/kaziabrarmahmud/ros-to-lerobot-port)

### 4. Train the Diffusion Model
Train your diffusion policy on the converted dataset locally or via the cloud using the Hugging Face LeRobot framework.

### 5. Start the Inference Server
Deploy your trained model by starting the inference server. This server receives states and serves action predictions.
* **Tool:** [infer-server-lerobot-v1](https://www.kaggle.com/code/kaziabrarmahmud/infer-server-lerobot-v1)

### 6. Run the Inference Environment
Finally, launch the inference environment locally in Gazebo to observe the trained policy controlling the robot autonomously:
```bash
roslaunch push_t_env push_t_infer_env.launch
```

## Links & Resources

*   **Dataset Converter (ROS to LeRobot):**  
    [Kaggle Notebook: ros-to-lerobot-port](https://www.kaggle.com/code/kaziabrarmahmud/ros-to-lerobot-port)  
    *This script handles the conversion of raw demonstration data generated from the Gazebo environment into the specific dataset structure expected by LeRobot.*

*   **Inference Server:**  
    [Kaggle Notebook: infer-server-lerobot-v1](https://www.kaggle.com/code/kaziabrarmahmud/infer-server-lerobot-v1)  
    *The inference server code responsible for running the trained Diffusion Policy and serving action predictions during deployment.*

## Author
**Kazi Abrar Mahmud**