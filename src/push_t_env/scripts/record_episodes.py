#!/usr/bin/env python3
import rospy
import os
import json
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import cv2
import imageio
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import String
from cv_bridge import CvBridge
from huggingface_hub import HfApi, hf_hub_download  # <-- Added hf_hub_download
from dotenv import load_dotenv
from trajectory_msgs.msg import JointTrajectory

class LeRobotDatasetRecorder:
    def __init__(self):
        rospy.init_node('lerobot_dataset_recorder', anonymous=True)
        
        # --- Load Environment Variables ---
        load_dotenv()
        self.hf_token = os.getenv("HF_TOKEN")
        if not self.hf_token:
            rospy.logwarn("HF_TOKEN not found in .env file. Upload to Hugging Face will likely fail.")
        
        # --- Config ---
        self.repo_id = rospy.get_param("~repo_id", "iFaz/gazebo_push_box_3dof_v2")
        self.fps = 10.0
        self.img_size = (512, 512)
        self.dataset_dir = os.path.join(os.path.expanduser("~"), "lerobot_dataset")
        self.chunk_size = 1000
        
        # --- Fetch Remote Dataset State for Appending ---
        self.remote_episodes = 0
        self.remote_frames = 0
        self.check_remote_dataset()
        
        # --- State Variables ---
        self.is_recording = False
        
        self.episode_idx = self.remote_episodes          # <-- Start from remote total
        self.total_frames_recorded = self.remote_frames  # <-- Resume from remote cumulative frame count


        self.current_episode_data = []
        self.current_episode_frames = []
        self.latest_img = None
        self.latest_joints = None
        self.latest_command = None 
        self.bridge = CvBridge()
        
        # --- Setup Directories ---
        self.data_dir = os.path.join(self.dataset_dir, "data", "chunk-000")
        self.video_dir = os.path.join(self.dataset_dir, "videos", "chunk-000", "observation.image")
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.video_dir, exist_ok=True)

        # --- Subscribers ---
        rospy.Subscriber("/push_t/overhead_camera/image_raw", Image, self.img_cb, queue_size=1)
        rospy.Subscriber("/joint_states", JointState, self.joint_cb, queue_size=1)
        rospy.Subscriber("/arm_ctrl/command", JointTrajectory, self.command_trajectory_cb, queue_size=1)
        rospy.Subscriber("/dataset_command", String, self.command_cb, queue_size=10)

        # --- Recording Loop ---
        rospy.Timer(rospy.Duration(1.0 / self.fps), self.record_step)
        rospy.loginfo("Dataset Recorder Backend Ready. Waiting for commands from GUI Panel...")
    
    def check_remote_dataset(self):
        """Checks Hugging Face for an existing dataset and fetches current progress metadata."""
        if not self.hf_token:
            return
        try:
            api = HfApi(token=self.hf_token)
            if api.repo_exists(repo_id=self.repo_id, repo_type="dataset"):
                rospy.loginfo(f"Remote repository '{self.repo_id}' exists. Fetching existing metadata...")
                try:
                    info_path = hf_hub_download(
                        repo_id=self.repo_id,
                        filename="info.json",
                        repo_type="dataset",
                        token=self.hf_token
                    )
                    with open(info_path, 'r') as f:
                        remote_info = json.load(f)
                    
                    self.remote_episodes = remote_info.get("total_episodes", 0)
                    self.remote_frames = remote_info.get("total_frames", 0)
                    rospy.loginfo(f"Successfully loaded remote metadata: {self.remote_episodes} episodes, {self.remote_frames} frames already exist. Appending active.")
                except Exception as e:
                    rospy.logwarn(f"Could not locate or parse 'info.json' on remote repo (it might be empty). Initializing clean: {e}")
            else:
                rospy.loginfo(f"Remote repository '{self.repo_id}' does not exist yet. Initializing fresh dataset workspace.")
        except Exception as e:
            rospy.logerr(f"Failed to establish cloud synchronization with Hugging Face Hub: {e}")

    def command_trajectory_cb(self, msg):
        if msg.points:
            self.latest_command = np.array(msg.points[0].positions[:3], dtype=np.float32)
    
    def img_cb(self, msg):
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.latest_img = cv2.resize(cv_img, self.img_size)
        except Exception as e:
            rospy.logerr(f"Image Error: {e}")

    def joint_cb(self, msg):
        if len(msg.position) >= 3:
            self.latest_joints = np.array(msg.position[:3], dtype=np.float32)

    def command_cb(self, msg):
        if msg.data == "TOGGLE_RECORD":
            self.toggle_record()
        elif msg.data == "UPLOAD":
            self.upload_dataset()

    def record_step(self, event):
        if not self.is_recording or self.latest_img is None or self.latest_joints is None or self.latest_command is None:
            return

        frame_idx = len(self.current_episode_data)
        timestamp = frame_idx / self.fps
        
        self.current_episode_data.append({
            "observation.state": self.latest_joints.tolist(),
            "action": self.latest_command.tolist(), 
            "episode_index": self.episode_idx,
            "frame_index": frame_idx,
            "timestamp": timestamp,
            "next.reward": 0.0, 
            "next.done": False,
            "next.success": False,
            "task_index": 0
        })
        self.current_episode_frames.append(cv2.cvtColor(self.latest_img, cv2.COLOR_BGR2RGB))
    
    def toggle_record(self):
        self.is_recording = not self.is_recording
        
        if self.is_recording:
            rospy.loginfo(f"--- STARTED RECORDING EPISODE {self.episode_idx} ---")
            self.current_episode_data = []
            self.current_episode_frames = []
        else:
            rospy.loginfo(f"--- STOPPED RECORDING. SAVING EPISODE {self.episode_idx} ---")
            self.save_episode()
            self.episode_idx += 1

    def save_episode(self):
        if not self.current_episode_data:
            rospy.logwarn("No data collected. Skipping save.")
            return

        n_frames = len(self.current_episode_data)
        for i in range(n_frames):
            self.current_episode_data[i]["index"] = self.total_frames_recorded + i
            if i == n_frames - 1:
                self.current_episode_data[i]["next.done"] = True
                self.current_episode_data[i]["next.success"] = True 

        self.total_frames_recorded += n_frames

        # Save Parquet
        df = pd.DataFrame(self.current_episode_data)
        parquet_path = os.path.join(self.data_dir, f"episode_{self.episode_idx:06d}.parquet")
        table = pa.Table.from_pandas(df)
        pq.write_table(table, parquet_path)

        # Save Video
        video_path = os.path.join(self.video_dir, f"episode_{self.episode_idx:06d}.mp4")
        writer = imageio.get_writer(video_path, fps=self.fps, codec='libx264', macro_block_size=None)
        for frame in self.current_episode_frames:
            writer.append_data(frame)
        writer.close()

        rospy.loginfo(f"Saved Episode {self.episode_idx} | {n_frames} frames.")

    def upload_dataset(self):
        if not self.hf_token:
            rospy.logerr("Cannot upload: HF_TOKEN is missing. Please set it in your .env file.")
            return

        rospy.loginfo("--- GENERATING info.json AND UPLOADING TO HUGGING FACE ---")
        
        info = {
            "codebase_version": "v2.0",
            "robot_type": "3dof_planar_arm",
            "total_episodes": self.episode_idx,
            "total_frames": self.total_frames_recorded,
            "total_tasks": 1,
            "total_videos": self.episode_idx,
            "total_chunks": 1,
            "chunks_size": self.chunk_size,
            "fps": self.fps,
            "splits": {"train": f"0:{self.episode_idx}"},
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            "features": {
                "observation.image": {
                    "dtype": "video",
                    "shape": [self.img_size[1], self.img_size[0], 3],
                    "names": ["height", "width", "channel"],
                    "video_info": {
                        "video.fps": self.fps,
                        "video.codec": "h264",
                        "video.pix_fmt": "yuv420p",
                        "video.is_depth_map": False,
                        "has_audio": False
                    }
                },
                "observation.state": {
                    "dtype": "float32",
                    "shape": [3],
                    "names": {"motors": ["joint_1", "joint_2", "joint_3"]}
                },
                "action": {
                    "dtype": "float32",
                    "shape": [3],
                    "names": {"motors": ["joint_1", "joint_2", "joint_3"]}
                },
                "episode_index": {"dtype": "int64", "shape": [1], "names": None},
                "frame_index": {"dtype": "int64", "shape": [1], "names": None},
                "timestamp": {"dtype": "float32", "shape": [1], "names": None},
                "next.reward": {"dtype": "float32", "shape": [1], "names": None},
                "next.done": {"dtype": "bool", "shape": [1], "names": None},
                "next.success": {"dtype": "bool", "shape": [1], "names": None},
                "index": {"dtype": "int64", "shape": [1], "names": None},
                "task_index": {"dtype": "int64", "shape": [1], "names": None}
            }
        }

        with open(os.path.join(self.dataset_dir, "info.json"), "w") as f:
            json.dump(info, f, indent=4)

        try:
            api = HfApi(token=self.hf_token)
            api.create_repo(repo_id=self.repo_id, repo_type="dataset", exist_ok=True)
            
            # upload_folder merges tracking data automatically and preserves un-targeted files.
            api.upload_folder(
                folder_path=self.dataset_dir,
                repo_id=self.repo_id,
                repo_type="dataset"
            )
            rospy.loginfo(f"SUCCESS! Dataset appended and uploaded to https://huggingface.co/datasets/{self.repo_id}")
        except Exception as e:
            rospy.logerr(f"Upload Failed: {e}")

if __name__ == '__main__':
    try:
        LeRobotDatasetRecorder()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass




#### IF separte episode recording/insertion is necessary

# #!/usr/bin/env python3
# import rospy
# import os
# import json
# import numpy as np
# import pandas as pd
# import pyarrow as pa
# import pyarrow.parquet as pq
# import cv2
# import imageio
# from sensor_msgs.msg import Image, JointState
# from std_msgs.msg import String
# from cv_bridge import CvBridge
# from huggingface_hub import HfApi, hf_hub_download
# from dotenv import load_dotenv
# from trajectory_msgs.msg import JointTrajectory

# class LeRobotDatasetRecorder:
#     def __init__(self):
#         rospy.init_node('lerobot_dataset_recorder', anonymous=True)
        
#         # --- Load Environment Variables ---
#         load_dotenv()
#         self.hf_token = os.getenv("HF_TOKEN")
#         if not self.hf_token:
#             rospy.logwarn("HF_TOKEN not found in .env file. Upload to Hugging Face will likely fail.")
        
#         # --- Config ---
#         self.repo_id = rospy.get_param("~repo_id", "iFaz/gazebo_push_t_3dof_v1")
#         self.fps = 10.0
#         self.img_size = (96, 96)
#         self.dataset_dir = os.path.join(os.path.expanduser("~"), "lerobot_dataset")
#         self.chunk_size = 1000
        
#         # =========================================================================
#         # 🎯 CONFIGURABLE TARGET SELECTION
#         # Change this value to target ANY single episode slot you want to fix/insert!
#         # =========================================================================
#         self.target_episode = 15
#         # =========================================================================

#         # --- Fetch Remote Dataset State for Tracking Cumulative Totals ---
#         self.remote_episodes = 0
#         self.remote_frames = 0
#         self.check_remote_dataset()
        
#         # --- State Variables ---
#         self.is_recording = False
#         self.episode_idx = self.target_episode         
#         self.total_frames_recorded = 0

#         self.current_episode_data = []
#         self.current_episode_frames = []
#         self.latest_img = None
#         self.latest_joints = None
#         self.latest_command = None 
#         self.bridge = CvBridge()
        
#         # --- Setup Directories ---
#         self.data_dir = os.path.join(self.dataset_dir, "data", "chunk-000")
#         self.video_dir = os.path.join(self.dataset_dir, "videos", "chunk-000", "observation.image")
#         os.makedirs(self.data_dir, exist_ok=True)
#         os.makedirs(self.video_dir, exist_ok=True)

#         # --- Subscribers ---
#         rospy.Subscriber("/push_t/overhead_camera/image_raw", Image, self.img_cb, queue_size=1)
#         rospy.Subscriber("/joint_states", JointState, self.joint_cb, queue_size=1)
#         rospy.Subscriber("/arm_ctrl/command", JointTrajectory, self.command_trajectory_cb, queue_size=1)
#         rospy.Subscriber("/dataset_command", String, self.command_cb, queue_size=10)

#         # --- Recording Loop ---
#         rospy.Timer(rospy.Duration(1.0 / self.fps), self.record_step)
#         rospy.loginfo(f"Dataset Recorder Patching Mode Ready! Locked on Target Episode: {self.target_episode}")
    
#     def check_remote_dataset(self):
#         """Checks Hugging Face for an existing dataset and fetches current progress metadata."""
#         if not self.hf_token:
#             return
#         try:
#             api = HfApi(token=self.hf_token)
#             if api.repo_exists(repo_id=self.repo_id, repo_type="dataset"):
#                 rospy.loginfo(f"Remote repository '{self.repo_id}' exists. Fetching existing metadata...")
#                 try:
#                     info_path = hf_hub_download(
#                         repo_id=self.repo_id,
#                         filename="info.json",
#                         repo_type="dataset",
#                         token=self.hf_token
#                     )
#                     with open(info_path, 'r') as f:
#                         remote_info = json.load(f)
                    
#                     self.remote_episodes = remote_info.get("total_episodes", 0)
#                     self.remote_frames = remote_info.get("total_frames", 0)
#                     rospy.loginfo(f"Successfully loaded remote metadata: {self.remote_episodes} episodes, {self.remote_frames} frames already exist.")
#                 except Exception as e:
#                     rospy.logwarn(f"Could not locate or parse 'info.json' on remote repo: {e}")
#             else:
#                 rospy.loginfo(f"Remote repository '{self.repo_id}' does not exist yet.")
#         except Exception as e:
#             rospy.logerr(f"Failed to establish cloud synchronization with Hugging Face Hub: {e}")

#     def command_trajectory_cb(self, msg):
#         if msg.points:
#             self.latest_command = np.array(msg.points[0].positions[:3], dtype=np.float32)
    
#     def img_cb(self, msg):
#         try:
#             cv_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
#             self.latest_img = cv2.resize(cv_img, self.img_size)
#         except Exception as e:
#             rospy.logerr(f"Image Error: {e}")

#     def joint_cb(self, msg):
#         if len(msg.position) >= 3:
#             self.latest_joints = np.array(msg.position[:3], dtype=np.float32)

#     def command_cb(self, msg):
#         if msg.data == "TOGGLE_RECORD":
#             self.toggle_record()
#         elif msg.data == "UPLOAD":
#             self.upload_dataset()

#     def record_step(self, event):
#         if not self.is_recording or self.latest_img is None or self.latest_joints is None or self.latest_command is None:
#             return

#         frame_idx = len(self.current_episode_data)
#         timestamp = frame_idx / self.fps
        
#         self.current_episode_data.append({
#             "observation.state": self.latest_joints.tolist(),
#             "action": self.latest_command.tolist(), 
#             "episode_index": self.episode_idx,
#             "frame_index": frame_idx,
#             "timestamp": timestamp,
#             "next.reward": 0.0, 
#             "next.done": False,
#             "next.success": False,
#             "task_index": 0
#         })
#         self.current_episode_frames.append(cv2.cvtColor(self.latest_img, cv2.COLOR_BGR2RGB))
    
#     def toggle_record(self):
#         self.is_recording = not self.is_recording
        
#         if self.is_recording:
#             rospy.loginfo(f"--- STARTED RECORDING TARGET EPISODE {self.episode_idx} ---")
#             self.current_episode_data = []
#             self.current_episode_frames = []
#         else:
#             rospy.loginfo(f"--- STOPPED RECORDING. SAVING TARGET EPISODE {self.episode_idx} ---")
#             self.save_episode()

#     def save_episode(self):
#         if not self.current_episode_data:
#             rospy.logwarn("No data collected. Skipping save.")
#             return

#         n_frames = len(self.current_episode_data)
#         for i in range(n_frames):
#             self.current_episode_data[i]["index"] = self.total_frames_recorded + i
#             if i == n_frames - 1:
#                 self.current_episode_data[i]["next.done"] = True
#                 self.current_episode_data[i]["next.success"] = True 

#         self.total_frames_recorded += n_frames

#         # Save Parquet using targeted formatting
#         df = pd.DataFrame(self.current_episode_data)
#         parquet_path = os.path.join(self.data_dir, f"episode_{self.episode_idx:06d}.parquet")
#         table = pa.Table.from_pandas(df)
#         pq.write_table(table, parquet_path)

#         # Save Video using targeted formatting
#         video_path = os.path.join(self.video_dir, f"episode_{self.episode_idx:06d}.mp4")
#         writer = imageio.get_writer(video_path, fps=self.fps, codec='libx264', macro_block_size=None)
#         for frame in self.current_episode_frames:
#             writer.append_data(frame)
#         writer.close()

#         rospy.loginfo(f"Saved Target Episode {self.episode_idx} | {n_frames} frames.")

#     def upload_dataset(self):
#         if not self.hf_token:
#             rospy.logerr("Cannot upload: HF_TOKEN is missing. Please set it in your .env file.")
#             return

#         # Target file resolution matching the configured episode
#         target_filename = f"episode_{self.target_episode:06d}"
#         parquet_path = os.path.join(self.data_dir, f"{target_filename}.parquet")
        
#         if not os.path.exists(parquet_path):
#             rospy.logerr(f"Local file {parquet_path} not found. Please record target data before uploading.")
#             return

#         # 1. Read the newly recorded local frame count for this targeted index
#         new_frames = pq.read_metadata(parquet_path).num_rows
#         rospy.loginfo(f"Detected {new_frames} frames in your local targeted {target_filename} patch.")

#         # 2. Re-calculate dataset properties
#         # If your target episode fits inside the remote total scope, use the cloud total.
#         # Otherwise, assume we are appending an entirely fresh endpoint index.
#         total_episodes_fixed = max(self.remote_episodes, 21) 
        
#         if self.remote_frames > 0:
#             total_frames_calculated = self.remote_frames + new_frames
#         else:
#             total_frames_calculated = 23705 + new_frames # Fallback safe baseline calculation
            
#         rospy.loginfo(f"--- RE-GENERATING FIXED info.json FOR {total_episodes_fixed} TOTAL EPISODES ---")
#         rospy.loginfo(f"Combined Dynamic Frame Count: {total_frames_calculated}")

#         info = {
#             "codebase_version": "v2.0",
#             "robot_type": "3dof_planar_arm",
#             "total_episodes": total_episodes_fixed,
#             "total_frames": total_frames_calculated,
#             "total_tasks": 1,
#             "total_videos": total_episodes_fixed,
#             "total_chunks": 1,
#             "chunks_size": self.chunk_size,
#             "fps": self.fps,
#             "splits": {"train": f"0:{total_episodes_fixed}"},
#             "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
#             "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
#             "features": {
#                 "observation.image": {
#                     "dtype": "video",
#                     "shape": [96, 96, 3],
#                     "names": ["height", "width", "channel"],
#                     "video_info": {
#                         "video.fps": self.fps,
#                         "video.codec": "h264",
#                         "video.pix_fmt": "yuv420p",
#                         "video.is_depth_map": False,
#                         "has_audio": False
#                     }
#                 },
#                 "observation.state": {
#                     "dtype": "float32",
#                     "shape": [3],
#                     "names": {"motors": ["joint_1", "joint_2", "joint_3"]}
#                 },
#                 "action": {
#                     "dtype": "float32",
#                     "shape": [3],
#                     "names": {"motors": ["joint_1", "joint_2", "joint_3"]}
#                 },
#                 "episode_index": {"dtype": "int64", "shape": [1], "names": None},
#                 "frame_index": {"dtype": "int64", "shape": [1], "names": None},
#                 "timestamp": {"dtype": "float32", "shape": [1], "names": None},
#                 "next.reward": {"dtype": "float32", "shape": [1], "names": None},
#                 "next.done": {"dtype": "bool", "shape": [1], "names": None},
#                 "next.success": {"dtype": "bool", "shape": [1], "names": None},
#                 "index": {"dtype": "int64", "shape": [1], "names": None},
#                 "task_index": {"dtype": "int64", "shape": [1], "names": None}
#             }
#         }

#         with open(os.path.join(self.dataset_dir, "info.json"), "w") as f:
#             json.dump(info, f, indent=4)

#         try:
#             api = HfApi(token=self.hf_token)
#             api.create_repo(repo_id=self.repo_id, repo_type="dataset", exist_ok=True)
            
#             # --- DYNAMIC ALLOW PATTERNS ---
#             # Automatically targets ONLY info.json and your specific numbered episode data/video files
#             allow_patterns = [
#                 "info.json",
#                 f"**/{target_filename}.parquet",
#                 f"**/{target_filename}.mp4"
#             ]

#             rospy.loginfo(f"Uploading target {target_filename} patch and updated info.json layout...")
#             api.upload_folder(
#                 folder_path=self.dataset_dir,
#                 repo_id=self.repo_id,
#                 repo_type="dataset",
#                 allow_patterns=allow_patterns
#             )
#             rospy.loginfo(f"SUCCESS! Target dataset slot {self.target_episode} successfully patched on remote repo.")
#         except Exception as e:
#             rospy.logerr(f"Upload Failed: {e}")

# if __name__ == '__main__':
#     try:
#         LeRobotDatasetRecorder()
#         rospy.spin()
#     except rospy.ROSInterruptException:
#         pass
