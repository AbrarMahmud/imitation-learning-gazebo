#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script 1 — ROS 1 / Python 3.8 — LeRobot Inference Bridge
=========================================================
Collects observations from Gazebo (image + joint states) and sends them
to the Kaggle/Colab inference server (Script 2).  Publishes the returned
actions as JointTrajectory commands.

Topics
------
  Subscribed:
    /push_t/overhead_camera/image_raw  (sensor_msgs/Image)
    /joint_states                      (sensor_msgs/JointState)

  Published:
    /arm_ctrl/command                  (trajectory_msgs/JointTrajectory)

Server API
----------
  POST /infer  { "observation.state": [...], "observation.image": "<b64 png>" }
               → { "action": [...] }

  POST /reset  (call on episode reset to clear the policy's obs queue)
  GET  /health (liveness check)

NOTE: image encoding matches the verified-working reference client
(pusht_remote_infer_client.py) — plain PIL PNG encode of the RGB frame,
with no BGR round-trip. See _pack_observation() below.
"""

import os
os.environ["PYTHONUNBUFFERED"] = "1"

import base64
import io
import threading
import time
from typing import Optional

import cv2
import numpy as np
import requests
import rospy
from cv_bridge import CvBridge
from PIL import Image as PILImage
from sensor_msgs.msg import Image, JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

# ── Constants ────────────────────────────────────────────────────────────────
IMG_SIZE        = (512, 512)   # (W, H) for cv2.resize  ← note cv2 is (W, H)
STATE_DIM       = 3
REQUEST_TIMEOUT = 10.0        # seconds; diffusion is slow on first call

NGROK_HEADERS   = {"ngrok-skip-browser-warning": "true"}  # ← add this


def encode_image_b64(frame: np.ndarray) -> str:
    """
    Encode an HWC RGB uint8 numpy frame as a base64 PNG string.

    Matches the reference client exactly: PIL.Image.fromarray() on the
    RGB array, saved straight to PNG. No cv2 BGR<->RGB round-trip — that
    extra conversion isn't needed (PIL writes channels in the order given)
    and is a needless place for a channel-swap bug to creep in.
    """
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    img = PILImage.fromarray(frame)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


class ROSInferenceBridge:

    def __init__(self, server: str, fps: int,
                 img_topic: str, state_topic: str, cmd_topic: str):

        self.server_url  = server.rstrip("/")
        self.target_fps  = fps
        self.img_topic   = img_topic
        self.state_topic = state_topic
        self.cmd_topic   = cmd_topic

        self._latest_image: Optional[np.ndarray] = None
        self._latest_state: Optional[np.ndarray] = None
        self._lock   = threading.Lock()
        self._bridge = CvBridge()

        self._cmd_pub = rospy.Publisher(
            self.cmd_topic, JointTrajectory, queue_size=1)
        self._joint_names = ["joint_1", "joint_2", "joint_3"]

        rospy.Subscriber(self.img_topic,   Image,      self._image_cb, queue_size=1)
        rospy.Subscriber(self.state_topic, JointState, self._state_cb, queue_size=1)

        rospy.loginfo("=== LeRobot Inference Bridge ===")
        rospy.loginfo(f"  server      : {self.server_url}")
        rospy.loginfo(f"  img_topic   : {self.img_topic}")
        rospy.loginfo(f"  state_topic : {self.state_topic}")
        rospy.loginfo(f"  cmd_topic   : {self.cmd_topic}")
        rospy.loginfo(f"  fps         : {self.target_fps}")

    # ── ROS callbacks ────────────────────────────────────────────────────────

    def _image_cb(self, msg: Image):
        try:
            # decode straight to RGB — this stays RGB all the way through
            # to encode_image_b64(), same as the reference client's
            # gym "pixels" observation (also RGB).
            cv_img  = self._bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
            # cv2.resize expects (W, H); resizing doesn't touch channel order
            resized = cv2.resize(cv_img, IMG_SIZE, interpolation=cv2.INTER_LINEAR)
            with self._lock:
                self._latest_image = resized.astype(np.uint8)
        except Exception as e:
            rospy.logwarn(f"Image callback error: {e}")

    def _state_cb(self, msg: JointState):
        try:
            positions = np.array(msg.position[:STATE_DIM], dtype=np.float32)
            with self._lock:
                self._latest_state = positions
        except Exception as e:
            rospy.logwarn(f"State callback error: {e}")

    # ── Observation packing ──────────────────────────────────────────────────

    def _pack_observation(self) -> Optional[dict]:
        """
        Returns a JSON-serialisable dict with the current image (base64 PNG)
        and joint state.  Returns None if either sensor hasn't fired yet.

        Payload shape matches the server contract exactly (same two keys
        the reference client sends): "observation.state" and
        "observation.image". No extra keys.
        """
        with self._lock:
            if self._latest_image is None or self._latest_state is None:
                return None
            image = self._latest_image.copy()
            state = self._latest_state.copy()

        try:
            image_b64 = encode_image_b64(image)
        except Exception as e:
            rospy.logwarn(f"Failed to encode image as PNG: {e}")
            return None

        return {
            "observation.state": np.asarray(state, dtype=np.float32).tolist(),
            "observation.image": image_b64,
        }

    # ── Server communication ─────────────────────────────────────────────────
    def _send_observation(self, obs: dict):
        try:
            r = requests.post(
                f"{self.server_url}/infer",
                json=obs,
                timeout=REQUEST_TIMEOUT,
                headers=NGROK_HEADERS,
            )
            if not r.ok:
                # Surface the server's actual error message instead of a
                # bare "400/500 Client/Server Error" with no context —
                # matches the reference client's infer() behavior.
                try:
                    detail = r.json().get("error", r.text)
                except Exception:
                    detail = r.text
                rospy.logerr(f"{r.status_code} from /infer: {detail}")
                return None, 0

            payload = r.json()
            if "error" in payload:
                rospy.logerr(f"Server returned error: {payload['error']}")
                return None, 0
            action     = np.array(payload["action"], dtype=np.float32)
            queue_size = payload.get("queue_size", -1)   # -1 if server doesn't send it
            return action, queue_size
        except requests.exceptions.Timeout:
            rospy.logwarn("Inference request timed out")
        except requests.exceptions.ConnectionError:
            rospy.logerr(f"Cannot reach server: {self.server_url}")
        except Exception as e:
            rospy.logwarn(f"Inference request failed: {e}")
        return None, 0

    def _reset_policy(self):
        """
        Tell the server to clear the policy's internal observation/action queues.
        Call this at node start-up (and whenever Gazebo resets an episode).
        """
        try:
            r = requests.post(
                f"{self.server_url}/reset", timeout=5.0, headers=NGROK_HEADERS)
            r.raise_for_status()
            rospy.loginfo("Policy queues reset on server.")
        except Exception as e:
            rospy.logwarn(f"Could not reset policy: {e}")

    def _health_check(self) -> bool:
        """Returns True if the server is reachable and healthy."""
        try:
            r = requests.get(
                f"{self.server_url}/health", timeout=5.0, headers=NGROK_HEADERS)
            r.raise_for_status()
            info = r.json()
            rospy.loginfo(
                f"Server healthy — model: {info.get('model')}  "
                f"device: {info.get('device')}  "
                f"image_keys: {info.get('image_keys')}"
            )
            return True
        except Exception as e:
            rospy.logwarn(f"Health check failed: {e}")
            return False

    # ── Action publishing ────────────────────────────────────────────────────

    def _publish_action(self, action: np.ndarray):
        msg             = JointTrajectory()
        msg.header.stamp = rospy.Time.now()
        msg.joint_names = self._joint_names
        pt              = JointTrajectoryPoint()
        pt.positions    = action.tolist()
        pt.velocities   = [0.0] * len(self._joint_names)
        pt.time_from_start = rospy.Duration(1.0 / self.target_fps)
        msg.points      = [pt]
        self._cmd_pub.publish(msg)
        rospy.logdebug(f"Action published: {action.tolist()}")

    # ── Main loop ────────────────────────────────────────────────────────────

    def run(self):
        # 1. Verify server is up
        rospy.loginfo("Checking server health …")
        while not rospy.is_shutdown():
            if self._health_check():
                break
            rospy.logwarn_throttle(10.0, "Waiting for inference server …")
            rospy.sleep(2.0)

        # 2. Reset the policy's internal queues (clears any stale obs from a
        #    previous run so the first inference sees a clean rolling window).
        self._reset_policy()

        # 3. Wait for first image + state
        rospy.loginfo("Waiting for first image + state …")
        while not rospy.is_shutdown():
            with self._lock:
                has_img   = self._latest_image is not None
                has_state = self._latest_state is not None
            if has_img and has_state:
                break
            rospy.logwarn_throttle(
                5.0, f"Still waiting — image:{has_img}  state:{has_state}")
            rospy.sleep(0.1)

        rospy.loginfo("Both observations received — inference loop running.")

        # 4. Inference loop
        rate = rospy.Rate(self.target_fps)
        while not rospy.is_shutdown():
            obs = self._pack_observation()
            if obs is None:
                rate.sleep()
                continue

            action, queue_size = self._send_observation(obs)   # ← unpack tuple
            if action is not None:
                self._publish_action(action)
                rospy.loginfo_throttle(
                    2.0, f"Action: {action.tolist()}  |  server_queue: {queue_size}")
            else:
                rospy.logwarn("No action received — skipping step")
            rate.sleep()

        rospy.loginfo("Bridge shutting down.")


if __name__ == "__main__":
    rospy.init_node("lerobot_inference_bridge", anonymous=False)

    server      = rospy.get_param("~server",      "https://578c-35-231-139-58.ngrok-free.app")
    fps         = int(rospy.get_param("~fps",      10))
    img_topic   = rospy.get_param("~img_topic",   "/push_t/overhead_camera/image_raw")
    state_topic = rospy.get_param("~state_topic", "/joint_states")
    cmd_topic   = rospy.get_param("~cmd_topic",   "/arm_ctrl/command")

    bridge = ROSInferenceBridge(server, fps, img_topic, state_topic, cmd_topic)
    try:
        bridge.run()
    except rospy.ROSInterruptException:
        pass