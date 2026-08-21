#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script 1 — ROS 1 / Python 3.8 — LeRobot Inference Bridge (synchronous)
========================================================================
Collects observations from Gazebo (image + joint states) and sends them
to the Kaggle/Colab inference server (Script 2) ONE AT A TIME.  Publishes
the returned action as a JointTrajectory command, then — and only then —
sends the next observation.

Lock-step contract
-------------------
  1. Capture the current image + joint state.
  2. POST that single observation to /infer and BLOCK until a response
     arrives (the server runs exactly one policy forward pass per call).
  3. Publish the single action returned.
  4. Repeat from 1.

There is no client-side buffering, prefetching, or read-ahead: the bridge
never has more than one in-flight request to the server, and it never
sends a new observation before the previous action has been applied.

Topics
------
  Subscribed:
    /push_t/overhead_camera/image_raw  (sensor_msgs/Image)
    /joint_states                      (sensor_msgs/JointState)

  Published:
    /arm_ctrl/command                  (trajectory_msgs/JointTrajectory)

Server API (synchronous server)
--------------------------------
  POST /infer  { "observation.state": [...], "observation.image": "<b64>" }
               → { "action": [...] }
               → { "error": "..." }              (4xx/5xx)

  POST /reset  (call on episode reset / node start-up to clear the policy's
               internal observation window)
  GET  /health (liveness check)
"""

import os
os.environ["PYTHONUNBUFFERED"] = "1"

import base64
import threading
import time
from typing import Optional

import cv2
import numpy as np
import requests
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

# ── Constants ────────────────────────────────────────────────────────────────
IMG_SIZE        = (96, 96)   # (W, H) for cv2.resize  ← note cv2 is (W, H)
STATE_DIM       = 3
REQUEST_TIMEOUT = 30.0        # seconds; a single synchronous forward pass can
                               # be slow (diffusion, first call, CPU fallback)

NGROK_HEADERS   = {"ngrok-skip-browser-warning": "true"}

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

        rospy.loginfo("=== LeRobot Inference Bridge (synchronous, single-step) ===")
        rospy.loginfo(f"  server      : {self.server_url}")
        rospy.loginfo(f"  img_topic   : {self.img_topic}")
        rospy.loginfo(f"  state_topic : {self.state_topic}")
        rospy.loginfo(f"  cmd_topic   : {self.cmd_topic}")
        rospy.loginfo(f"  target fps  : {self.target_fps}  (best-effort cap — actual "
                       f"rate is however fast the server can do one step)")

    # ── ROS callbacks ────────────────────────────────────────────────────────
    # These just cache the latest sensor readings. They do NOT trigger
    # inference — the main loop below pulls a fresh snapshot once per
    # iteration and drives the request/response cycle itself.

    def _image_cb(self, msg: Image):
        try:
            cv_img  = self._bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
            # cv2.resize expects (W, H)
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
        """
        with self._lock:
            if self._latest_image is None or self._latest_state is None:
                return None
            image = self._latest_image.copy()
            state = self._latest_state.copy()

        # Encode RGB image as PNG.  cv2.imencode expects BGR.
        success, png_buf = cv2.imencode(
            ".png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        if not success:
            rospy.logwarn("Failed to encode image as PNG")
            return None

        return {
            "observation.state": state.tolist(),
            "observation.image": base64.b64encode(
                png_buf.tobytes()).decode("utf-8"),
            "timestamp": time.time(),
        }

    # ── Server communication ─────────────────────────────────────────────────

    def _send_observation(self, obs: dict) -> Optional[np.ndarray]:
        """
        Sends exactly ONE observation and blocks until the server returns
        exactly ONE action (or fails). This call IS the synchronization
        point — nothing else in the loop proceeds until it returns.
        """
        try:
            r = requests.post(
                f"{self.server_url}/infer",
                json=obs,
                timeout=REQUEST_TIMEOUT,
                headers=NGROK_HEADERS,
            )
            r.raise_for_status()
            payload = r.json()
            if "error" in payload:
                rospy.logerr(f"Server returned error: {payload['error']}")
                return None
            return np.array(payload["action"], dtype=np.float32)
        except requests.exceptions.Timeout:
            rospy.logwarn("Inference request timed out")
        except requests.exceptions.ConnectionError:
            rospy.logerr(f"Cannot reach server: {self.server_url}")
        except Exception as e:
            rospy.logwarn(f"Inference request failed: {e}")
        return None

    def _reset_policy(self):
        """
        Tell the server to clear the policy's internal observation/action
        window. Call this at node start-up (and whenever Gazebo resets an
        episode) so the first inference of an episode sees a clean history.
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
                f"image_keys: {info.get('image_keys')}  "
                f"mode: {info.get('mode', 'n/a')}"
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

        # 2. Reset the policy's internal window (clears any stale obs from a
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

        rospy.loginfo("Both observations received — synchronous step loop running.")

        # 4. Synchronous, single-step loop:
        #    pack ONE observation -> block on the server for ONE action ->
        #    publish it -> only then start the next iteration. `rate.sleep()`
        #    is just a best-effort cap on the loop rate when the server is
        #    faster than target_fps; it never makes the client run ahead of
        #    the server, since we've already blocked on the request above.
        rate = rospy.Rate(self.target_fps)
        while not rospy.is_shutdown():
            obs = self._pack_observation()
            if obs is None:
                rate.sleep()
                continue

            step_start = time.time()
            action = self._send_observation(obs)  # blocks for exactly one step
            step_elapsed = time.time() - step_start

            if action is not None:
                self._publish_action(action)
                rospy.loginfo_throttle(
                    2.0, f"Action: {action.tolist()}  |  step time: {step_elapsed:.3f}s")
            else:
                rospy.logwarn("No action received — skipping this step")
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
