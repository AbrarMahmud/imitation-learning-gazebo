#!/usr/bin/env python3
import rospy
import sys
import copy
import moveit_commander
from sensor_msgs.msg import Joy

class PlanarJoystickTeleop:
    def __init__(self):
        rospy.init_node('joystick_teleop', anonymous=True)
        moveit_commander.roscpp_initialize(sys.argv)

        # UPDATE THIS: Change 'arm' to your actual MoveIt planning group name
        self.group_name = "arm" 
        self.move_group = moveit_commander.MoveGroupCommander(self.group_name)

        # Joystick state
        self.x_axis = 0.0
        self.y_axis = 0.0
        
        # Tuning parameters
        self.speed = 0.02  # Maximum meters to move per step
        self.deadzone = 0.15 # Ignore stick drift

        rospy.Subscriber("/joy", Joy, self.joy_callback, queue_size=1)
        
        # Run the control loop at 10 Hz
        rospy.Timer(rospy.Duration(0.1), self.control_loop)
        rospy.loginfo("Joystick Teleop Ready. Move the left stick to control X/Y.")

    def joy_callback(self, data):
        # Default mapping for Xbox/PlayStation controllers:
        # data.axes[0] = Left stick horizontal (Left = 1.0, Right = -1.0)
        # data.axes[1] = Left stick vertical (Up = 1.0, Down = -1.0)
        self.y_axis = data.axes[0]
        self.x_axis = data.axes[1]

    def control_loop(self, event):
        # Skip if joystick is in the deadzone (centered)
        if abs(self.x_axis) < self.deadzone and abs(self.y_axis) < self.deadzone:
            return

        # Get current pose to act as a baseline
        current_pose = self.move_group.get_current_pose().pose
        target_pose = copy.deepcopy(current_pose)

        # Apply joystick deltas to X and Y
        target_pose.position.x += (self.x_axis * self.speed)
        target_pose.position.y += (self.y_axis * self.speed)
        
        # Note: Z, Roll, Pitch, and Yaw are untouched to maintain planar movement

        # Command MoveIt
        self.move_group.set_pose_target(target_pose)
        
        # wait=False allows for continuous overriding, making it feel more like a video game
        self.move_group.go(wait=False)
        self.move_group.stop() # Ensure no residual movement
        self.move_group.clear_pose_targets()

if __name__ == '__main__':
    try:
        PlanarJoystickTeleop()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
