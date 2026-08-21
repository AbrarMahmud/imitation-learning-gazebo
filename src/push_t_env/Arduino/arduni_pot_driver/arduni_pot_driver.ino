/*
 * ROS1 (rosserial) node
 * ----------------------
 * Reads 3 potentiometers on A3, A4, A5, maps each to a joint angle range,
 * applies a low-pass (EMA) filter to prevent sudden jumps, and publishes
 * a trajectory_msgs/JointTrajectory message on topic "/arm_ctrl/command".
 *
 * Requires on the ROS PC side:
 *   sudo apt install ros-<distro>-rosserial ros-<distro>-rosserial-arduino
 * On the Arduino IDE side:
 *   Sketch -> Include Library -> Manage Libraries -> search "Rosserial Arduino Library"
 *   (or generate ros_lib via rosserial_arduino's make_libraries.py)
 *
 * Run:
 *   rosrun rosserial_python serial_node.py /dev/ttyUSB0 _baud:=57600
 */

#include <ros.h>
#include <trajectory_msgs/JointTrajectory.h>
#include <trajectory_msgs/JointTrajectoryPoint.h>

// ---------------- Pin definitions ----------------
const int pot1Pin = A5;
const int pot2Pin = A4;
const int pot3Pin = A3;

// ---------------- Calibration (raw ADC -> radians) ----------------
// Edit these per-pot values to match your measured min/max raw readings
// and desired output angle range.
struct PotCalib {
  float rawMin;
  float rawMax;
  float outMin;
  float outMax;
};

PotCalib potCalib[3] = {
  { 820.0,  170.0, -3.1416/2, 3.1416/2 },  // pot1: raw DOWN  -> angle UP
  { 820.0,  170.0, -3.1416/2, 3.1416/2 },  // pot2: raw UP    -> angle UP
  { 170.0,  820.0, -3.1416/2, 3.1416/2 }   // pot3: raw UP    -> angle UP
};
// Generic float mapping (like Arduino's map(), but works with floats)
// Works even if inMin > inMax (i.e. a "reversed" pot calibration).
float mapFloat(float x, float inMin, float inMax, float outMin, float outMax) {
  // clamp input to the calibration range so output never leaves [outMin, outMax],
  // regardless of whether inMin/inMax are given in ascending or descending order
  float lo = min(inMin, inMax);
  float hi = max(inMin, inMax);
  if (x < lo) x = lo;
  if (x > hi) x = hi;
  return (x - inMin) * (outMax - outMin) / (inMax - inMin) + outMin;
}

// ---------------- Low-pass (EMA) filter ----------------
// Smaller alpha = smoother but slower response, larger alpha = snappier but noisier.
// Tune 0.0 - 1.0 to taste.
const float FILTER_ALPHA = 0.05;  //0.15
float filteredRaw[3] = { 0, 0, 0 };
bool filterInit = false;

float lowPassFilter(float newValue, float &prevFiltered) {
  prevFiltered = FILTER_ALPHA * newValue + (1.0 - FILTER_ALPHA) * prevFiltered;
  return prevFiltered;
}

// ---------------- ROS setup ----------------
ros::NodeHandle nh;

trajectory_msgs::JointTrajectory traj_msg;
ros::Publisher arm_pub("/arm_ctrl/command", &traj_msg);

// Static allocations required by rosserial (no dynamic std::vector on the wire)
char* jointNames[3] = { "joint_1", "joint_2", "joint_3" };
trajectory_msgs::JointTrajectoryPoint traj_point;
float positions[3];

uint32_t seqCounter = 0;

void setup() {
  nh.getHardware()->setBaud(57600);
  nh.initNode();
  nh.advertise(arm_pub);

  // Pre-fill static message fields
  traj_msg.joint_names = jointNames;
  traj_msg.joint_names_length = 3;

  traj_point.positions = positions;
  traj_point.positions_length = 3;
  traj_point.velocities_length = 0;
  traj_point.accelerations_length = 0;
  traj_point.effort_length = 0;
  traj_point.time_from_start.sec = 1;
  traj_point.time_from_start.nsec = 0;

  traj_msg.points = &traj_point;
  traj_msg.points_length = 1;

  traj_msg.header.frame_id = "";
}

void loop() {
  // 1. Read raw ADC values
  int raw1 = analogRead(pot1Pin);
  int raw2 = analogRead(pot2Pin);
  int raw3 = analogRead(pot3Pin);

  // Initialize filter state on first pass to avoid a startup ramp-in
  if (!filterInit) {
    filteredRaw[0] = raw1;
    filteredRaw[1] = raw2;
    filteredRaw[2] = raw3;
    filterInit = true;
  }

  // 2. Smooth the raw readings (prevents sudden jumps / noise)
  float f1 = lowPassFilter((float)raw1, filteredRaw[0]);
  float f2 = lowPassFilter((float)raw2, filteredRaw[1]);
  float f3 = lowPassFilter((float)raw3, filteredRaw[2]);

// 3. Map filtered raw values to each joint's radian range (Up to two decimal place)
  positions[0] = round(mapFloat(f1, potCalib[0].rawMin, potCalib[0].rawMax,
                                 potCalib[0].outMin, potCalib[0].outMax) * 100.0) / 100.0;
  positions[1] = round(mapFloat(f2, potCalib[1].rawMin, potCalib[1].rawMax,
                                 potCalib[1].outMin, potCalib[1].outMax) * 100.0) / 100.0;
  positions[2] = round(mapFloat(f3, potCalib[2].rawMin, potCalib[2].rawMax,
                                 potCalib[2].outMin, potCalib[2].outMax) * 100.0) / 100.0;

  // 4. Fill header and publish
  traj_msg.header.seq = seqCounter++;
  traj_msg.header.stamp = nh.now();

  arm_pub.publish(&traj_msg);

  nh.spinOnce();
  delay(50); // ~20 Hz publish rate
}