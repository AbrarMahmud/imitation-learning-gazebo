#!/usr/bin/env python3
import sys
import rospy
from std_msgs.msg import String
from python_qt_binding.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton

class DatasetControlPanel(QWidget):
    def __init__(self):
        super(DatasetControlPanel, self).__init__()
        rospy.init_node('dataset_control_panel', anonymous=True)
        
        # Publisher to send commands to the recorder script
        self.pub = rospy.Publisher('/dataset_command', String, queue_size=10)
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('Push-T Dataset Controller')
        self.resize(300, 200)
        layout = QVBoxLayout()

        # Record Button (Green)
        self.btn_record = QPushButton('START / STOP RECORDING', self)
        self.btn_record.setMinimumHeight(80)
        self.btn_record.setStyleSheet("background-color: rgb(0, 230, 0); color: black; font-weight: bold; font-size: 16px;")
        self.btn_record.clicked.connect(self.send_toggle_record)
        layout.addWidget(self.btn_record)

        # Upload Button (Blue)
        self.btn_upload = QPushButton('UPLOAD TO HF CLOUD', self)
        self.btn_upload.setMinimumHeight(80)
        self.btn_upload.setStyleSheet("background-color: rgb(0, 100, 255); color: white; font-weight: bold; font-size: 16px;")
        self.btn_upload.clicked.connect(self.send_upload)
        layout.addWidget(self.btn_upload)

        self.setLayout(layout)

    def send_toggle_record(self):
        rospy.loginfo("Sending RECORD command...")
        self.pub.publish("TOGGLE_RECORD")

    def send_upload(self):
        rospy.loginfo("Sending UPLOAD command...")
        self.pub.publish("UPLOAD")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    panel = DatasetControlPanel()
    panel.show()
    # ROS spin is handled by the Qt event loop
    sys.exit(app.exec_())
