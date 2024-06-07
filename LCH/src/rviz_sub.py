import sys
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import rospy

class RVizViewer(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RViz Viewer")
        self.layout = QVBoxLayout()
        self.image_label = QLabel()
        self.layout.addWidget(self.image_label)
        self.setLayout(self.layout)
        
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/rviz/image", Image, self.image_callback)

    def image_callback(self, ros_image):
        cv_image = self.bridge.imgmsg_to_cv2(ros_image, "bgr8")
        height, width, channel = cv_image.shape
        bytes_per_line = 3 * width
        q_image = QImage(cv_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(q_image))

def main():
    rospy.init_node('rviz_image_viewer', anonymous=True)
    app = QApplication(sys.argv)
    rviz_viewer = RVizViewer()
    rviz_viewer.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()

