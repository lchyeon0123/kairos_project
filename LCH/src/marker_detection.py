#!/usr/bin/env python
import roslib
roslib.load_manifest('LCH')
import sys
import rospy
import cv2
from std_msgs.msg import String
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import cv2.aruco as aruco
import yaml
import os
from geometry_msgs.msg import Pose, TransformStamped
from tf2_ros import TransformBroadcaster
from scipy.spatial.transform import Rotation as R

class ArucoDetection:

    def __init__(self, video_source):
        self.pose_pub = rospy.Publisher("pose", Pose, queue_size=1)
        self.tfbroadcaster = TransformBroadcaster()
        self.bridge = CvBridge()
        self.cap = open_video_from_arg(video_source)
        
        if not self.cap.isOpened():
            print("Failed to open video source.")
            sys.exit()

    def callback(self):
        ret, cv_image = self.cap.read()
        if not ret:
            print("Failed to capture image from video source.")
            return

        try:
            cv_image1 = cv2.resize(cv_image, dsize=(10, 10), interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            gray1 = cv2.cvtColor(cv_image1, cv2.COLOR_BGR2GRAY)
            pixelsum = cv2.sumElems(gray1)
            print('sum : ', pixelsum)
            aruco_dict = aruco.Dictionary_get(aruco.DICT_ARUCO_ORIGINAL)
            parameters = aruco.DetectorParameters_create()
            corners, ids, rejected_img_points = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
            
            p = Pose()

            if np.all(ids is not None):
                print('marker id : ', ids)
                for i in range(0, len(ids)):
                    rvec, tvec, markerPoints = aruco.estimatePoseSingleMarkers(
                        corners[i], 0.1, matrix_coefficients, distortion_coefficients
                    )
                    R_matrix, _ = cv2.Rodrigues(rvec) # 로드리그스 공식을 사용하여 회전 벡터를 회전 행렬로 변환
                    #R_matrix: 출력 회전 행렬

                    #회전 행렬과 변위 벡터를 결합하여 변환 행렬을 생성
                    tvec1 = np.reshape(tvec, (3, 1)) #tvec를 (3, 1) 형태로 재구성.
                    #변위 벡터를 3x1 행렬로 변환하여 이후의 계산에 사용하기 위함

                    t = TransformStamped() #TransformStamped 객체를 생성
                    t.header.stamp = rospy.Time.now() #현재 시간으로 타임스탬프를 설정
                    t.header.frame_id = 'camera_depth_frame'
                    t.child_frame_id = 'aruco_marker_frame_{}'.format(ids[i][0])

                    # 마커의 위치
                    t.transform.translation.x = tvec1[0]
                    t.transform.translation.y = tvec1[1]
                    t.transform.translation.z = tvec1[2]

                    rotation_matrix = np.eye(4) #4x4 단위 행렬을 생성
                    rotation_matrix[0:3, 0:3] = R_matrix #R_matrix를 이 단위 행렬의 상위 3x3 부분에 할당하여 회전 행렬을 설정
                    r = R.from_matrix(rotation_matrix[0:3, 0:3]) #scipy.spatial.transform.Rotation을 사용하여 회전 행렬을 쿼터니언으로 변환
                    #r: 회전 행렬을 나타내는 Rotation 객체

                    quat = r.as_quat() #쿼터니언

                    t.transform.rotation.x = quat[0] #쿼터니언의 x 성분
                    t.transform.rotation.y = -quat[1]
                    t.transform.rotation.z = quat[2]
                    t.transform.rotation.w = -quat[3]

                    self.tfbroadcaster.sendTransform(t)
                    #tf2_ros.TransformBroadcaster 객체를 사용하여 변환을 ROS TF 프레임워크로 전송

                    aruco.drawDetectedMarkers(cv_image.copy(), corners, ids)
                    aruco.drawAxis(cv_image, matrix_coefficients, distortion_coefficients, rvec, tvec, 0.05)
                    
                    ct = np.array([tvec1[0], tvec1[1], tvec1[2]]) #마커의 위치를 ct라는 배열에 저장
                    #tvec1[0], [1], [2]는 각각 x축, y축, z축을 따른 마커의 변위

                    if (ct[0] < -0.02): 
                        cv2.arrowedLine(cv_image, (150, 240), (50, 240), (138,43,226), 3)
                    elif (ct[0] > 0.02):
                        cv2.arrowedLine(cv_image, (490, 240), (590, 240), (138,43,226), 3)
                    if (ct[0] > 0 and ct[0] < 0.02):
                        print('-------------------------------------------------------')
                    if (ct[0] > -0.02 and ct[0] < 0.02):
                        str='Front'
                        cv2.putText(cv_image, str, (280, 100), cv2.FONT_HERSHEY_PLAIN, 3, (138, 43, 226), 3)

                    str='o'
                    cv_image = draw_data(cv_image, 0, ct[0], ct[1])
                

                    p.position.x = tvec1[0]
                    p.position.y = tvec1[1]
                    p.position.z = tvec1[2]
                    p.orientation.x = 0.0 # 회전에 대한 정보가 없으므로 모든 값을 0으로 설정
                    p.orientation.y = 0.0
                    p.orientation.z = 0.0
                    p.orientation.w = 1.0 # 방향이 고정되어 있다는 것을 의미

        except CvBridgeError as e:
            print(e)

        cv2.imshow("Image window", cv_image)
        cv2.waitKey(3)

        try:
            self.pose_pub.publish(p)
        except CvBridgeError as e:
            print(e)


def open_video_from_arg(video_input):
    try:
        source = int(video_input)
        print("Trying to open video ID", video_input)
        in_video = cv2.VideoCapture(source)
        return in_video
    except ValueError:
        print("Trying to open video URL", video_input)
        in_video = cv2.VideoCapture(video_input)
        return in_video


def parse_video_in():
    video_input = "0"
    if len(sys.argv) > 1:
        video_input = sys.argv[1]
    print("Trying to open video source", video_input)
    in_video = open_video_from_arg(video_input)
    if not in_video.isOpened():
        print("Failed to open video input:", video_input)
        return None
    print("Video input", video_input, "successfully opened")
    return video_input


def main(args):
    rospy.init_node('aruco_detection', anonymous=True)
    video_source = parse_video_in()
    if video_source is None:
        return
    ic = ArucoDetection(video_source)
    try:
        rate = rospy.Rate(10)  # 10 Hz
        while not rospy.is_shutdown():
            ic.callback()
            rate.sleep()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()


def draw_data(original_img, curv, center_dist1, center_dist2):
    new_img = original_img
    h = new_img.shape[0]
    font = cv2.FONT_HERSHEY_DUPLEX
    text = 'Curve' + '{:04.2f}'.format(float(curv)) + 'degree'
    cv2.putText(new_img, text, (40, 70), font, 0.8, (255, 255, 255), 1, cv2.LINE_AA)
    
    direction1 = ''
    direction2 = ''
    abs_center_dist1 = abs(center_dist1)
    abs_center_dist2 = abs(center_dist2)
    if center_dist1 > 0.02:
        direction1 = 'left'
        text = '{:.2f}'.format(float(abs_center_dist1)) + 'm ' + direction1 + ' of center'
        cv2.putText(new_img, text, (40, 120), font, 0.8, (255, 255, 255), 1, cv2.LINE_AA)
    elif center_dist1 < -0.02:
        direction1 = 'right'
        text = '{:.2f}'.format(float(abs_center_dist1)) + 'm ' + direction1 + ' of center'
        cv2.putText(new_img, text, (40, 120), font, 0.8, (255, 255, 255), 1, cv2.LINE_AA)
    if center_dist2 > 0.02:
        direction2 = 'down'
        text = '{:.2f}'.format(float(abs_center_dist2)) + 'm ' + direction2 + ' of center'
        cv2.putText(new_img, text, (40, 170), font, 0.8, (255, 255, 255), 1, cv2.LINE_AA)
    elif center_dist2 < -0.02:
        direction2 = 'up'
        text = '{:.2f}'.format(float(abs_center_dist2)) + 'm ' + direction2 + ' of center'
        cv2.putText(new_img, text, (40, 170), font, 0.8, (255, 255, 255), 1, cv2.LINE_AA)

    return new_img

if __name__ == '__main__':
    with open('/home/kairos/noetic_ws/src2/calibration_params.yml') as f:
        data = f.read()
        vegetables = yaml.load(data, Loader=yaml.FullLoader)
        k = vegetables['K']
        d = vegetables['D']
        kd = k['kdata']
        kd = np.reshape(kd, (3, 3))
        dd = d['ddata']
        matrix_coefficients = np.array(kd)
        distortion_coefficients = np.array(dd)
    f.close()
    main(sys.argv)
