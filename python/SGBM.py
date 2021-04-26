#!/usr/bin/env python3

import cv2
import numpy as np
import sys
import glob
import math
import time
from numpy import inf
import copy
# ros
import rospy
import message_filters
from std_msgs.msg import Header
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

class StereoSGBM:
    def __init__(self):
        rospy.init_node('stereo_SGBM')
        self.leftImage = np.zeros([1,1,1], dtype=np.uint8)
        self.rightImage = np.zeros([1,1,1], dtype=np.uint8)
        self.depthImage = np.zeros([1,1,1], dtype=np.uint8)
        self.bridge = CvBridge()
        self.image_left_sub = message_filters.Subscriber("/zed/zed_node/left/image_rect_gray", Image)
        self.image_right_sub = message_filters.Subscriber("/zed/zed_node/right/image_rect_gray", Image)
        self.image_depth_sub = message_filters.Subscriber("/zed/zed_node/depth/depth_registered", Image)
        ts = message_filters.TimeSynchronizer([self.image_left_sub, self.image_right_sub, self.image_depth_sub], 10)
        ts.registerCallback(self.callbackImage)
        self.resultSize = (640, 360)
        self.maxDepth = 15

    def settingBM(self):
        self.setBMNum(10)
        self.setBMBlockSize(7)
        cv2.namedWindow('BM')
        cv2.createTrackbar('Num','BM',1,30,self.setBMNum)
        cv2.createTrackbar('BlockSize','BM',7,10,self.setBMBlockSize)
        cv2.setTrackbarPos('Num','BM', 10)
        cv2.setTrackbarPos('BlockSize','BM', 7)

    def settingSGBM(self):
        self.setSGBMWindowSize(8)
        self.setSGBMMinDisp(0)
        self.setSGBMBlockSize(6)
        cv2.namedWindow('SGBM')
        cv2.createTrackbar('windowSize','SGBM',1, 30, self.setSGBMWindowSize)
        cv2.createTrackbar('minDisp','SGBM', 0, 7, self.setSGBMMinDisp)
        cv2.createTrackbar('BlockSize','SGBM',1,10,self.setSGBMBlockSize)
        cv2.setTrackbarPos('windowSize','SGBM', 8)
        cv2.setTrackbarPos('minDisp','SGBM', 0)
        cv2.setTrackbarPos('BlockSize','SGBM', 6)

    def setBMNum(self, num):
        self.num = 16*num

    def setBMBlockSize(self, blockSize):
        self.blockSize = blockSize * 2 + 1

    def setSGBMWindowSize(self, windowSize):
        self.windowSize = windowSize

    def setSGBMMinDisp(self, minDisp):
        self.minDisp = minDisp * 16
        self.numDisp = 96 - self.minDisp

    def setSGBMBlockSize(self, blockSize):
        self.blockSize = blockSize * 2 + 1

    def callbackImage(self, leftMsg, rightMsg, depthMsg):
        try:
            self.leftImage = self.bridge.imgmsg_to_cv2(leftMsg)
            self.rightImage = self.bridge.imgmsg_to_cv2(rightMsg)
            self.depthImage = self.bridge.imgmsg_to_cv2(depthMsg)
        except CvBridgeError as e:
            print(e)
        
    def showImage(self, algorithm):
        if (algorithm == 'BM'):
            self.settingBM()
        elif(algorithm == 'SGBM'):
            self.settingSGBM()
        while not rospy.is_shutdown():
            # result = np.nan_to_num(self.depthImage)
            cv2.imshow("left", cv2.resize(self.leftImage, self.resultSize , interpolation=cv2.INTER_CUBIC))    
            depth = np.nan_to_num(copy.deepcopy(self.depthImage))
            depth[depth == inf] = 0
            depth = np.clip(depth, 0,  self.maxDepth) / self.maxDepth * 255.0 
            depth = cv2.normalize(depth, depth, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            # depth =  cv2.applyColorMap(depth.astype(np.uint8), cv2.COLORMAP_AUTUMN)
            cv2.imshow("depth", cv2.resize(depth , self.resultSize , interpolation=cv2.INTER_CUBIC))        
            startTime = time.time()
            if (algorithm == 'BM'):
                disp = self.stereoToDepthByBM(self.leftImage, self.rightImage)
            elif(algorithm == 'SGBM'):
                disp = self.stereoToDepthBySGBM(self.leftImage, self.rightImage)
            else:
                print('algorithm Type error')
                return
            endTime = time.time()
            # disp =  cv2.applyColorMap(disp.astype(np.uint8), cv2.COLORMAP_JET)
            temp = copy.deepcopy(disp)
            disp = 255 - temp
            disp[temp == 0] = 0
            disp[disp >= 240] = 0
            cv2.rectangle(disp, (0, 0), (170, 85), (0), -1)
            cv2.putText(disp, algorithm, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255), 3, cv2.LINE_AA)
            cv2.putText(disp, 'FPS ' + str(round(1 / (endTime - startTime), 2)), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255), 3, cv2.LINE_AA)
            
            cv2.imshow(algorithm, cv2.resize(disp, self.resultSize , interpolation=cv2.INTER_CUBIC))
            if (cv2.waitKey(1) & 0xFF ==ord('q')):
                break
    def stereoToDepthByBM(self, imgL, imgR):
        stereo = cv2.StereoBM_create(numDisparities=self.num, blockSize=self.blockSize)
        stereo.setDisp12MaxDiff(1)
        stereo.setSpeckleRange(32)
        stereo.setSpeckleWindowSize(150)
        disparity = stereo.compute(imgL, imgR)
        disp = cv2.normalize(disparity, disparity, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        return disp

    def stereoToDepthBySGBM(self, imgL, imgR):
        stereo = cv2.StereoSGBM_create(minDisparity=self.minDisp,
                                    numDisparities=self.numDisp,
                                    blockSize=self.blockSize,
                                    P1=8 * 3 * self.windowSize ** 2,
                                    P2=32 * 3 * self.windowSize ** 2,
                                    disp12MaxDiff=1,
                                    uniquenessRatio=10,
                                    speckleWindowSize=150,
                                    speckleRange=2
                                    )
        disparity = stereo.compute(imgL, imgR)

        disp = cv2.normalize(disparity, disparity, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        return disp
    
if __name__ == '__main__':
    stereoSGBM = StereoSGBM()
    try:
        # stereoSGBM.showImage('SGBM')
        stereoSGBM.showImage('BM')
        # rospy.spin()
    except KeyboardInterrupt:
        print ("Shutting down ROS Image feature detector module")
    cv2.destroyAllWindows()