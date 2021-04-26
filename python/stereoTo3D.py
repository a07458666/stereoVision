import cv2
import numpy as np
import csv
import math
import sys
import glob

class ObjectData:
    def __init__(self,d,x,y,w,h):
        self.d = d
        self.x = x
        self.y = y
        self.w = w
        self.h = h

class StereoTo3D:
    def __init__(self, fov, imageSize, step = 20):
        self.fov = fov
        self.imageW = imageSize[0]
        self.imageH = imageSize[1]
        self.step = step
        self.scale = (self.imageH / step) % self.imageH
        self.scannerlen = 255.0 / (self.imageH / self.step)
        self.csvData = {}
        return;

    def drawMap(self, poses):
        result = np.zeros((self.imageH, self.imageW, 3), np.uint8)
        for i in range(0, int(self.imageH/ self.step)):
            center = (int(self.imageW / 2), self.imageH)
            axes = (int(i * self.scale), int(i * self.scale))
            angle = 0
            startAngle = -90 + self.fov / 2
            endAngle = -90 - self.fov / 2
            color = (255 - i * self.scannerlen, 255 - i * self.scannerlen, 255 - i * self.scannerlen)
            thickness = 2
            cv2.ellipse(result, center, axes, angle, startAngle, endAngle, color, thickness)
        for pose in poses:
            pose_color = (255,153,18)
            startPoseAngle = endAngle + ((pose.x + pose.w) / self.imageW * self.fov)
            endPoseAngle =  endAngle + (pose.x / self.imageW * self.fov)
            poseAxes = (int(pose.d * self.scale), int(pose.d * self.scale))
            cv2.ellipse(result, center, poseAxes, angle, startPoseAngle, endPoseAngle, pose_color,3)
        for pose in poses:
            pointColor = (225,0,0)        
            object_x, object_y = self.convert2D(pose, center)
            cv2.circle(result, (int(object_x), int(object_y)), 3, pointColor, -1)
            pose.x += pose.w
            object_x, object_y = self.convert2D(pose, center)
            cv2.circle(result, (int(object_x), int(object_y)), 3, pointColor, -1)
            w = 85
            h = 12
            cv2.rectangle(result, (int(object_x) + 5, int(object_y + 3)), (int(object_x) + 5 + w, int(object_y + 3) - h), (0,255,0), -1)
            cv2.putText(result, str(pose.d) + 'm person', (int(object_x) + 5, int(object_y)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1)
        return result

    def convert2D(self, pose, center):
        theta = (pose.x / self.imageW * self.fov) - (self.fov / 2)
        theta = theta * math.pi / 180
        pixD = pose.d * self.scale
        object_x = center[0] + (pixD * math.sin(theta))
        object_y = center[1] - (pixD * math.cos(theta))
        return object_x, object_y
    
    def loadCSV(self, filename):
        with open(csvPath, newline='') as csvfile:
            rows = csv.reader(csvfile)
            for row in rows:
                self.csvData[row[0]] = ObjectData(float(row[1]), int(float(row[2])),int(float(row[3])), int(float(row[4])), int(float(row[5])))

    def getObjectData(self, frameID):
        if(frameID in self.csvData):
            return self.csvData[frameID]
        else:
            return None

if __name__ == '__main__':
    imagePath = sys.argv[1]
    csvPath = sys.argv[2]
    # print('imagePath', imagePath)
    # print('csvPath: ', csvPath)
    # print(glob.glob(imagePath))
    fov = 50
    imageSize = (640, 363)
    stereoTo3D = StereoTo3D(fov, imageSize)
    stereoTo3D.loadCSV(csvPath)
    for path in glob.glob(imagePath):
        frameID = path.split('/')[-1][:-8]
        print('frameID: ', frameID)
        objectData = stereoTo3D.getObjectData(str(frameID))
        if (objectData == None):
            continue
        result = stereoTo3D.drawMap([objectData]);
        img = cv2.imread(path)
        cv2.imshow('img ', img)
        cv2.imshow('result ', result)
        cv2.waitKey(0)