from time import time
from camera_manager import CameraManager
import cv2
from camera_manager import CameraManager
import time

import os
os.environ['DISPLAY'] = ':0'

cameraManager = CameraManager()

cameraManager.add_camera("cam0",0)
cameraManager.add_camera("cam1","./sample/fire1.avi",True)
cameraManager.add_camera("cam2","./sample/Result-06-10-2021.mp4",True)

print(cameraManager.get_all_ID())
while(True):
    frame0 = cameraManager.get_camera('cam0').get_frame_raw()
    if(frame0 is not None):
        print(frame0)
        cv2.imshow('cam0',frame0)
    # print("proc1: " + str(cameraManager.get_all_ID()))
    # if('cam1' in cameraManager.get_all_ID()):
    #     frame1 = cameraManager.get_camera('cam1').get_frame_raw()
    #     if(frame1 is not None):
    #         cv2.imshow('cam1',frame1)
    # frame2 = cameraManager.get_camera('cam2').get_frame_raw()
    # if(frame2 is not None):
    #     cv2.imshow('cam2',frame2)
    
cv2.waitKey(0)
cap.release()
cv2.destroyAllWindows()
