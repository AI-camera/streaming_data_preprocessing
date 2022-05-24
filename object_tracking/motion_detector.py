import numpy as np
import cv2

class OpticalFlowMotionDetector:
    def __init__(self):
        self.old_gray = None
        self.motion_points = None
        self.frame_differencing_list = []
        self.feature_params = dict(maxCorners = 100,qualityLevel = 0.3,minDistance = 7, blockSize = 7 )
        self.lk_params = dict( winSize  = (15,15),maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        self.color = np.random.randint(0,255,(100,3))
        self.threshold = 10

    def get_motion_points(self):
        return self.motion_points

    def refresh_motion_points(self,new_frame):
        frame = new_frame
        p0 = cv2.goodFeaturesToTrack(self.old_gray, mask = None, **self.feature_params)
        if(frame is None):
            return
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(self.old_gray, frame_gray, p0, None, **self.lk_params)

        # Select good points
        good_new = p1[st==1]
        good_old = p0[st==1]

        # Now update the previous frame and previous points
        self.old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1,1,2)

        self.motion_points = enumerate(zip(good_new,good_old))
        return self.motion_points

    def detect_optical_flow(self,new_frame):
        # First frame 
        if self.old_gray is None:
            self.old_gray = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)
            return 0

        self.refresh_motion_points(new_frame)
        # find the greatest motion value in image
        motion_value = 0
        for i, (new,old) in self.get_motion_points():
            a,b = new.ravel()
            c,d = old.ravel()
            new_motion_value = np.sqrt((a-c)**2 + (b-d)**2)
            if new_motion_value > motion_value:
                motion_value = new_motion_value

        #Motion values jumps irregularly
        if(motion_value > 10):
            motion_value = 0
        
        return motion_value
    
    def detect_frame_differencing(self, new_frame):
        new_frame_gray = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)
        if len(self.frame_differencing_list) < 10:
            self.frame_differencing_list.append(new_frame_gray)
            return 0
        
        if len(self.frame_differencing_list) > 10:
            self.frame_differencing_list = self.frame_differencing_list[1:]
        
        mean = 0
        for frame in self.frame_differencing_list:
            mean = mean + frame

        mean = mean/len(self.frame_differencing_list)

        difference = new_frame_gray - mean
        difference = cv2.GaussianBlur(difference,(5,5),0)
        difference_sum = 1
        for i in range(len(difference)):
            for j in range(len(difference)):
                if difference[i][j] > self.threshold:
                    difference_sum += 1

        return difference_sum
        
