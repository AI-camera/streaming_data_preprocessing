import cv2
import sys
# import os

(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')


class ObjectTracking:
    def __init__(self,first_frame):
        pass
        # os.environ['DISPLAY'] = ':0'
        # Set up tracker.
        # Instead of MIL, you can also use

        tracker_types = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
        tracker_type = tracker_types[7]

        if int(major_ver) < 4 and int(minor_ver) < 3:
            self.tracker = cv2.cv2.Tracker_create(tracker_type)
        else:
            if tracker_type == 'BOOSTING':
                self.tracker = cv2.TrackerBoosting_create()
            if tracker_type == 'MIL':
                self.tracker = cv2.TrackerMIL_create()
            if tracker_type == 'KCF':
                self.tracker = cv2.TrackerKCF_create()
            if tracker_type == 'TLD':
                self.tracker = cv2.TrackerTLD_create()
            if tracker_type == 'MEDIANFLOW':
                self.tracker = cv2.TrackerMedianFlow_create()
            if tracker_type == 'GOTURN':
                self.tracker = cv2.TrackerGOTURN_create()
            if tracker_type == 'MOSSE':
                self.tracker = cv2.TrackerMOSSE_create()
            if tracker_type == "CSRT":
                self.tracker = cv2.TrackerCSRT_create()

        # Read video
        # video = cv2.VideoCapture(0)

        # Exit if video not opened.
        # if not video.isOpened():
        #     print("Could not open video")
        #     sys.exit()

        # Define an initial bounding box
        bbox = (287, 23, 86, 320)

        # Uncomment the line below to select a different bounding box
        # bbox = cv2.selectROI(first_frame, False)

        # Initialize tracker with first frame and bounding box
        ok = self.tracker.init(first_frame, bbox)

    def track(self,frame):
        # Read a new frame
        # ok, frame = video.read()
        # if not ok:
        #     break
        
        # # Start timer
        # timer = cv2.getTickCount()

        # # Update tracker
        ok, bbox = self.tracker.update(frame)

        # # Calculate Frames per second (FPS)
        # fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);

        # Draw bounding box
        if ok:
            # Tracking success
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            frame = cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
        else :
            # Tracking failure
            cv2.putText(frame, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)

        return frame

        # Display tracker type on frame
        # cv2.putText(frame, tracker_type + " Tracker", (100,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2);
    
        # Display FPS on frame
        # cv2.putText(frame, "FPS : " + str(int(fps)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2);

        # Display result
        # cv2.imshow("Tracking", frame)

        # Exit if ESC pressed
        # k = cv2.waitKey(1) & 0xff
        # if k == 27 : break
