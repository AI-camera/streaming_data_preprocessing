import cv2
import threading

import os
os.environ['DISPLAY'] = ':0'


# video_capture_0 = cv2.VideoCapture('video/test.mp4')
# video_capture_1 = cv2.VideoCapture('video/bigbuck.mp4')
# video_capture_2 = cv2.VideoCapture(0)

# while True:
#     ret0, frame0 = video_capture_0.read()
#     ret1, frame1 = video_capture_1.read()
#     ret2, frame2 = video_capture_2.read()

#     if(ret0):
#         cv2.imshow('Cam 0', frame0)
    
#     if(ret1):
#         cv2.imshow('Cam 1', frame1)

#     if(ret2):
#         cv2.imshow('Cam 2', frame2)
    
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# video_capture_0.release()
# video_capture_1.release()
# video_capture_2.release()
# cv2.destroyAllWindows()


def video(source):
    cap = cv2.VideoCapture(source)
    while(True):
        ret, frame = cap.read()
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()

if __name__== "__main__":
    try:
        thread1 = threading.Thread(target= video, args=('./sample/fire1.avi', ))
        thread2 = threading.Thread(target= video, args=('./sample/Result-06-10-2021.mp4', ))
        thread1.start()
        thread2.start()
    except:
        print('error')
    cv2.destroyAllWindows()