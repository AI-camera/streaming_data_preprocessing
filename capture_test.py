import cv2
from time import time
source = cv2.VideoCapture("http://192.168.137.161:81/mainserver/video_feed_raw/cam1")
while True:
    start = time()
    ret, frame = source.read()
    print("frame take:" + str(time() - start))
    if ret:
        cv2.imshow("this",frame)
    if cv2.waitKey(1) == ord("q"):
        break

source.release()
cv2.destroyAllWindows()