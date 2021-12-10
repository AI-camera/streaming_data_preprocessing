import cv2
from multiprocessing import Process

def video(source, count): 

  cap = cv2.VideoCapture(source)
  while(True):
    ret, frame = cap.read()
    if ret == True:
      cv2.imshow('frame_'+str(count),frame)
      if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    else :
        break
  cap.release()
  cv2.destroyAllWindows()


if __name__ == '__main__':
    p1= Process(target = video, args=('./sample/fire1.avi', 0,))
    p2= Process(target = video, args=('./sample/fire1.avi', 1,))
    p1.start() 
    p2.start()

    p1.join()
    p2.join()