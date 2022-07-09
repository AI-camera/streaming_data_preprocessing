import cv2
import time
import numpy as np
import ExtractKeypoint as ext
from skimage.io import imread
from scipy.ndimage import zoom
import tensorflow as tf
import sys
from gtts import gTTS
from playsound import playsound
import pyttsx3
def imread_size(in_path):
    t_img = imread(in_path)
    return zoom(t_img, [64/t_img.shape[0], 64/t_img.shape[1]]+([1] if len(t_img.shape)==3 else []), order = 2)
################################
wCam, hCam = 640, 480
################################

# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture("sample/hand_01.mp4")
cap.set(3, wCam)
cap.set(4, hCam)
pTime = 0
detector = ext.handDetector(detectionCon=0.7)
isTrue = [0, 0, 0]
model = tf.keras.models.load_model("./models/modelb0_full.h5")
mytext = ''
language = 'en'
voice = pyttsx3.init()

# Define video writer properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
codec = cv2.VideoWriter_fourcc('M','J','P','G')
out = cv2.VideoWriter("./output/output_hand.avi", codec, fps, (width, height))

while True:
    try:
        success, img = cap.read()
        img_new = detector.findHands(img)
        lmList = detector.findPosition(img_new, draw=True)
        if lmList:
            X = []
            Y = []
            for i in range(21):
                X.append(lmList[0][i][1])
                Y.append(lmList[0][i][2])
            x_max = max(X)
            x_min = min(X)
            y_max = max(Y)
            y_min = min(Y)
            success, img = cap.read()
            img_crop = img[int(y_min):int(y_max), int(x_min):int(x_max), :]
            cv2.imwrite('test.png', img_crop)
            img = cv2.cvtColor(img_crop, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LINEAR)
            predict_result = model.predict(img[None, :])
            predict_result_id = np.argmax(np.squeeze(predict_result))
            if np.squeeze(predict_result)[predict_result_id] > 0.85:
                cv2.putText(img_new, str(predict_result_id), (40, 100), cv2.FONT_HERSHEY_COMPLEX,
                            1, (255, 0, 0), 3)
                if isTrue[0] == predict_result_id and isTrue[1] == 5 and isTrue[2] == 0:
                    mytext = str(predict_result_id) + "function selected"
                    voice.say(mytext)
                    voice.runAndWait()
                    cv2.putText(img_new, 'ok', (40, 150), cv2.FONT_HERSHEY_COMPLEX,
                                1, (255, 0, 0), 3)
                    isTrue[2] = 1
                    print(predict_result_id)
                elif isTrue[0] == predict_result_id and isTrue[1] != 5:
                    isTrue[1] += 1
                    isTrue[2] = 0
                elif isTrue[0] != predict_result_id:
                    isTrue[0] = predict_result_id
                    isTrue[1] = 1
                    isTrue[2] = 0
            cTime = time.time()
            fps = 1 / (cTime - pTime)
            pTime = cTime
            # cv2.putText(img_new, f'FPS: {int(fps)}', (40, 50), cv2.FONT_HERSHEY_COMPLEX,
            #             1, (255, 0, 0), 3)
            # cv2.imshow("Img", img_new)
            out.write(img_new)
        else:
            # cv2.imshow("Img", img)
            out.write(img)
    except:
        success, img = cap.read()
        # cv2.imshow("Img", img)
        out.write(img)
        continue

    sys.stdout.flush()
    # cv2.waitKey(1)
    out.close()