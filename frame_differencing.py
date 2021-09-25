import numpy as np
import cv2,time,pandas

from datetime import datetime

static_back = None
time = []

df = pandas.DataFrame(collumns = ["Start","End"])

video = cv2.VideoCapture(0)

while True:
    check, frame = video.read()

    motion = 0

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21,21),0)

    if static_back is None:
        static_back = gray
        continue

    diff_frame = cv2.absdiff(static_back,gray)

    thresh_frame = cv2.threshold(diff_frame, 30)