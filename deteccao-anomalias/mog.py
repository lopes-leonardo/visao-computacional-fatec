import cv2
import numpy as np

cap = cv2.VideoCapture(0)
fgbg = cv2.createBackgroundSubtractorMOG2()

while True:
    _, frame = cap.read()

    fgmask = fgbg.apply(frame, learningRate=-1)

    cv2.imshow("video", frame)
    cv2.imshow("mask", fgmask)

    k = cv2.waitKey(30) & 0xff
    if k == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
