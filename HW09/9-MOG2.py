import numpy as np
import cv2
import time
import datetime

color=((0, 205, 205),(154, 250, 0),(34,34,178),(211, 0, 148),(255, 118, 72),(137, 137, 139))

cap = cv2.VideoCapture("origin.MP4") # 文件名"origin.MP4

fgbg = cv2.createBackgroundSubtractorMOG2()

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(datetime.datetime.now().strftime("%A_%d_%B_%Y_%I_%M_%S%p")+'.avi',fourcc, 10.0, (1920,1080))


while True:
    ret, frame = cap.read()
    fgmask = fgbg.apply(frame)

    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, element)

    contours, hierarchy = cv2.findContours(fgmask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    count=0
    for cont in contours:
        Area = cv2.contourArea(cont)
        if Area < 2000:
            continue

        count += 1

        print("{}-prospect:{}".format(count,Area),end="  ")

        rect = cv2.boundingRect(cont)

        print("x:{} y:{}".format(rect[0],rect[1]))

        cv2.rectangle(frame,(rect[0],rect[1]),(rect[0]+rect[2],rect[1]+rect[3]),color[count%6],1)
        cv2.rectangle(fgmask,(rect[0],rect[1]),(rect[0]+rect[2],rect[1]+rect[3]),(0xff, 0xff, 0xff), 1)

        y = 10 if rect[1] < 10 else rect[1]
        cv2.putText(frame, str(count), (rect[0], y), cv2.FONT_HERSHEY_COMPLEX, 0.4, (0, 255, 0), 1)

    cv2.imshow('frame', frame)
    cv2.imshow('frame2', fgmask)
    out.write(frame)
    k = cv2.waitKey(30)&0xff
    if k == 27:
        break


out.release()
cap.release()
cv2.destoryAllWindows()
