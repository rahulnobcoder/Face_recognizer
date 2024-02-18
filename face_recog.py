import cv2 as cv
import numpy as np
import model
import os

p=[]
for i in os.listdir(r'D:\college\sem_6\img_prc\face rec\Faces\train'):
    p.append(i)
dir=r'D:\college\sem_6\img_prc\face rec\Faces\train'
print(p)
model.train(dir,p)
cap=cv.VideoCapture(0)
while True:
    ret,frame=cap.read()
    if ret:
        c_img=frame
        img=cv.cvtColor(c_img,cv.COLOR_BGR2GRAY)
        c_img=model.predict(img,c_img,p)
        cv.imshow('face',c_img)
        if cv.waitKey(25) & 0xFF==ord('q'):
            break
    else:
        break
