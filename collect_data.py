import cv2
import imutils
from FaceDetector import *
import os 

dir ="dataset/"
name ="Bangau_open"
path = dir+name+"/"
if not os.path.exists(path):
    os.makedirs(path)
	
	
camera = cv2.VideoCapture(1)
fd = faceDetector('fd_models/haarcascade_frontalface_default.xml')
while True:
    ret, frame = camera.read()
    frame = imutils.resize(frame, width = 800)
    k = cv2.waitKey(3) & 0xFF
    if k == ord('k'):
        for i in range(100):
            
            ret, frame = camera.read()
            frame = imutils.resize(frame, width = 800)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
            faceRects = fd.detect(gray)
            for (x, y, w, h) in faceRects:
                roi = frame[y:y+h,x:x+w]
                #roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                roi = cv2.resize(roi,(100, 100))
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.imwrite(path+'%d.jpg'%i,roi)
            cv2.imshow('frame', frame)
            cv2.waitKey(1)
        break
                 
    cv2.imshow('frame', frame)

camera.release()
cv2.destroyAllWindows()