import cv2
import numpy as np
from PIL import Image
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
#Doc hinh anh dau vao
img = cv2.imread('/home/pi/athang.jpg')
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print(gray_img.shape)
#tim kiem khuon mat trong anh
faces = face_cascade.detectMultiScale(gray_img, scaleFactor = 1.005, minNeighbors = 5, minSize = (200, 200))
print(type(faces))
print(faces)
#ve hinh chu nhat
for x,y,w,h in faces:
    roi_gray = gray_img[y:y+h,x:x+w]
    roi_color = img[y:y+h, x:x+w]
    img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    #eyes = eye_cascade.detectMutilScale(roi_gray)
    #for(ex,ey,ew,eh) in eyes:
    #    cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
resized = cv2.resize(img, (int(img.shape[1]/2),int(img.shape[0]/2)))
cv2.imshow("image",resized)
cv2.waitKey(500)
#cv2.destroyAllWindows();


