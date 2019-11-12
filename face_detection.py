from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2
width = 640;
height = 480;
camera = PiCamera();
camera.resolution = (width, height);
camera.framerate = 60;
rawCapture = PiRGBArray(camera, size = (width, height));
time.sleep(0.0001)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml');
print("\n Waiting...");
for frame in camera.capture_continuous(rawCapture, format = 'bgr', use_video_port = True):
    image = frame.array;
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY);
    faces = face_cascade.detectMultiScale(gray, scaleFactor = 1.1, minNeighbors = 5, minSize = (100, 100), flags = cv2.CASCADE_SCALE_IMAGE);
#    print("Found" + str(len(faces)) + "faces");
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w];
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2);
#        print(x, y, w, h);
    cv2.imshow('Frame', image);
    cv2.waitKey(1);
    rawCapture.truncate(0);