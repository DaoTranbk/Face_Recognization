from picamera.array import PiRGBArray
from picamera import PiCamera
import warnings
warnings.filterwarnings("ignore")
import numpy as np 
import cv2
import time
#########################
from keras.models import model_from_yaml
from keras.preprocessing.image import img_to_array
#########################
from rPPG.rPPG_Extracter import *
from rPPG.rPPG_lukas_Extracter import *
import matplotlib.pyplot as plt
#########################

width = 400;
height = 350;
camera = PiCamera();
camera.resolution = (width, height);
camera.framerate = 28;
rawCapture = PiRGBArray(camera, size = (width, height));
time.sleep(0.000001)
# load YAML and create model
yaml_file = open("trained_model/RGB_rPPG_merge_softmax_.yaml", 'r')
loaded_model_yaml = yaml_file.read()
yaml_file.close()
model = model_from_yaml(loaded_model_yaml)
# load weights into new model
model.load_weights("trained_model/RGB_rPPG_merge_softmax_.h5")
print("[INFO] Model is loaded from disk")
# Create Local Binary Patterns Histograms for face recognization
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Load the trained mode
recognizer.read('trainer/trainer.yml')
font = cv2.FONT_HERSHEY_SIMPLEX

dim = (128,128)
def get_rppg_pred(frame):
    use_classifier = True  # Toggles skin classifier
    use_flow = False       # (Mixed_motion only) Toggles PPG detection with Lukas Kanade optical flow          
    sub_roi = []           # If instead of skin classifier, forhead estimation should be used set to [.35,.65,.05,.15]
    use_resampling = False  # Set to true with webcam 
    
    fftlength = 300
    fs = 20
    f = np.linspace(0,fs/2,fftlength/2 + 1) * 60;

    timestamps = []
    time_start = [0]

    break_ = False

    rPPG_extracter = rPPG_Extracter()
    rPPG_extracter_lukas = rPPG_Lukas_Extracter()
    bpm = 0
    
    dt = time.time()-time_start[0]
    time_start[0] = time.time()
    if len(timestamps) == 0:
        timestamps.append(0)
    else:
        timestamps.append(timestamps[-1] + dt)
        
    rPPG = []

    rPPG_extracter.measure_rPPG(frame,use_classifier,sub_roi) 
    rPPG = np.transpose(rPPG_extracter.rPPG)
    
        # Extract Pulse
    if rPPG.shape[1] > 10:
        if use_resampling :
            t = np.arange(0,timestamps[-1],1/fs)
            
            rPPG_resampled= np.zeros((3,t.shape[0]))
            for col in [0,1,2]:
                rPPG_resampled[col] = np.interp(t,timestamps,rPPG[col])
            rPPG = rPPG_resampled
        num_frames = rPPG.shape[1]

        t = np.arange(num_frames)/fs
    return rPPG
    

def make_pred(li):
    [single_img,rppg] = li
    single_img = cv2.resize(single_img, dim)
    single_x = img_to_array(single_img)
    single_x = np.expand_dims(single_x, axis=0)
    single_pred = model.predict([single_x,rppg])
    return single_pred


    
cascPath = 'rPPG/util/haarcascade_frontalface_default.xml'
faceCascade = cv2.CascadeClassifier(cascPath)


video_capture = cv2.VideoCapture(0)

collected_results = []
counter = 0          # count collected buffers
frames_buffer = 5    # how many frames to collect to check for
accepted_falses = 1  # how many should have zeros to say it is real
for frame in camera.capture_continuous(rawCapture, format = 'bgr', use_video_port = True):
    image = frame.array;
    # Capture frame-by-frame
    #ret, frame = video_capture.read()
    #if ret:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize = (100, 100),
        flags = cv2.CASCADE_SCALE_IMAGE
    )
        
        # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        sub_img = image[y:y+h,x:x+w]
        rppg_s = get_rppg_pred(sub_img)
        rppg_s = rppg_s.T

        pred = make_pred([sub_img,rppg_s])

        collected_results.append(np.argmax(pred))
        counter += 1

        #cv2.putText(image, "Real: "+str(pred[0][0]), (50,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), lineType=cv2.LINE_AA)
        #cv2.putText(image, "Fake: "+str(pred[0][1]), (50,60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), lineType=cv2.LINE_AA)
        if len(collected_results) == frames_buffer:
                #print(sum(collected_results))
            if sum(collected_results) <= accepted_falses:
                #cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.rectangle(image, (x-20,y-20), (x+w+20,y+h+20), (0,255,0), 4)
                Id = recognizer.predict(gray[y:y+h,x:x+w])[0]
                if(Id == 1):
                    name = "dao tc"
                
                elif(Id == 2):
                    name = "anh Phuc"
                elif (Id == 3):
                    name = "anh chien"
                elif (Id == 4):
                    name = "chi hien"
                elif(Id == 5):
                    name = "anh hung"
                elif(Id == 6 ):
                    name = "anh Xung"
                elif(Id == 7 ):
                    name = "anh Phong"
                elif(Id == 9):
                    name = "Phuc nq"
                else:
                    name = "Unknow";
                cv2.rectangle(image, (x-22,y-90), (x+w+22, y-22), (0,255,0), -1)
                cv2.putText(image, str(name), (x,y-60), font, 2, (255,255,255), 4)
            else:
                cv2.rectangle(image, (x-20,y-20), (x+w+20,y+h+20), (0,0,255), 4)
            collected_results.pop(0)

        

        # Display the resulting frame
    cv2.imshow('To quit press q', image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    rawCapture.truncate(0);

# When everything is done, release the capture
#video_capture.release()
#cv2.destroyAllWindows()
