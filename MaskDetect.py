import cv2
import dlib
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

model = load_model('MODELS/mask_detector.h5')
ctr = 0

cap = cv2.VideoCapture(0)

def maskDetect(img):
    img_copy = img.copy()
    resized = cv2.resize(img_copy,(224, 224))
    resized = img_to_array(resized)
    resized = preprocess_input(resized)
    resized = np.expand_dims(resized, axis=0)
    mask,_ = model.predict([resized])[0]
    return mask

while True:
    _, img = cap.read()
    #img = cv2.imread("mask.jpg",0)
    print(maskDetect(img))
    if(maskDetect(img) > 0.5):
        ctr += 1
    if(ctr > 5):
        print("True")
        ctr = 0
    cv2.imshow('Video',img)
    key = cv2.waitKey(1) 
    if key == 27: 
        break # press esc the frame is destroyed    
