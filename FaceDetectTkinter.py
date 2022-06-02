from tkinter import * 
from PIL import ImageTk, Image
import cv2
import dlib
import numpy as np
#from tensorflow.keras.models import load_model
#from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
#from tensorflow.keras.preprocessing.image import img_to_array
import serial 
import time
import threading

detector = dlib.get_frontal_face_detector() 
predictor = dlib.shape_predictor("MODELS/shape_predictor_68_face_landmarks.dat") 
#model = load_model('MODELS/mask_detector.h5')
font = ImageFont.truetype("Fonts/segoeui.ttf",40)
fontSmall = ImageFont.truetype("Fonts/segoeui.ttf",30)
ambientTemp = 0
bodyTemp = 0
dist = 0

screenWidth = 640
screenHeight = 480
faceAreaWidth = 160
faceAreaHeight = 190

foreheadCoord = 0
faceRadius = 80

#ser = serial.Serial('COM6',9600)
root = Tk()
# Create a frame
app = Frame(root, bg="white")
app.grid()
# Create a label in the frame
lmain = Label(app)
lmain.grid()

ltextvar = StringVar()

ltext = Label(root, textvariable = ltextvar, font="SegoeUI").grid(row=1, column=0)
# Capture from camera
cap = cv2.VideoCapture(0)

# function for video streaming
def video_stream():
    _, img = cap.read()
    img = cv2.flip(img,1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print(maskDetect(img))
    faces = detector(gray) 
    cv2.ellipse(img,(screenWidth//2,screenHeight//2-20),(faceAreaWidth,faceAreaHeight),0,0,360,(255,255,255),1,cv2.LINE_AA)
    ltextvar.set("Please stand at the designated place.")
    
    #Face Landscape
    for face in faces: 
        x1 = face.left() 
        y1 = face.top() 
        x2 = face.right() 
        y2 = face.bottom() 
        centerCoord = ((x1+x2)//2,(y1+y2)//2)
        #cv2.circle(img,centerCoord,faceRadius,(0,255,0),1,cv2.LINE_AA)
        # Then we can also do cv2.rectangle function (frame, (x1, y1), (x2, y2), (0, 255, 0), 3) 
        landmarks = predictor(gray, face) 
        foreheadCoord = landmarks.part(28) - dlib.point(0,50)
       
        cv2.circle(img, (foreheadCoord.x,foreheadCoord.y),2,(0, 0, 255), -1)
        cv2.circle(img, (x1,(y1+y2)//2),2,(0, 0, 255), -1)
        cv2.circle(img, (x2,(y1+y2)//2),2,(0, 0, 255), -1)
        cv2.circle(img, ((x1+x2)//2,y1),2,(0, 0, 255), -1)
        cv2.circle(img, ((x1+x2)//2,y2),2,(0, 0, 255), -1)
     
        if(x1 < screenWidth//2-faceAreaWidth): 
            ltextvar.set("Please move to the right.")
            cv2.ellipse(img,(screenWidth//2,screenHeight//2-20),(faceAreaWidth,faceAreaHeight),0,0,360,(0,0,255),1,cv2.LINE_AA)
        elif(x2 > screenWidth//2+faceAreaWidth):
            ltextvar.set("Please move to the left.")           
            cv2.ellipse(img,(screenWidth//2,screenHeight//2-20),(faceAreaWidth,faceAreaHeight),0,0,360,(0,0,255),1,cv2.LINE_AA)
        elif(y1-20 < screenHeight//2-faceAreaHeight):
            ltextvar.set("Please move downwards.")            
            cv2.ellipse(img,(screenWidth//2,screenHeight//2-20),(faceAreaWidth,faceAreaHeight),0,0,360,(0,0,255),1,cv2.LINE_AA)
        elif(y2+20 > screenHeight//2+faceAreaHeight):
            ltextvar.set("Please move upwards.")            
            cv2.ellipse(img,(screenWidth//2,screenHeight//2-20),(faceAreaWidth,faceAreaHeight),0,0,360,(0,0,255),1,cv2.LINE_AA)
        elif(dist >  60):
             ltextvar.set("Please stand closer.")
        else:
            ltextvar.set("Temperature :"+str(bodyTemp))
            cv2.ellipse(img,(screenWidth//2,screenHeight//2-20),(faceAreaWidth,faceAreaHeight),0,0,360,(0,255,0),1,cv2.LINE_AA)
        # We are then accesing the landmark points  
        for n in range(0, 68): 
            x = landmarks.part(n).x 
            y = landmarks.part(n).y 
            cv2.circle(img, (x, y), 2, (255, 255, 0), -1) 
    cv2image = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
    img1 = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img1)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)
    lmain.after(1, video_stream) 
    
def serialCapture():
    global ambientTemp
    global bodyTemp
    global dist
    while True:
        time.sleep(0.5)
        b = ser.readline()
        ambientTemp = float(b.decode().split(',')[0])
        print(ambientTemp)  
        bodyTemp = float(b.decode().split(',')[1])
        print(bodyTemp) 
        dist = float(b.decode().split(',')[2])
        print(dist) 
        bodyTemp = float(bodyTemp) + 15

def maskDetect(img):
    img_copy = img.copy()
    resized = cv2.resize(img_copy,(254, 254))
    resized = img_to_array(resized)
    resized = preprocess_input(resized)
    resized = np.expand_dims(resized, axis=0)
    mask,_ = model.predict([resized])[0]
    return mask

video_stream()
thread1 = threading.Thread(name='Serial', target=serialCapture)
#thread1.start()
root.mainloop()