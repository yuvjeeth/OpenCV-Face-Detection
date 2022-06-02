from PIL import ImageTk, Image, ImageFont, ImageDraw
import cv2
import dlib
import numpy as np
import serial 
import time
import threading
from playsound import playsound
from pygame import mixer
from picamera.array import PiRGBArray
from picamera import PiCamera

##Initialize resources
mixer.init()
detector = dlib.get_frontal_face_detector() 
predictor = dlib.shape_predictor("MODELS/shape_predictor_68_face_landmarks.dat") 
font = ImageFont.truetype("Fonts/segoeui.ttf",40)
titleImage = cv2.imread("Resources/UIElements/Title.png")
leftImage = cv2.imread("Resources/UIElements/MoveLeft.png")
rightImage = cv2.imread("Resources/UIElements/MoveRight.png")
upImage = cv2.imread("Resources/UIElements/MoveUp.png")
downImage = cv2.imread("Resources/UIElements/MoveDown.png")

##Serial function values
ambientTemp = 0
bodyTemp = 0
dist = 0

##Global Declarations
screenWidth = 640
screenHeight = 360
baseWidth = 1280
baseHeight = 720
faceAreaWidth = 110
faceAreaHeight = 130
foreheadCoord = 0
statusBG = [(0,baseHeight - 80),(baseWidth-1,baseHeight-1)]
resultBG = [(0,0),(baseWidth - 1, 80)]
widthFactor =  80
distFromSensor = 60

# Capture from camera PiCamera
camera = PiCamera()
camera.resolution = (640, 360)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=(640, 360))
time.sleep(0.1)

# Capture from camera USB camera
#cap = cv2.VideoCapture(0)

tempDispDelay = 0
resultPlayed = False
checkingPlayed = False
faceDetected = False
serialCaptureStop = False
resultInt = -1
faceAreaLoc = (screenWidth//2, screenHeight//2)
faceAreaCoordinates = [(screenWidth//2 - faceAreaWidth - widthFactor, screenHeight//2 - faceAreaHeight), (screenWidth//2 + faceAreaWidth  + widthFactor, screenHeight//2 - faceAreaHeight), (screenWidth//2 + faceAreaWidth + widthFactor, screenHeight//2 + faceAreaHeight), (screenWidth//2 - faceAreaWidth - widthFactor, screenHeight//2 + faceAreaHeight)]

def serialCapture():
    global ambientTemp
    global bodyTemp
    global dist
    global serialCaptureStop
    #Initialize Serial Port
    ser = serial.Serial('/dev/ttyUSB0',9600)
    while True:
        time.sleep(0.5)
        b = ser.readline()
        ambientTemp = float(b.decode().split(',')[0])
        bodyTemp = float(b.decode().split(',')[1])
        print(bodyTemp)
        dist = float(b.decode().split(',')[2])
        bodyTemp = float(bodyTemp) + 15
        if(serialCaptureStop == True):
            break

thread1 = threading.Thread(name='Serial', target=serialCapture)
#thread1.start()

# Loop for video streaming
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    imgBase = np.zeros((720,1280,3), np.uint8)
    #_, img = cap.read() USB Camera
    img = frame.array
    img = cv2.flip(img,1)
    roi = img[faceAreaCoordinates[0][1]:faceAreaCoordinates[2][1] +1, faceAreaCoordinates[0][0]:faceAreaCoordinates[2][0] +1]
    grayROI = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    faces = detector(grayROI)  

    cv2.circle(img, faceAreaCoordinates[0],2,(255, 0, 255), 2)
    cv2.circle(img, faceAreaCoordinates[1],2,(255, 0, 255), 2)
    cv2.circle(img, faceAreaCoordinates[2],2,(255, 0, 255), 2)
    cv2.circle(img, faceAreaCoordinates[3],2,(255, 0, 255), 2)

    if(resultInt == 0):
        mixer.music.load("Resources/Stop.mp3")
        mixer.music.play()
    
    elif(resultInt == 1):
        mixer.music.load("Resources/Proceed.mp3")
        mixer.music.play()
       
    if(resultPlayed == True):
        resultInt = -1

    if(len(faces)==0):
        cv2.ellipse(img,faceAreaLoc,(faceAreaWidth,faceAreaHeight),0,0,360,(255,255,255),2,cv2.LINE_AA)
        imgRgb = cv2.cvtColor(imgBase, cv2.COLOR_BGR2RGB)
        pilImg = Image.fromarray(imgRgb)
        draw = ImageDraw.Draw(pilImg)
        draw.rectangle(statusBG,fill ='#3483eb',outline='#006eff')
        draw.text((410,635),"Detecting Person...", font=font)
        imgProcessed = cv2.cvtColor(np.array(pilImg), cv2.COLOR_RGB2BGR)
        imgBase = imgProcessed
        tempDispDelay = 0
        faceDetected = False
        if(resultPlayed == True):
            checkingPlayed = False
            resultPlayed = False

    #Face Landscape
    for face in faces: 
        x1 = face.left() 
        y1 = face.top() 
        x2 = face.right() 
        y2 = face.bottom() 
        centerCoord = ((x1+x2)//2,(y1+y2)//2)
        #cv2.circle(img,centerCoord,faceRadius,(0,255,0),1,cv2.LINE_AA)
        # Then we can also do cv2.rectangle function (frame, (x1, y1), (x2, y2), (0, 255, 0), 3) 

    for face in faces: 
        x1 = face.left() 
        y1 = face.top() 
        x2 = face.right() 
        y2 = face.bottom() 
        centerCoord = ((x1+x2)//2,(y1+y2)//2)
        #cv2.circle(img,centerCoord,faceRadius,(0,255,0),1,cv2.LINE_AA)
        # Then we can also do cv2.rectangle function (frame, (x1, y1), (x2, y2), (0, 255, 0), 3) 

        if(x1 + faceAreaCoordinates[0][0] < faceAreaLoc[0]-faceAreaWidth): 
            if(faceDetected == False):
                tempDispDelay = 0
                resultPlayed = False
                cv2.ellipse(img,faceAreaLoc,(faceAreaWidth,faceAreaHeight),0,0,360,(27,170,247),2,cv2.LINE_AA)
                #cv2.arrowedLine(imgBase, (baseWidth//2 + 300,baseHeight//2),(baseWidth//2 + 600,baseHeight//2),(255,55,0),15,tipLength=0.5)
                imgBase[250 : 250 + rightImage.shape[0], 950 : 950 + rightImage.shape[1]] = rightImage
                imgRgb = cv2.cvtColor(imgBase, cv2.COLOR_BGR2RGB)
                pilImg = Image.fromarray(imgRgb)
                draw = ImageDraw.Draw(pilImg)
                draw.rectangle(statusBG,fill ='#3483eb',outline='#006eff')
                draw.text((410,635),"Please Move Right", font=font)
                imgProcessed = cv2.cvtColor(np.array(pilImg), cv2.COLOR_RGB2BGR)
                imgBase = imgProcessed
                continue

        if(x2 + faceAreaCoordinates[0][0] > faceAreaLoc[0]+faceAreaWidth):
            if(faceDetected == False):
                tempDispDelay = 0
                resultPlayed = False
                cv2.ellipse(img,faceAreaLoc,(faceAreaWidth,faceAreaHeight),0,0,360,(27,170,247),2,cv2.LINE_AA)
                imgBase[250 : 250 + leftImage.shape[0], 50 : 50 + leftImage.shape[1]] = leftImage
                imgRgb = cv2.cvtColor(imgBase, cv2.COLOR_BGR2RGB)
                pilImg = Image.fromarray(imgRgb)
                draw = ImageDraw.Draw(pilImg)
                draw.rectangle(statusBG,fill ='#3483eb',outline='#006eff')
                draw.text((410,635),"Please Move Left", font=font)
                imgProcessed = cv2.cvtColor(np.array(pilImg), cv2.COLOR_RGB2BGR)
                imgBase = imgProcessed
                continue

        if(y1 + faceAreaCoordinates[0][1] < faceAreaLoc[1]-faceAreaHeight):
            if(faceDetected == False):
                tempDispDelay = 0
                resultPlayed = False
                cv2.ellipse(img,faceAreaLoc,(faceAreaWidth,faceAreaHeight),0,0,360,(27,170,247),2,cv2.LINE_AA)
                imgBase[250 : 250 + downImage.shape[0], 70 : 70 + downImage.shape[1]] = downImage
                imgRgb = cv2.cvtColor(imgBase, cv2.COLOR_BGR2RGB)
                pilImg = Image.fromarray(imgRgb)
                draw = ImageDraw.Draw(pilImg)
                draw.rectangle(statusBG,fill ='#3483eb',outline='#006eff')
                draw.text((380,635),"Please Move Down", font=font)
                imgProcessed = cv2.cvtColor(np.array(pilImg), cv2.COLOR_RGB2BGR)
                imgBase = imgProcessed
                continue

        if(y2 + faceAreaCoordinates[0][1] > faceAreaLoc[1]+faceAreaHeight): 
            if(faceDetected == False):
                tempDispDelay = 0
                resultPlayed = False
                cv2.ellipse(img,faceAreaLoc,(faceAreaWidth,faceAreaHeight),0,0,360,(27,170,247),2,cv2.LINE_AA) 
                imgBase[250 : 250 + upImage.shape[0], 1000 : 1000 + upImage.shape[1]] = upImage
                imgRgb = cv2.cvtColor(imgBase, cv2.COLOR_BGR2RGB)
                pilImg = Image.fromarray(imgRgb)
                draw = ImageDraw.Draw(pilImg)
                draw.rectangle(statusBG,fill ='#3483eb',outline='#006eff')
                draw.text((430,635),"Please Move Up", font=font)
                imgProcessed = cv2.cvtColor(np.array(pilImg), cv2.COLOR_RGB2BGR)
                imgBase = imgProcessed
                continue


        if(dist >  distFromSensor):
            if(faceDetected == False):
                tempDispDelay = 0
                resultPlayed = False
                cv2.ellipse(img,faceAreaLoc,(faceAreaWidth,faceAreaHeight),0,0,360,(27,170,247),2,cv2.LINE_AA)
                imgRgb = cv2.cvtColor(imgBase, cv2.COLOR_BGR2RGB)
                pilImg = Image.fromarray(imgRgb)
                draw = ImageDraw.Draw(pilImg)
                draw.rectangle(statusBG,fill ='#3483eb',outline='#006eff')
                draw.text((370,635),"Please Stand Closer", font=font)
                imgProcessed = cv2.cvtColor(np.array(pilImg), cv2.COLOR_RGB2BGR)
                imgBase = imgProcessed
                continue

        #landmarks = predictor(grayROI, face)
        #foreheadCoord = landmarks.part(28) - dlib.point(0,50)
        #cv2.circle(roi, (foreheadCoord.x,foreheadCoord.y),2,(0, 0, 255), -1)
        cv2.circle(roi, (x1,(y1+y2)//2),2,(0, 0, 255), -1)
        cv2.circle(roi, (x2,(y1+y2)//2),2,(0, 0, 255), -1)
        cv2.circle(roi, ((x1+x2)//2,y1),2,(0, 0, 255), -1)
        cv2.circle(roi, ((x1+x2)//2,y2),2,(0, 0, 255), -1)
        
        tempDispDelay += 1
        if(tempDispDelay > 25):
            if(bodyTemp <= 99):
                cv2.ellipse(img,faceAreaLoc,(faceAreaWidth,faceAreaHeight),0,0,360,(0,255,0),2,cv2.LINE_AA)
                imgRgb = cv2.cvtColor(imgBase, cv2.COLOR_BGR2RGB)
                pilImg = Image.fromarray(imgRgb)
                draw = ImageDraw.Draw(pilImg)
                #draw.rectangle(resultBG,fill ='#3fde1f',outline='#4cf72a')
                #draw.text((420,-5),"You Can Proceed", font=font)
                draw.rectangle(statusBG,fill ='#3fde1f',outline='#4cf72a')
                draw.text((410,635), "Temperature: " +str(bodyTemp) +"F", font=font)
                imgProcessed = cv2.cvtColor(np.array(pilImg), cv2.COLOR_RGB2BGR)
                imgBase = imgProcessed
                faceDetected = True
                if(resultPlayed == False):
                    resultInt = 1
                    resultPlayed = True
            else:
                cv2.ellipse(img,faceAreaLoc,(faceAreaWidth,faceAreaHeight),0,0,360,(0,0,255),2,cv2.LINE_AA)
                imgRgb = cv2.cvtColor(imgBase, cv2.COLOR_BGR2RGB)
                pilImg = Image.fromarray(imgRgb)
                draw = ImageDraw.Draw(pilImg)
                #draw.rectangle(resultBG,fill ='#ff2d26',outline='#ff0800')
                #draw.text((490,-5),"Please Stop", font=font)
                draw.rectangle(statusBG,fill ='#ff2d26',outline='#ff0800')
                draw.text((400,635), "Temperature: " +str(bodyTemp) +"F", font=font)
                imgProcessed = cv2.cvtColor(np.array(pilImg), cv2.COLOR_RGB2BGR)
                imgBase = imgProcessed
                faceDetected = True
                if(resultPlayed == False):
                    resultInt = 0
                    resultPlayed = True
        else:
            cv2.ellipse(img,faceAreaLoc,(faceAreaWidth,faceAreaHeight),0,0,360,(0,225,234),2,cv2.LINE_AA)
            imgRgb = cv2.cvtColor(imgBase, cv2.COLOR_BGR2RGB)
            pilImg = Image.fromarray(imgRgb)
            draw = ImageDraw.Draw(pilImg)
            draw.rectangle(statusBG,fill ='#ff9900',outline='#ffb03b')
            draw.text((320,635),"Checking Temperature...", font=font)
            imgProcessed = cv2.cvtColor(np.array(pilImg), cv2.COLOR_RGB2BGR)
            imgBase = imgProcessed
            if(checkingPlayed == False):
                checkingPlayed = True
                mixer.music.load("Resources/Checking.mp3")
                mixer.music.play()

    imgCrop = img[screenHeight//2 - faceAreaHeight - 10 :screenHeight//2 + faceAreaHeight + 10, screenWidth//2 - faceAreaWidth -10 :screenWidth//2 + faceAreaWidth + 10]
    imgCrop = cv2.resize(imgCrop,(490,550))
    imgProcessed = cv2.cvtColor(np.array(pilImg), cv2.COLOR_RGB2BGR) 
    imgBase = imgProcessed
    imgBase[0:titleImage.shape[0],0:titleImage.shape[1]] = titleImage
    imgBase[75: 75 + imgCrop.shape[0], 400 : 400 + imgCrop.shape[1]] = imgCrop

    #cv2.namedWindow("Video", cv2.WND_PROP_FULLSCREEN)
    #cv2.setWindowProperty("Video",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
    cv2.imshow('Video',imgBase)

    key = cv2.waitKey(1)
    rawCapture.truncate(0)
    if key == 27: 
        serialCaptureStop = True
        break # Esc to exit code    
