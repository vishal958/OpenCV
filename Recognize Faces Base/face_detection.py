#Importing Libraries
import numpy as np
import cv2
import face_recognition
import os

#Path to known Images
path ='knownimages'

#List to store all known images 
images =[]

#Target labels
classNames = []

#Code to get all known target
myList = os.listdir(path)

#Loop to store known images and target labels
for cls in myList:
  currImage = cv2.imread(f'{path}/{cls}')
  images.append(currImage)
  classNames.append(os.path.splitext(cls)[0])

#Function to find Encoding of Image in Imagelist
def findEncodings(images):
  encodeList = []
  for img in images:
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    encoding = face_recognition.face_encodings(img)[0]
    encodeList.append(encoding)
  return encodeList
 
 
#Storing all image encoding in variable   
encodeListKnown = findEncodings(images)
print('Encoding Complete')

#Creating cv2 instance
cap = cv2.VideoCapture(0)

while True:
  success, img = cap.read()
  #Converting to RGB 
  imgS = cv2.resize(img,(0,0),None,0.25,0.25)
  imgS = cv2.cvtColor(imgS,cv2.COLOR_BGR2RGB)
  
  # Gettting current frame and encoding
  faceCurrentFrame = face_recognition.face_locations(imgS)
  encodingCurrentFrame = face_recognition.face_encodings(imgS,faceCurrentFrame)
  
  
  # Face distance calulation between known face and current frame of webcam.
  for encodeFace,faceLoc in zip(encodingCurrentFrame,faceCurrentFrame):
    matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
    faceDist = face_recognition.face_distance(encodeListKnown,encodeFace)
    # Storing the index which has minimum distance
    matchIndex = np.argmin(faceDist)
    
    if matches[matchIndex]:
      name = classNames[matchIndex].upper()
      y1,x2,y2,x1 = faceLoc
      y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
      cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
      cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
      # Adding name label text .
      cv2.putText(img,name,(x1,y2),cv2.FONT_HERSHEY_COMPLEX,0.8,(255,255,255),2)
  
  #Creating Named Window 
  cv2.namedWindow('WebCam',cv2.WINDOW_NORMAL)
  #Resize Window
  cv2.resizeWindow('WebCam',(1000,600))
  cv2.imshow('WebCam',img)
  cv2.waitKey(1)    
      
   
    
  
  
  




