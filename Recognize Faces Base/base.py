import numpy as np
import cv2
import face_recognition



imgKnown = face_recognition.load_image_file('./ImageBasic/elon1.jpg')
imgKnown = cv2.cvtColor(imgKnown,cv2.COLOR_BGR2RGB)


imgTest = face_recognition.load_image_file('./ImageBasic/elon2.jpg')
imgTest = cv2.cvtColor(imgTest,cv2.COLOR_BGR2RGB)

faceloc = face_recognition.face_locations(imgKnown)[0]
encodeElon = face_recognition.face_encodings(imgKnown)[0]
cv2.rectangle(imgKnown,(faceloc[3],faceloc[0]),(faceloc[1],faceloc[2]),(255,0,0),2)


facelocTest = face_recognition.face_locations(imgTest)[0]
encodeTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest,(facelocTest[3],facelocTest[0]),(facelocTest[1],facelocTest[2]),(255,0,0),2)

results= face_recognition.compare_faces([encodeElon],encodeTest)
facedist = face_recognition.face_distance([encodeElon],encodeTest)
print(results,facedist)

cv2.putText(imgTest,f'{results}{round(facedist[0],2)}', (50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
cv2.imshow('Elon',imgKnown)
cv2.imshow('Elon Test',imgTest)
cv2.waitKey(0)




