import numpy as np
import cv2 as cv

haar_cascade = cv.CascadeClassifier('haar_face.xml')

people = ['Chris Hemsworth','Leonardo Dicaprio','Tom Cruise','Will Smith']

faces_recognizer = cv.face.LBPHFaceRecognizer_create()
faces_recognizer.read('face_trained.yml')

img = cv.imread(r'C:\Users\jeshw\OneDrive\Desktop\project\face recog\validation\test1.jpg')
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

face_rect = haar_cascade.detectMultiScale(gray,1.05,7)

for (x,y,w,h) in face_rect:
    faces_roi = gray[y:y+h, x:x+w]

    label,confidence = faces_recognizer.predict(faces_roi)
    print(f'label={people[label]} with a confidence of {confidence}')

    cv.putText(img,str(people[label]), (20,20), cv.FONT_HERSHEY_COMPLEX, 1.0, (0,255,0), 2 )
    cv.rectangle(img, (x,y), (x+w,y+h),(0,255,0), 2)

cv.imshow('Detected Face',img)
cv.waitKey(0)
