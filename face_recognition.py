import numpy as np
import cv2 as cv

# Load Haar cascade for face detection
haar_cascade = cv.CascadeClassifier('haar_face.xml')

# List of people used during training (labels must match training order)
people = ['Chris Hemsworth', 'Leonardo Dicaprio', 'Tom Cruise', 'Will Smith']

# Load the trained LBPH face recognizer model
faces_recognizer = cv.face.LBPHFaceRecognizer_create()
faces_recognizer.read('face_trained.yml')

# Load the test image
img = cv.imread(r'C:\Users\jeshw\OneDrive\Desktop\project\face recog\validation\test2.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # Convert to grayscale

# Detect faces in the test image
face_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=7)

# Loop through each detected face and recognize it
for (x, y, w, h) in face_rect:
    # Extract the region of interest (face area)
    faces_roi = gray[y:y+h, x:x+w]

    # Predict the label and confidence score using the recognizer
    label, confidence = faces_recognizer.predict(faces_roi)

    print(f'label = {people[label]} with a confidence of {confidence:.2f}')

    # Annotate the original image
    cv.putText(img, str(people[label]), (x, y - 10), cv.FONT_HERSHEY_COMPLEX, 1.0, (0, 255, 0), 2)
    cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Show the annotated image in a window
cv.imshow('Detected Face', img)
cv.waitKey(0)
cv.destroyAllWindows()
