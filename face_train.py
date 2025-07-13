import os
import cv2 as cv
import numpy as np

# List of people (folder names inside the training directory)
people = ['Chris Hemsworth', 'Leonardo Dicaprio', 'Tom Cruise', 'Will Smith']

# Path to training data directory
DIR = r'***path***\face recognition LBPH\train'

# Load Haar Cascade face detector
haar_cascade = cv.CascadeClassifier('haar_face.xml')

# Lists to hold the face data and corresponding labels
features = []
labels = []

def create_train():
    """
    Reads each image from each person's folder,
    detects the face using Haar Cascade,
    and stores the face ROI and label in the features and labels lists.
    """
    for person in people:
        path = os.path.join(DIR, person)
        label = people.index(person)  # Numeric label for the person

        for img_name in os.listdir(path):
            img_path = os.path.join(path, img_name)

            # Load image and convert to grayscale
            img_array = cv.imread(img_path)
            if img_array is None:
                continue  # Skip unreadable files

            gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)

            # Detect face(s) in the image
            faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

            # Extract and store the region of interest (ROI) for each detected face
            for (x, y, w, h) in faces_rect:
                faces_roi = gray[y:y+h, x:x+w]
                features.append(faces_roi)
                labels.append(label)

# Run training data creation
create_train()
print('Training data collection completed.')

# Convert lists to numpy arrays
features = np.array(features, dtype='object')
labels = np.array(labels)

# Create and train the LBPH face recognizer
face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.train(features, labels)

# Save the trained model and the features/labels
face_recognizer.save('face_trained.yml')
np.save('features.npy', features)
np.save('labels.npy', labels)

print('Model training and saving complete.')
