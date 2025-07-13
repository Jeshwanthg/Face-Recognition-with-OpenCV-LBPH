# Face-Recognition-with-OpenCV-LBPH

This project implements a face recognition system using OpenCV's Haar Cascade Classifier for face detection and the LBPH (Local Binary Patterns Histograms) algorithm for face recognition.

It supports:
- Training on your own image dataset
- Predicting from test images

---

## 📁 Project Structure

face-recognition-lbph/
│
├── haar_face.xml # Haar cascade for face detection
├── face_train.py # Script to train the model
├── face_recognition.py # Script to test prediction on a test image
├── face_trained.yml # Saved trained model
├── features.npy # Saved face feature data
├── labels.npy # Saved labels
├── val/ # Folder with test images (validation)
│ └── test.jpg
├── train/ # Training dataset (1 folder per person)
│ ├── Chris Hemsworth/
│ ├── Leonardo Dicaprio/
│ ├── Tom Cruise/
│ └── Will Smith/
└── README.md # You're reading it!

<img width="593" height="424" alt="image" src="https://github.com/user-attachments/assets/84983772-ceb6-489b-baa6-b48064bfd220" />



🛠️ How It Works:

🧪 Training the Recognizer
The training script reads face images from subfolders inside train/, detects the face, extracts the region of interest (ROI), and trains an LBPH model:

python face_train.py

This will:

- Detect faces using haar_face.xml
- Train the LBPH face recognizer

Save:

- Trained model → face_trained.yml
- Feature data → features.npy
- Labels → labels.npy

📷 Running Face Recognition on a Test Image:

python face_recognition.py

This will:

- Load the trained model
- Load the test image (val/test.jpg)
- Detect faces
- Predict their labels
- Display the results with confidence score and bounding box

🧠 Sample Prediction Output:

"label=Tom Cruise with a confidence of 52.89"

#The lower the confidence, the better the match (LBPH distance measure).
