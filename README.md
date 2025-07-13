# Face-Recognition-with-OpenCV-LBPH

This project implements a face recognition system using OpenCV's Haar Cascade Classifier for face detection and the LBPH (Local Binary Patterns Histograms) algorithm for face recognition.

It supports:
- Training on your own image dataset
- Predicting from test images

---

## 📁 Project Structure


<img width="768" height="563" alt="image" src="https://github.com/user-attachments/assets/469cf364-92e6-441d-8969-ce5f962ecc13" />



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
