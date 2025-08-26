# Facial Emotion Detection 🎭

A real-time Facial Emotion Detection System that integrates Haar Cascade for face detection and a Convolutional Neural Network (CNN) for emotion classification. The application processes live video streams or static images and classifies emotions such as Happy, Sad, and Neutral using Python, TensorFlow, and OpenCV.

##  🚀 Features

Real-time face detection using Haar Cascade

Emotion classification with CNN (Happy, Sad, Neutral)

Supports live video stream via webcam or static images

Lightweight and efficient for real-time applications

Built with Python, TensorFlow, and OpenCV

## 📂 Dataset

This project is trained on the FER-2013 dataset (Kaggle), which contains 35,000+ labeled grayscale images of size 48x48 pixels.
For this project, only 3 classes were used: Happy, Sad, and Neutral.

## ⚙️ System Architecture

Face Detection – Haar Cascade detects the face from video frames/images.

Preprocessing – Cropped faces are resized to 48x48 grayscale and normalized.

Emotion Classification – CNN model predicts the emotion label.

Output – The detected emotion is displayed on screen in real-time.

## 🧠 Model Details

Input Shape: 48x48 grayscale images

Layers:

Conv2D + MaxPooling (feature extraction)

Flatten + Dense layers

Dropout to prevent overfitting

Softmax output layer with 3 classes

Optimizer: Adam

Loss Function: Categorical Crossentropy

Accuracy: ~85% on validation data

## 💻 Tech Stack

Programming Language: Python

Libraries/Frameworks: TensorFlow, Keras, OpenCV, NumPy

Dataset: FER-2013 (Kaggle)

## 📈 Results

Achieved ~85% accuracy on the validation dataset.

Real-time predictions with minimal lag.

🔮 Future Enhancements

Add more emotion classes (Surprise, Fear, Disgust, Angry).

Improve accuracy using transfer learning (VGG16, MobileNet, EfficientNet).
