# Facial Emotion Detection 🎭

A real-time Facial Emotion Detection System built with Haar Cascade for face detection and a Convolutional Neural Network (CNN) for emotion classification. The system is deployed using Streamlit, allowing users to capture images via webcam or upload pictures, and classifies emotions such as Happy, Sad, and Neutral.

## 🚀 Features

Real-time face detection using Haar Cascade.

Emotion classification into Happy, Sad, and Neutral.

Streamlit Web App with intuitive UI.

Two input modes:

📷 Capture image from webcam.

⬆️ Upload an image from your device.

Visual results with bounding boxes and emotion labels.

📊 Emotion analysis summary with percentage distribution and bar chart.

## 📂 Dataset

This project is trained on the FER-2013 dataset (Kaggle), which contains over 35,000 labeled grayscale facial images of size 48x48 pixels.
For this project, only 3 classes were used: Happy, Sad, and Neutral.

📥 Dataset link: [Kaggle FER-2013](https://www.kaggle.com/datasets/deadskull7/fer2013)

## ⚙️ System Workflow

Face Detection → Haar Cascade identifies faces in the frame.

Preprocessing → Faces are cropped, converted to grayscale, resized to 48×48, and normalized.

Emotion Classification → CNN model predicts the emotion.

Results → Display emotion label on the face, along with an overall summary.

## 🧠 Model Details

Input shape: 48×48 grayscale images.

Architecture: Convolutional layers + MaxPooling → Flatten → Dense layers → Dropout → Softmax output.

Optimizer: Adam.

Loss function: Categorical Crossentropy.

Accuracy: ~85% on validation data.

## 💻 Tech Stack

Language: Python

Libraries: TensorFlow, Keras, OpenCV, NumPy, Streamlit, PIL

Dataset: FER-2013 (Kaggle)

## 📈 Results

Achieved ~85% accuracy on validation dataset.

Real-time emotion classification with minimal latency.

Professional Streamlit web interface with side-by-side analysis.

## 🔮 Future Enhancements

Extend classification to more emotions (Surprise, Fear, Disgust, Angry).

Improve accuracy with Transfer Learning (VGG16, ResNet, MobileNet).

Deploy as a cloud-hosted web application or Android app.
