# Brain Tumor Classification

This repository contains a deep learning pipeline for classifying brain tumors from MRI images. Using both a custom Convolutional Neural Network (CNN) and a transfer learning approach with the Xception model, this project identifies four categories of brain conditions: **Glioma**, **Meningioma**, **Pituitary tumors**, and **No Tumor**.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Model Training](#model-training)
- [Web Application](#web-application)
- [Usage](#usage)
- [Results](#results)
- [Conclusion](#conclusion)

## Overview

Brain tumor classification is a critical step in medical diagnostics, helping healthcare professionals make timely and accurate treatment decisions. This project applies advanced machine learning techniques to automate the classification process, providing interpretable and reliable results through a user-friendly web application.

## Dataset

The dataset used in this project is sourced from Kaggle: [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset).

The dataset includes MRI images categorized into four classes:

- **Glioma**
- **Meningioma**
- **Pituitary Tumors**
- **No Tumor**

Each class contains sufficient examples for effective training and testing.

## Model Training

This project implements two deep learning models for classification:

1. **Custom CNN**:
   - A neural network designed from scratch with convolutional, pooling, and dropout layers to optimize feature extraction and prevent overfitting.

2. **Xception Model**:
   - A pre-trained model fine-tuned for this classification task using transfer learning.

### Training Details:
- Images are preprocessed (resized and normalized) to ensure consistency.
- Models are trained for **5 epochs** with data augmentation.
- Performance metrics such as **accuracy**, **precision**, and **recall** are tracked during training.

### Training Code:
```python
img_shape = (299, 299, 3)

base_model = tf.keras.applications.Xception(
    include_top=False, weights="imagenet", input_shape=img_shape, pooling="max"
)

model = Sequential([
    base_model,
    Flatten(),
    Dropout(0.3),
    Dense(128, activation="relu"),
    Dropout(rate=0.25),
    Dense(4, activation="softmax"),
])

model.compile(
    optimizer=Adamax(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy", Precision(), Recall()]
)

hist = model.fit(
    tr_gen, epochs=5, validation_data=valid_gen
) 
```
## Results

The models achieved the following performance metrics:

- **Custom CNN**:
  - Accuracy: 95.48%
  - Precision: 95.83%
  - Recall: 95.35%

- **Xception Model**:
  - Accuracy: 99.71%
  - Precision: 99.71%
  - Recall: 99.69%

## Web Application

An interactive **Streamlit** web application was developed to allow users to upload MRI images and receive real-time predictions.

### Features:
- **Model Selection**: Users can choose between the Xception model and the custom CNN.
- **Image Upload**: Upload MRI scans in JPG, JPEG, or PNG formats.
- **Prediction Results**: Displays the predicted class and confidence level.
- **Saliency Maps**: Highlights regions of the brain that influenced the prediction.
- **Probability Visualization**: Provides a bar chart of probabilities for each class.

### `app.py` Code
```python
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import plotly.graph_objects as go
import cv2
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Constants
output_dir = 'saliency_maps'
os.makedirs(output_dir, exist_ok=True)

# Helper Functions
def generate_saliency_map(model, img_array, class_index, img_size):
    with tf.GradientTape() as tape:
        img_tensor = tf.convert_to_tensor(img_array)
        tape.watch(img_tensor)
        predictions = model(img_tensor)
        target_class = predictions[:, class_index]

    gradients = tape.gradient(target_class, img_tensor)
    gradients = tf.math.abs(gradients)
    gradients = tf.reduce_max(gradients, axis=-1)
    gradients = gradients.numpy().squeeze()

    # Normalize and process gradients
    gradients = cv2.resize(gradients, img_size)
    gradients = cv2.GaussianBlur(gradients, (11, 11), 0)
    heatmap = cv2.applyColorMap(np.uint8(255 * gradients), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    return heatmap

# Load Pre-trained Models
def load_xception_model(model_path):
    img_shape = (299, 299, 3)
    base_model = tf.keras.applications.Xception(include_top=False, weights='imagenet', input_shape=img_shape, pooling='max')
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Dense(4, activation='softmax')
    ])
    model.load_weights(model_path)
    return model

# Streamlit App UI
st.title('Brain Tumor Classification')
st.write('Upload an MRI scan to classify it into one of the four categories: Glioma, Meningioma, Pituitary Tumor, or No Tumor.')

uploaded_file = st.file_uploader("Choose an Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    selected_model = st.radio("Select Model", ("Transfer Learning - Xception", "Custom CNN"))
    if selected_model == "Transfer Learning - Xception":
        model = load_xception_model('/content/xception_model.weights.h5')
        img_size = (299, 299)
    else:
        model = load_model('/content/cnn_model.h5')
        img_size = (224, 224)

    labels = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
    img = image.load_img(uploaded_file, target_size=img_size)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    class_index = np.argmax(prediction[0])
    result = labels[class_index]

    st.write(f'**Predicted Class:** {result}')
    st.write('**Class Probabilities:**')
    for label, prob in zip(labels, prediction[0]):
        st.write(f"{label}: {prob * 100:.2f}%")

    saliency_map = generate_saliency_map(model, img_array, class_index, img_size)
    col1, col2 = st.columns(2)
    with col1:
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    with col2:
        st.image(saliency_map, caption="Saliency Map", use_column_width=True)

    # Bar Chart
    probabilities = prediction[0]
    sorted_indices = np.argsort(probabilities)[::-1]
    sorted_labels = [labels[i] for i in sorted_indices]
    sorted_probabilities = probabilities[sorted_indices]

    fig = go.Figure(go.Bar(
        x=sorted_probabilities,
        y=sorted_labels,
        orientation='h',
        marker_color=['red' if label == result else 'blue' for label in sorted_labels]
    ))
    fig.update_layout(
        title="Class Probabilities",
        xaxis_title="Probability",
        yaxis_title="Class",
        height=400,
        width=600
    )
    st.plotly_chart(fig)
```
## Deploying the Web Application

The web application can be deployed publicly using **ngrok**. Below are the steps to set it up and generate a public URL for your Streamlit app.

### Deployment Code
```python
! pip install streamlit pyngrok python-dotenv

from threading import Thread
from pyngrok import ngrok
import os

# Set your ngrok authentication token
ngrok_token = "your_ngrok_token"  # Replace with your ngrok token
ngrok.set_auth_token(ngrok_token)

def run_streamlit():
    os.system("streamlit run /content/app.py --server.port 8501")

# Start the Streamlit app in a separate thread
thread = Thread(target=run_streamlit)
thread.start()

# Generate a public URL with ngrok
public_url = ngrok.connect(addr='8501')
print(f"Access the app here: {public_url}")
```
### Instructions

1. **Install Required Dependencies**:
   Install `streamlit`, `pyngrok`, and `python-dotenv` to set up and deploy the app:
   ```bash
   pip install streamlit pyngrok python-dotenv

2. **Run the Deployment Script**:
   Use the following Python script to start the Streamlit app and generate a public URL using ngrok:
   ```python
   from threading import Thread
   from pyngrok import ngrok
   import os

   # Set your ngrok authentication token
   ngrok_token = "your_ngrok_token"  # Replace with your ngrok token
   ngrok.set_auth_token(ngrok_token)

   def run_streamlit():
       os.system("streamlit run app.py --server.port 8501")

   # Start the Streamlit app in a separate thread
   thread = Thread(target=run_streamlit)
   thread.start()

   # Generate a public URL with ngrok
   public_url = ngrok.connect(addr='8501')
   print(f"Access the app here: {public_url}")

3. **Access the App**:
   - The app is running locally on port `8501`.
   - A public URL has been generated using ngrok: [Access the App](https://1f9b-35-204-249-245.ngrok-free.app/)
   - Share this link with others or use it to access the app on any device with internet access.

