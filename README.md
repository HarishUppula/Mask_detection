# Face Mask Detection using CNN and Streamlit

## Project Overview

This project detects whether a person is wearing a face mask or not using a Convolutional Neural Network (CNN) model and a Streamlit web application.

The application supports:

* Real-time webcam detection
* Image upload prediction
* Live prediction display as **Mask** or **No Mask**

The trained model is stored as `mask_detection.h5` and loaded directly into the Streamlit app for fast predictions.

---

## Project Structure

```text
Face_Mask_Detection/
│
├── app.py
├── mask_detection.h5
├── requirements.txt
├── README.md
├── dataset/
│   ├── with_mask/
│   └── without_mask/
│
└── training_notebook.ipynb
```

---

## Technologies Used

* Python
* Deep Learning
* CNN (Convolutional Neural Network)
* TensorFlow / Keras
* OpenCV
* NumPy
* Streamlit

---

## Model Classes

The model predicts two classes:

* `Mask`
* `No Mask`

```python
labels_dict = {
    1: "Mask",
    0: "No Mask"
}
```

---

## Features

### 1. Webcam Detection

* Click **Open Camera**
* Webcam starts automatically
* Live prediction is shown on screen

### 2. Image Upload Detection

* Upload `.jpg`, `.jpeg`, or `.png`
* Model predicts whether mask is present or not

---

## Installation

### Step 1: Clone the Project

```bash
git clone <your-repository-link>
cd Face_Mask_Detection
```

### Step 2: Create Virtual Environment

```bash
python -m venv venv
```

### Step 3: Activate Virtual Environment

### Windows

```bash
venv\Scripts\activate
```

### Linux / Mac

```bash
source venv/bin/activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## requirements.txt

```txt
streamlit
numpy
opencv-python
tensorflow
h5py
pillow
```

---

## Run the Application

```bash
streamlit run app.py
```

---

## How It Works

1. Input image is captured from webcam or uploaded manually
2. Image is resized to model input size
3. Pixel values are normalized
4. Model predicts class probabilities
5. Final result is displayed as:

   * Mask
   * No Mask

---

## Future Improvements

* Face detection before mask prediction
* Multiple face detection
* Alert system for no-mask detection
* Deployment on cloud platforms
* Mobile support

---

## Author

Developed as a Deep Learning CNN project using Streamlit for real-time face mask detection.

---

## License

This project is for educational and learning purposes.
