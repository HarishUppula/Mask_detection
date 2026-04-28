import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Face Mask Detection", page_icon="😷", layout="centered")

st.title(" Face Mask Detection using Webcam")
st.write("Click the button below to open your camera and detect whether you are wearing a mask or not.")

# Load trained model
model = load_model("Mask_detector.h5")

# Labels (change order if your training labels are different)
labels_dict = {
    1: "Mask",
    0: "No Mask"
}

IMG_SIZE = 128
# Upload image option
uploaded_file = st.file_uploader("Upload an image for mask detection", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    resized_img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    normalized_img = resized_img / 255.0
    reshaped_img = np.reshape(normalized_img, [1, IMG_SIZE, IMG_SIZE, 3])

    prediction = model.predict(reshaped_img, verbose=0)
    predicted_label = np.argmax(prediction)
    upload_result = labels_dict[predicted_label]

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    st.image(img_rgb, caption=f"Prediction: {upload_result}", use_container_width=True)
    st.success(f"Uploaded Image Result: {upload_result}")


run = st.button("Open Camera")
FRAME_WINDOW = st.image([])
status_text = st.empty()

if run:
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("Unable to access webcam")
    else:
        stop_button = st.button("Stop Camera")

        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                st.warning("Failed to capture frame")
                break

            # Resize for model prediction
            resized = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
            normalized = resized / 255.0
            reshaped = np.reshape(normalized, [1, IMG_SIZE, IMG_SIZE, 3])

            prediction = model.predict(reshaped, verbose=0)
            predicted_label = np.argmax(prediction)
            result = labels_dict[predicted_label]

            # Display prediction on frame
            cv2.putText(
                frame,
                f"Prediction: {result}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            FRAME_WINDOW.image(frame)
            status_text.success(f"Detected: {result}")

            if stop_button:
                break

        cap.release()
        cv2.destroyAllWindows()



