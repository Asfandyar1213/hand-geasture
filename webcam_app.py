import streamlit as st
import cv2
import tensorflow as tf
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from collections import deque

# Load the trained model
model = tf.keras.models.load_model("asl_model.h5")

# Label mapping (A-Y excluding J and Z)
labels = [chr(i + 65) for i in range(26) if i != 9 and i != 25]

# Preprocessing function
def preprocess(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (28, 28))
    normalized = resized / 255.0
    reshaped = normalized.reshape(1, 28, 28, 1)
    return reshaped

class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.frame_queue = deque(maxlen=10)
    
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Preprocess the frame and predict the gesture
        preprocessed_frame = preprocess(img)
        prediction = model.predict(preprocessed_frame)
        predicted_class = np.argmax(prediction)
        label = labels[predicted_class]
        
        # Display the label on the frame
        cv2.putText(img, f"Prediction: {label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Display the processed frame with prediction
        return img

# Streamlit App Layout
st.title("Real-time ASL Hand Gesture Recognition")
st.write("Use your webcam to show hand gestures (A-Y excluding J and Z), and the model will predict the letter.")

# Start webcam stream with Streamlit WebRTC
webrtc_streamer(key="example", video_transformer_factory=VideoTransformer, media_stream_constraints={"video": True, "audio": False})
