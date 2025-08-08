import streamlit as st
import sounddevice as sd
from scipy.io.wavfile import write
import librosa
import numpy as np
from joblib import load
import uuid
import os

st.set_page_config(page_title="Speech Emotion Recognition", layout="centered")
st.title("🎙️ Real-Time Speech Emotion Recognition")
st.write("Click the button below and speak for 3 seconds...")

DURATION = 3  # seconds
SAMPLE_RATE = 22050  # Hz

# Load trained model
model = load("emotion_model.joblib")

# Feature extraction function
def extract_features(file_path):
    audio, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    mfccs_mean = np.mean(mfccs.T, axis=0)
    return mfccs_mean.reshape(1, -1)

if st.button("🎤 Record and Predict"):
    st.write("Recording...")
    recording = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1)
    sd.wait()

    # Save temporary audio file
    file_name = f"temp_{uuid.uuid4().hex}.wav"
    write(file_name, SAMPLE_RATE, recording)

    st.success("Recording complete!")

    # Extract features and predict
    features = extract_features(file_name)
    prediction = model.predict(features)[0]

    # Show result
    st.subheader(f"🧠 Detected Emotion: `{prediction.upper()}`")

    # Clean up
    os.remove(file_name)
