import os
import numpy as np
import librosa
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from joblib import dump

# Path to your dataset (change r" if needed)
DATASET_PATH =r"C:\Users\DELL\Downloads\archive (1)\audio_speech_actors_01-24\Actor_21"

# Emotion code mapping based on RAVDESS naming
emotion_map = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

def extract_features(file_path):
    try:
        audio, sr = librosa.load(file_path, sr=22050)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        return np.mean(mfccs.T, axis=0)
    except Exception as e:
        print(f"Error extracting from {file_path}: {e}")
        return None

X, y = [], []

# Loop through dataset and extract features
for root, _, files in os.walk(DATASET_PATH):
    for file in files:
        if file.endswith(".wav"):
            parts = file.split("-")
            if len(parts) >= 3:
                emotion_code = parts[2]
                label = emotion_map.get(emotion_code)
                if label:
                    path = os.path.join(root, file)
                    features = extract_features(path)
                    if features is not None:
                        X.append(features)
                        y.append(label)

print(f"✅ Found {len(X)} valid audio files with features.")

# 🔍 Step 2: Check before training
if len(X) == 0 or len(y) == 0:
    print("❌ No valid audio data found. Please check dataset path and contents.")
    exit()

# Convert to NumPy arrays
X = np.array(X)
y = np.array(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = SVC(kernel='linear', probability=True)
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
print(f"🎯 Model trained with accuracy: {accuracy:.2f}")

# Save model
dump(model, "emotion_model.joblib")
print("✅ Model saved as emotion_model.joblib")
