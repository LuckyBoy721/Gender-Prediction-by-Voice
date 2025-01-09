import os
import librosa
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
from imblearn.over_sampling import SMOTE  # Import SMOTE dari imbalanced-learn

# Fungsi untuk ekstraksi fitur
def extract_features(audio_path):
    try:
        print(f"Processing file: {audio_path}")  # Tambahkan print untuk file yang sedang diproses
        # Load audio
        audio, sr = librosa.load(audio_path, sr=16000)
        
        # Extract features
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        pitch, _ = librosa.piptrack(y=audio, sr=sr)
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
        
        # Combine features
        features = np.concatenate((
            np.mean(mfcc, axis=1), 
            np.std(mfcc, axis=1),
            [np.mean(pitch), np.std(pitch)],
            [np.mean(spectral_centroid), np.std(spectral_centroid)]
        ))
        
        return features
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None

# Path dataset
dataset_path = r"C:\Users\Asyauri\Documents\Semester 3\App Prototype\deploy\datasets"
categories = ["Male", "Female"]

# Ekstraksi fitur dari dataset
print("Starting feature extraction...")
data = []
labels = []

for category in categories:
    print(f"Processing category: {category}")  # Tambahkan print untuk kategori yang sedang diproses
    folder_path = os.path.join(dataset_path, category)
    for filename in os.listdir(folder_path):
        if filename.endswith(".wav"):
            file_path = os.path.join(folder_path, filename)
            features = extract_features(file_path)
            if features is not None:
                data.append(features)
                labels.append(category)

print("Feature extraction complete. Total samples:", len(data))

# Konversi label menjadi numerik
print("Encoding labels...")
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

# Simpan Label Encoder
joblib.dump(label_encoder, "label_encoder.pkl")
print("Label encoder saved as 'label_encoder.pkl'.")

# Konversi ke DataFrame
print("Converting data to DataFrame...")
df = pd.DataFrame(data)
df["label"] = labels_encoded

# Split dataset
print("Splitting dataset into training and testing sets...")
X = df.iloc[:, :-1].values
y = df["label"].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

# Aplikasikan SMOTE untuk menyeimbangkan data training
print("Applying SMOTE for balancing the dataset...")
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

print(f"After SMOTE: Training samples: {len(X_train_balanced)}, Testing samples: {len(X_test)}")

# Scaling fitur
print("Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_balanced)
X_test_scaled = scaler.transform(X_test)

# Simpan scaler
joblib.dump(scaler, "scaler.pkl")
print("Scaler saved as 'scaler.pkl'.")

# Simpan dataset yang sudah diseimbangkan
print("Saving balanced dataset to 'gender_dataset_balanced.npz'...")
np.savez("gender_dataset_balanced.npz", X_train=X_train_scaled, X_test=X_test_scaled, y_train=y_train_balanced, y_test=y_test)
print("Dataset preparation complete!")
