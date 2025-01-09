import streamlit as st
import os
import numpy as np
import librosa
import librosa.display
import joblib
import json
import matplotlib.pyplot as plt
import base64


# Fungsi untuk ekstraksi fitur
def extract_features(audio_path):
    try:
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
        st.error(f"Error processing audio file: {e}")
        return None

# Fungsi untuk visualisasi audio
def visualize_audio(audio_path):
    try:
        audio, sr = librosa.load(audio_path, sr=16000)
        
        # Plot Waveform
        st.subheader("Waveform")
        fig, ax = plt.subplots()
        librosa.display.waveshow(audio, sr=sr, ax=ax)
        ax.set(title="Waveform of the Uploaded Audio")
        st.pyplot(fig)

        # Plot Mel-Spectrogram
        st.subheader("Mel-Spectrogram")
        mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
        mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)

        fig, ax = plt.subplots()
        img = librosa.display.specshow(mel_spectrogram_db, sr=sr, x_axis='time', y_axis='mel', ax=ax, cmap='coolwarm')
        ax.set(title="Mel-Spectrogram of the Uploaded Audio")
        fig.colorbar(img, ax=ax, format="%+2.0f dB")
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Error visualizing audio file: {e}")

# Load model dan preprocessing tools
@st.cache_resource
def load_model_and_tools():
    models = {
        "Random Forest": joblib.load("random_forest_model.pkl"),
        "SVM": joblib.load("svm_model.pkl"),
        "KNN": joblib.load("knn_model.pkl")
    }
    scaler = joblib.load("scaler.pkl")
    encoder = joblib.load("label_encoder.pkl")
    return models, scaler, encoder

# Fungsi untuk memuat skor akurasi
@st.cache_data
def load_model_accuracies():
    try:
        with open("model_accuracies.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        st.warning("Model accuracy file not found. Ensure 'model_accuracies.json' exists.")
        return {}

# Fungsi untuk menampilkan horizontal bar chart akurasi model
def display_accuracy_chart(accuracies):
    st.subheader("Model Accuracy Chart")
    fig, ax = plt.subplots()
    model_names = list(accuracies.keys())
    scores = list(accuracies.values())
    
    ax.barh(model_names, scores, color='skyblue')
    ax.set_xlabel('Accuracy')
    ax.set_title('Model Accuracy Comparison')
    st.pyplot(fig)

# Fungsi untuk prediksi gender
def predict_gender(audio_path, models, scaler, encoder):
    feature = extract_features(audio_path)
    if feature is not None:
        feature_scaled = scaler.transform([feature])
        predictions = {}
        for model_name, model in models.items():
            prediction = model.predict(feature_scaled)
            probabilities = model.predict_proba(feature_scaled)
            predictions[model_name] = {
                "gender": encoder.inverse_transform(prediction)[0],
                "probabilities": probabilities[0]
            }
        return predictions
    else:
        return None

# Streamlit interface
def set_background(image_path):
    with open(image_path, "rb") as image_file:
        image_data = image_file.read()
    b64_image = base64.b64encode(image_data).decode()
    bg_style = f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{b64_image}");
            background-size: cover;
            background-position: center;
        }}
        </style>
    """
    st.markdown(bg_style, unsafe_allow_html=True)

background_home = "a.png"


def main():
    st.set_page_config(page_title="Gender Prediction", layout="wide", initial_sidebar_state="collapsed")  
    # Sidebar navigation
    
    if 'page' not in st.session_state:
        st.session_state.page = "Main Menu"

    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Main Menu", "Gender Prediction", "Model Accuracies"])

    if page == "Main Menu":
        st.session_state.page = "Main Menu"
        set_background(background_home)
        st.title("Welcome to the Gender Prediction by Voice App")
        st.write("""
            This application allows you to predict gender based on voice audio files. It uses machine learning models such as Random Forest, SVM, and KNN.
            You can upload a `.wav` file and visualize its audio features like waveform and Mel-spectrogram. Additionally, the app provides model performance comparisons in terms of accuracy.
            Use the 'Gender Prediction' section to predict the gender from an audio sample, and explore model accuracies in the 'Model Accuracies' section.
        """)
        st.write("Click the navigation button to go to the Gender Prediction page:")
        

    elif page == "Gender Prediction":
        st.session_state.page = "Gender Prediction"
        st.title("Gender Prediction from Voice")
        st.write("Upload a .wav file to predict the gender and visualize its features.")
        
        # Upload audio file
        uploaded_file = st.file_uploader("Choose a .wav file", type=["wav"])
        
        if uploaded_file is not None:
            # Save uploaded file locally
            with open("uploaded_audio.wav", "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            st.audio("uploaded_audio.wav", format="audio/wav")
            
            # Visualize audio
            visualize_audio("uploaded_audio.wav")
            
            # Load models and tools
            models, scaler, encoder = load_model_and_tools()
            
            # Predict gender
            predictions = predict_gender("uploaded_audio.wav", models, scaler, encoder)
            
            if predictions:
                # Display predictions for each model
                st.subheader("Predictions")
                for model_name, result in predictions.items():
                    st.write(f"**{model_name}:** {result['gender']}")
                    
                    # Display probabilities
                    st.write(f"Prediction Probabilities ({model_name})")
                    prob_df = {gender: prob for gender, prob in zip(encoder.classes_, result['probabilities'])}
                    st.bar_chart(prob_df)
            
            else:
                st.error("Failed to process the audio file. Please try again.")

    elif page == "Model Accuracies":
        st.session_state.page = "Model Accuracies"
        st.title("Model Accuracies")
        
        # Load and display model accuracies
        model_accuracies = load_model_accuracies()
        if model_accuracies:
            st.write("Below is a comparison of the accuracy scores for each model used in this application. These scores reflect how well each model performs in predicting gender from voice features.")
            display_accuracy_chart(model_accuracies)
        else:
            st.error("No accuracy data available.")
            st.write("Please ensure the 'model_accuracies.json' file is available in the project directory.")

if __name__ == "__main__":
    main()
