import streamlit as st
import torchaudio
import torchaudio.transforms as T
import numpy as np
import tensorflow as tf
from io import BytesIO

# Load the pre-trained model (ensure the model path is correct)
loaded_model = tf.keras.models.load_model('voice_detecting_model.h5')

def extract_mfcc(file_content, n_mfcc=13):
    waveform, sample_rate = torchaudio.load(file_content)
    mfcc_transform = T.MFCC(
        sample_rate=sample_rate,
        n_mfcc=n_mfcc,
        melkwargs={'n_fft': 400, 'hop_length': 160, 'n_mels': 23}
    )
    mfcc = mfcc_transform(waveform).squeeze(0).detach().numpy()
    
    if mfcc.shape[1] < 500:
        mfcc = np.pad(mfcc, ((0, 0), (0, 500 - mfcc.shape[1])), mode='constant', constant_values=0)
    return mfcc[:,:500]

# Streamlit app interface
st.title("Voice Recognition App")

# File uploader for audio files
audio_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])
if audio_file is not None:
    # Display the audio file player
    st.audio(audio_file, format='audio/wav')

    # Convert the uploaded file to a BytesIO object for processing
    file_bytes = BytesIO(audio_file.getvalue())

    # Extract features
    mfcc_features = extract_mfcc(file_bytes)

    # Model prediction and display results
    predictions = loaded_model.predict(np.array([mfcc_features]))
    labels = ["Bonafide", "AI Spoofing", "Replay Spoofing"]
    predicted_label = labels[np.argmax(predictions[0])]
    st.success(f"Prediction: {predicted_label}")
