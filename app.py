import streamlit as st
import torchaudio
import torchaudio.transforms as T
import numpy as np
import soundfile as sf
import tensorflow as tf

# Load the pre-trained model (ensure the model path is correct)
loaded_model = tf.keras.models.load_model('voice_detecting_model.h5')

def extract_mfcc(filename, n_mfcc=13):
    waveform, sample_rate = torchaudio.load(filename)
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

# Allow the user to choose between uploading or recording audio
# audio_input = st.radio("Select audio input type:", ("Upload", ""))

audio_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])
if audio_file is not None:
    # Process the uploaded audio file
    st.audio(audio_file, format='audio/wav')
    mfcc_features = extract_mfcc(audio_file)

# Model prediction and display results
if 'mfcc_features' in locals():
    predictions = loaded_model.predict(np.array([mfcc_features]))
    labels = ["Bonafide", "AI Spoofing", "Replay Spoofing"]
    predicted_label = labels[np.argmax(predictions[0])]
    st.success(f"Prediction: {predicted_label}")
