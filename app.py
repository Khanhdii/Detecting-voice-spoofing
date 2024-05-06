import streamlit as st
import torchaudio
import torchaudio.transforms as T
import numpy as np
import tensorflow as tf
from tempfile import NamedTemporaryFile

# Attempt to import sounddevice and handle the ImportError if it occurs
try:
    import sounddevice as sd
    SD_AVAILABLE = True
except ImportError:
    SD_AVAILABLE = False
    st.error("SoundDevice is not available on this server.")

# Import soundfile conditionally since it might be used only for recording
if SD_AVAILABLE:
    import soundfile as sf

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
    return mfcc[:, :500]

# Streamlit app interface
st.title("Voice Recognition App")

# Allow the user to choose between uploading or recording audio
audio_input = st.radio("Select audio input type:", ("Upload", "Record"))

if audio_input == "Upload":
    audio_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])
    if audio_file is not None:
        # Process the uploaded audio file
        st.audio(audio_file, format='audio/wav')
        with NamedTemporaryFile(delete=True, suffix='.wav') as temp_file:
            temp_file.write(audio_file.getvalue())
            mfcc_features = extract_mfcc(temp_file.name)

elif audio_input == "Record" and SD_AVAILABLE:
    record_button = st.button("Record Audio")
    if record_button:
        # Record audio from microphone
        st.info("Recording... Say something!")
        audio_data = sd.rec(int(5 * 16000), samplerate=16000, channels=1, dtype='float32')
        sd.wait()
        st.info("Recording complete!")

        gain_factor = 100.0  # Adjust this value to increase the volume
        audio_data = audio_data * gain_factor  # Apply gain to increase volume
        audio_data = np.clip(audio_data, -1.0, 1.0)

        # Save audio to a file
        with NamedTemporaryFile(delete=True, suffix='.wav') as temp_file:
            sf.write(temp_file.name, audio_data, 16000)
            # Display the recorded audio
            st.audio(temp_file.name, format='audio/wav')
            mfcc_features = extract_mfcc(temp_file.name)

# Model prediction and display results
if 'mfcc_features' in locals():
    predictions = loaded_model.predict(np.array([mfcc_features]))
    labels = ["Bonafide", "AI Spoofing", "Replay Spoofing"]
    predicted_label = labels[np.argmax(predictions[0])]
    st.success(f"Prediction: {predicted_label}")
