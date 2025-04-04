import os
import torch
import torchaudio
import streamlit as st
import glob
from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification
import torchaudio.transforms as T

# 📌 Paths
MODEL_DIR = "models/wav2vec2_finetuned"
LABELS = ["Real", "Fake"]
TEST_AUDIO_DIR = "/Users/lakshiitakalyanasundaram/Desktop/momenta/upload"  # 📂 Replace with your actual folder path

# 🎙 Load Processor & Model
st.title("🔊 Audio Deepfake Detection")
st.write("Upload a `.wav` file or select a sample to check if it's **Real** or **Fake**.")

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
model = Wav2Vec2ForSequenceClassification.from_pretrained(
    "facebook/wav2vec2-base", num_labels=2
)

# 📌 Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 🔄 Load latest trained model
latest_model = None

if os.path.exists(MODEL_DIR):
    saved_models = glob.glob(f"{MODEL_DIR}/batch_*.pth")
    if saved_models:
        latest_model = max(saved_models, key=os.path.getctime)

if latest_model:
    st.success(f"🔄 Loaded model from `{latest_model}`")
    model.load_state_dict(torch.load(latest_model))
    model.eval()
else:
    st.error("⚠️ No trained model found! Exiting...")
    st.stop()

# 📌 Load audio files from dropdown
audio_files = sorted(glob.glob(TEST_AUDIO_DIR + "/*.wav"))
audio_options = [os.path.basename(file) for file in audio_files]

# 📌 File Upload or Select
uploaded_file = st.file_uploader("📂 Upload a `.wav` file", type=["wav"])
selected_audio = st.selectbox("🎵 Or select a sample audio:", ["None"] + audio_options)

def preprocess_audio(file_path):
    """Load, resample, and preprocess audio files."""
    waveform, sample_rate = torchaudio.load(file_path)
    
    # Resample to 16,000 Hz if necessary
    if sample_rate != 16000:
        resampler = T.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)
        sample_rate = 16000  

    # Ensure fixed length (1 sec)
    target_length = 16000  
    if waveform.shape[1] < target_length:
        waveform = torch.cat([waveform, torch.zeros(1, target_length - waveform.shape[1])], dim=1)
    else:
        waveform = waveform[:, :target_length]

    input_values = processor(
        waveform.squeeze(0).numpy(),  
        sampling_rate=sample_rate,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=target_length
    ).input_values

    return input_values.to(device)

# 🔹 Run Prediction
if st.button("🔍 Analyze Audio"):
    file_path = None

    if uploaded_file:
        file_path = "temp_uploaded.wav"
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())
    elif selected_audio != "None":
        file_path = os.path.join(TEST_AUDIO_DIR, selected_audio)
    else:
        st.warning("⚠️ Please upload a `.wav` file or select one from the dropdown.")
        st.stop()

    st.audio(file_path, format="audio/wav")
    input_values = preprocess_audio(file_path)

    with torch.no_grad():
        logits = model(input_values).logits
        predicted_label = torch.argmax(logits, dim=-1).item()

    st.success(f"🎙️ **Prediction:** `{LABELS[predicted_label]}`")
