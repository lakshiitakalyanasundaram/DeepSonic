import os
import torch
import torchaudio
import streamlit as st
import glob
import numpy as np
from huggingface_hub import hf_hub_download
from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification
import torchaudio.transforms as T
import tempfile
from scipy.io.wavfile import write  # Add this import for saving WAV files

# 🔧 Constants
REPO_ID = "3004lakshu/wav2vec2_trained"
MODEL_FILENAME = "last_trained.pth"
LABELS = ["Real", "Fake"]
# Use a relative path for sample audio - create this folder in the same directory as your script!
TEST_AUDIO_DIR = "/Users/lakshiitakalyanasundaram/Desktop/momenta/upload"
SAMPLING_RATE = 16000  # For Wav2Vec2 model
# Define a maximum duration for preprocessing (in seconds)
MAX_PREPROCESS_DURATION_S = 5
TARGET_LENGTH = SAMPLING_RATE * MAX_PREPROCESS_DURATION_S # Target length in samples

# --- Create sample audio directory if it doesn't exist ---
if not os.path.exists(TEST_AUDIO_DIR):
    try:
        os.makedirs(TEST_AUDIO_DIR)
        st.info(f"Created sample audio directory: '{TEST_AUDIO_DIR}'. Please add some .wav files there.")
    except OSError as e:
        st.error(f"Could not create sample audio directory '{TEST_AUDIO_DIR}': {e}")
        TEST_AUDIO_DIR = None # Disable sample loading if directory creation fails

# --- UI Setup ---
st.set_page_config(layout="wide")  # Use wider layout
st.title("🔊 DeepSonic: Audio Deepfake Detection")
st.write(f"Upload a `.wav` file, select a sample, or record up to {MAX_PREPROCESS_DURATION_S} seconds to check if it's **Real** or **Fake**.")

# Center the "📤 Upload or Select Audio" section
st.markdown("""
    <style>
        .centered-container {
            display: flex;
            justify-content: center;
            align-items: center;
            flex-direction: column;
            height: 100%;
        }
    </style>
    <div class="centered-container">
""", unsafe_allow_html=True)

# --- Model Loading Section ---
@st.cache_resource # Cache the processor and model loading
def load_model_and_processor():
    try:
        processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
        model = Wav2Vec2ForSequenceClassification.from_pretrained(
            "facebook/wav2vec2-base", num_labels=len(LABELS) # Use len(LABELS)
        )
        downloaded_model_path = hf_hub_download(repo_id=REPO_ID, filename=MODEL_FILENAME)
        # Load model state dict onto the correct device directly
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.load_state_dict(torch.load(downloaded_model_path, map_location=device))
        model.to(device)
        model.eval()
        return processor, model, device
    except Exception as e:
        st.error(f"❌ Failed to load model from Hugging Face Hub ({REPO_ID}/{MODEL_FILENAME}).")
        st.exception(e)
        st.stop() # Stop execution if model fails to load

processor, model, device = load_model_and_processor()
st.success(f"✅ Loaded model and processor to device: `{device}`")


# --- Audio Loading Section ---
audio_options = ["None"]
if TEST_AUDIO_DIR and os.path.exists(TEST_AUDIO_DIR):
    try:
        audio_files = sorted(glob.glob(os.path.join(TEST_AUDIO_DIR, "*.wav")))
        if audio_files:
             audio_options.extend([os.path.basename(file) for file in audio_files])
        else:
            st.warning(f"No `.wav` files found in the '{TEST_AUDIO_DIR}' directory.")
    except Exception as e:
        st.error(f"Error scanning audio directory '{TEST_AUDIO_DIR}': {e}")
else:
     st.warning("Sample audio directory not found or not specified.")


# 🔊 Audio Preprocessing Function (Updated)
def preprocess_audio(waveform, sample_rate):
    """
    Resamples, pads/truncates audio to TARGET_LENGTH, and processes for Wav2Vec2.
    """
    # Ensure waveform is 2D [channel, time]
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)

    # Resample to 16kHz if necessary
    if sample_rate != SAMPLING_RATE:
        resampler = T.Resample(orig_freq=sample_rate, new_freq=SAMPLING_RATE)
        waveform = resampler(waveform)
        sample_rate = SAMPLING_RATE # Update sample rate after resampling

    # Pad or truncate to TARGET_LENGTH (now potentially longer than 1 sec)
    current_length = waveform.shape[1]
    if current_length < TARGET_LENGTH:
        # Pad with zeros
        padding = torch.zeros(waveform.shape[0], TARGET_LENGTH - current_length)
        waveform = torch.cat([waveform, padding], dim=1)
    elif current_length > TARGET_LENGTH:
        # Truncate
        waveform = waveform[:, :TARGET_LENGTH]

    # Process using the Wav2Vec2 processor
    # Squeeze to 1D if it's mono before passing to processor
    processed_waveform = waveform.squeeze().numpy() if waveform.shape[0] == 1 else waveform.numpy()

    input_values = processor(
        processed_waveform,
        sampling_rate=sample_rate,
        return_tensors="pt",
        padding="max_length", # Pad/truncate to max_length defined here
        truncation=True,
        max_length=TARGET_LENGTH, # Use the target length defined earlier
    ).input_values

    return input_values.to(device) # Move tensor to the correct device

# 🔍 Prediction Logic (remains the same)
def predict_audio(input_values):
    """
    Performs inference on the preprocessed audio input.
    """
    with torch.no_grad():
        logits = model(input_values).logits
        predicted_label_idx = torch.argmax(logits, dim=-1).item()
    return predicted_label_idx

# --- UI Sections in Columns ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("📤 Upload or Select Audio")
    uploaded_file = st.file_uploader("📂 Upload a `.wav` file", type=["wav"])
    selected_audio = st.selectbox("🎼 Or choose from sample audios:", audio_options)

    if st.button("📊 Analyze File"):
        file_path = None
        temp_file_to_delete = None

        if uploaded_file:
            # Use tempfile for uploaded files
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    file_path = tmp_file.name
                    temp_file_to_delete = file_path # Mark for deletion later
                st.info(f"Processing uploaded file: {uploaded_file.name}")
            except Exception as e:
                st.error(f"Error handling uploaded file: {e}")
                st.stop()

        elif selected_audio != "None" and TEST_AUDIO_DIR:
            file_path = os.path.join(TEST_AUDIO_DIR, selected_audio)
            st.info(f"Processing selected sample: {selected_audio}")
        else:
            st.warning("⚠️ Please upload or select a file first.")
            st.stop()

        if file_path:
            st.audio(file_path, format="audio/wav")
            try:
                with st.spinner("Analyzing..."):
                    # Load and preprocess audio
                    waveform, sample_rate = torchaudio.load(file_path)
                    input_values = preprocess_audio(waveform, sample_rate) # Use the updated function

                    # Get prediction
                    predicted_label_idx = predict_audio(input_values)
                    predicted_label = LABELS[predicted_label_idx]

                    # Display result
                    if predicted_label == "Real":
                         st.success(f"✅ **Prediction:** `{predicted_label}`")
                    else:
                         st.error(f"🚨 **Prediction:** `{predicted_label}`") # Use error for fake

            except Exception as e:
                st.error(f"❌ Error analyzing audio file: {e}")
            finally:
                 # Clean up temporary file if it exists
                if temp_file_to_delete and os.path.exists(temp_file_to_delete):
                    try:
                        os.remove(temp_file_to_delete)
                        # st.write(f"Cleaned up temp file: {temp_file_to_delete}") # Optional debug message
                    except Exception as e:
                        st.warning(f"Could not delete temporary file {temp_file_to_delete}: {e}")

# Close the centered-container div
st.markdown("</div>", unsafe_allow_html=True)

# --- Footer or additional info ---
st.markdown("---")
st.caption(f"Model: {REPO_ID} | Max Analysis Duration: {MAX_PREPROCESS_DURATION_S}s | Sample Rate: {SAMPLING_RATE/1000:.0f}kHz")
