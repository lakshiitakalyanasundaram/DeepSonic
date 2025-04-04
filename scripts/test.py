import os
import torch
import torchaudio
import glob
import torchaudio.transforms as T
from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification

# 📌 Paths
MODEL_DIR = "models/wav2vec2_finetuned"
LABELS = ["Real", "Fake"]

# 🎙 Load Processor & Model
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
model = Wav2Vec2ForSequenceClassification.from_pretrained(
    "facebook/wav2vec2-base", num_labels=2
)

# 📌 Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 🔄 Load latest trained model
if os.path.exists(MODEL_DIR):
    saved_models = glob.glob(f"{MODEL_DIR}/batch_*.pth")
    if saved_models:
        latest_model = max(saved_models, key=os.path.getctime)  # Get the most recent file
        print(f"🔄 Loading model from {latest_model}...")
        model.load_state_dict(torch.load(latest_model, map_location=device))
        model.eval()
    else:
        print("⚠️ No trained model found! Exiting...")
        exit()
else:
    print("⚠️ Model directory does not exist! Exiting...")
    exit()

def preprocess_audio(file_path):
    """Load, resample, and preprocess audio files."""
    waveform, sample_rate = torchaudio.load(file_path)
    
    # Resample to 16,000 Hz if necessary
    if sample_rate != 16000:
        resampler = T.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)
        sample_rate = 16000  # Update sample rate

    # Ensure fixed length (1 second)
    target_length = 16000  # 1 second of audio at 16kHz
    if waveform.shape[1] < target_length:
        waveform = torch.cat([waveform, torch.zeros(1, target_length - waveform.shape[1])], dim=1)
    else:
        waveform = waveform[:, :target_length]

    input_values = processor(
        waveform.squeeze(0).numpy(),  # Convert tensor to numpy
        sampling_rate=sample_rate,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=target_length
    ).input_values

    return input_values.to(device)

# 🔹 Get user input for file path
file_path = input("Enter the path to the audio file: ").strip()

if not os.path.exists(file_path):
    print("⚠️ File does not exist! Please check the path.")
    exit()

# 🔹 Process and classify the audio file
input_values = preprocess_audio(file_path)

with torch.no_grad():
    logits = model(input_values).logits
    predicted_label = torch.argmax(logits, dim=-1).item()

print(f"🎙️ Prediction for {os.path.basename(file_path)}: {LABELS[predicted_label]}")
