import os
import torch
import torchaudio
import glob
import numpy as np
from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification
import torchaudio.transforms as T
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 📌 Paths
MODEL_DIR = "models/wav2vec2_finetuned"
TEST_REAL_DIR = "/Users/lakshiitakalyanasundaram/Downloads/for-norm/validation/real"
TEST_FAKE_DIR = "/Users/lakshiitakalyanasundaram/Downloads/for-norm/validation/fake"
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
latest_model = None

if os.path.exists(MODEL_DIR):
    saved_models = glob.glob(f"{MODEL_DIR}/batch_*.pth")
    if saved_models:
        latest_model = max(saved_models, key=os.path.getctime)  # Get the most recent file

if latest_model:
    print(f"🔄 Loading model from {latest_model}...")
    model.load_state_dict(torch.load(latest_model, map_location=device))
    model.eval()
else:
    print("⚠️ No trained model found! Exiting...")
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

# 🔹 Collect test files (Only first 100 from each class)
real_files = sorted(glob.glob(os.path.join(TEST_REAL_DIR, "*.wav")))[:100]
fake_files = sorted(glob.glob(os.path.join(TEST_FAKE_DIR, "*.wav")))[:100]

# 🔹 Ground truth labels
y_true = [0] * len(real_files) + [1] * len(fake_files)  # 0 = Real, 1 = Fake

# 🔹 Make predictions
y_pred = []

print(f"🔍 Evaluating {len(real_files) + len(fake_files)} test samples...")

for file in real_files + fake_files:
    input_values = preprocess_audio(file)

    with torch.no_grad():
        logits = model(input_values).logits
        predicted_label = torch.argmax(logits, dim=-1).item()
    
    y_pred.append(predicted_label)

# 📊 Compute metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average="binary")
recall = recall_score(y_true, y_pred, average="binary")
f1 = f1_score(y_true, y_pred, average="binary")

# 📢 Print results
print("\n🔎 **Model Evaluation Results:**")
print(f"✅ Accuracy  : {accuracy:.4f}")
print(f"✅ Precision : {precision:.4f}")
print(f"✅ Recall    : {recall:.4f}")
print(f"✅ F1 Score  : {f1:.4f}")
