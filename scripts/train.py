import os
import torch
import torchaudio
import torch.nn as nn
import torch.optim as optim
from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification

# 📌 Paths
DATA_DIR = "/Users/lakshiitakalyanasundaram/Desktop/momenta/data/raw_audio"
FAKE_DIR = os.path.join(DATA_DIR, "fake")
REAL_DIR = os.path.join(DATA_DIR, "real")

# 🎙 Load Processor & Model
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
model = Wav2Vec2ForSequenceClassification.from_pretrained(
    "facebook/wav2vec2-base", num_labels=2
)

# 📌 Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 🔹 Hyperparameters
EPOCHS = 10
BATCH_SIZE = 8
LEARNING_RATE = 1e-5

# 🔹 Loss function & Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

def load_audio(file_path):
    """Load and preprocess audio files."""
    waveform, sample_rate = torchaudio.load(file_path)
    waveform = waveform.squeeze(0)  # (1, T) -> (T)
    target_length = 16000  # Ensure fixed length
    
    if waveform.shape[0] < target_length:
        waveform = torch.cat([waveform, torch.zeros(target_length - waveform.shape[0])])
    else:
        waveform = waveform[:target_length]

    input_values = processor(
        waveform.numpy(),
        sampling_rate=sample_rate,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=target_length
    ).input_values
    
    return input_values.to(device)

# 🚀 Training (Full Dataset, Balanced Batches)
batch_size = 100  # 100 real + 100 fake = 200 total

total_real_samples = len(os.listdir(REAL_DIR))
total_fake_samples = len(os.listdir(FAKE_DIR))

total_samples = min(total_real_samples, total_fake_samples)

for batch_start in range(0, total_samples, batch_size):
    X_train, y_train = [], []
    batch_end = min(batch_start + batch_size, total_samples)
    
    # Load 100 real samples
    real_files = sorted(os.listdir(REAL_DIR))[batch_start:batch_end]
    for file in real_files:
        if file.endswith(".wav"):
            X_train.append(load_audio(os.path.join(REAL_DIR, file)))
            y_train.append(0)
    
    # Load 100 fake samples
    fake_files = sorted(os.listdir(FAKE_DIR))[batch_start:batch_end]
    for file in fake_files:
        if file.endswith(".wav"):
            X_train.append(load_audio(os.path.join(FAKE_DIR, file)))
            y_train.append(1)

    if not X_train:
        print(f"⚠️ Skipping batch {batch_start}-{batch_end}, no valid data.")
        continue

    # Convert lists to tensors
    X_train = torch.cat(X_train).to(device)
    y_train = torch.tensor(y_train, dtype=torch.long).to(device)
    
    print(f"🚀 Training Batch {batch_start}-{batch_end} ({len(X_train)} samples)...")
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        
        for i in range(0, len(X_train), BATCH_SIZE):
            optimizer.zero_grad()
            outputs = model(X_train[i:i+BATCH_SIZE]).logits
            loss = criterion(outputs, y_train[i:i+BATCH_SIZE])
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {total_loss/len(X_train):.4f}")
    
    # Save model after each batch
    os.makedirs("models/wav2vec2_finetuned", exist_ok=True)
    torch.save(model.state_dict(), f"models/wav2vec2_finetuned/batch_{batch_start}-{batch_end}.pth")
    
    print(f"✅ Batch {batch_start}-{batch_end} Training Complete & Saved!")
