import os
import numpy as np
import librosa
import torchaudio
from tqdm import tqdm

# 📌 Set Paths
DATASET_PATH = "/Users/lakshiitakalyanasundaram/Downloads/for-norm/training"  
PROCESSED_PATH = "/Users/lakshiitakalyanasundaram/Desktop/INTERNSHIP/momenta/Lakshiita_kalyanasundaram/processed_data"

os.makedirs(PROCESSED_PATH, exist_ok=True)

# 🔹 Recursively Find All .wav Files
all_files = []
for root, _, files in os.walk(DATASET_PATH):
    for file in files:
        if file.endswith(".wav"):
            all_files.append(os.path.join(root, file))

print(f"Found {len(all_files)} audio files.")  # ✅ Check if files are detected

# 🔹 Function to Extract Features
def extract_features(audio_path, sr=16000, n_mfcc=40):
    waveform, sample_rate = torchaudio.load(audio_path)
    waveform = waveform.mean(dim=0).numpy()  # Convert to mono

    # ✅ Extract Features
    mfcc = librosa.feature.mfcc(y=waveform, sr=sr, n_mfcc=n_mfcc)
    mel_spec = librosa.feature.melspectrogram(y=waveform, sr=sr)
    chroma = librosa.feature.chroma_stft(y=waveform, sr=sr)

    # ✅ Convert to Log Scale
    mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

    return mfcc, mel_spec, chroma

# 🔹 Process Files
for file in tqdm(all_files, desc="Processing Audio Files"):
    try:
        mfcc, mel_spec, chroma = extract_features(file)

        base_filename = os.path.basename(file).replace(".wav", "")
        np.save(os.path.join(PROCESSED_PATH, f"{base_filename}_mfcc.npy"), mfcc)
        np.save(os.path.join(PROCESSED_PATH, f"{base_filename}_mel.npy"), mel_spec)
        np.save(os.path.join(PROCESSED_PATH, f"{base_filename}_chroma.npy"), chroma)
    except Exception as e:
        print(f"⚠️ Error processing {file}: {e}")

print("✅ Preprocessing Complete! Features saved in:", PROCESSED_PATH)
