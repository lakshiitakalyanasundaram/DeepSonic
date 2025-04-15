import os
import torch
import torchaudio
import numpy as np
from flask import Flask, request, jsonify, send_from_directory, render_template, url_for
from werkzeug.utils import secure_filename
from huggingface_hub import hf_hub_download
from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification
import torchaudio.transforms as T
import glob

app = Flask(__name__, static_folder='static', template_folder='frontend')

# 🔧 Constants
REPO_ID = "3004lakshu/wav2vec2_trained"
MODEL_FILENAME = "last_trained.pth"
LABELS = ["Real", "Fake"]
SAMPLING_RATE = 16000  # For Wav2Vec2 model
MAX_PREPROCESS_DURATION_S = 5
TARGET_LENGTH = SAMPLING_RATE * MAX_PREPROCESS_DURATION_S  # Target length in samples

# Configure upload folder
UPLOAD_FOLDER = 'upload'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Use the relative path for sample audio directory
SAMPLE_AUDIO_DIR = UPLOAD_FOLDER

# Load model and processor
def load_model_and_processor():
    try:
        processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
        model = Wav2Vec2ForSequenceClassification.from_pretrained(
            "facebook/wav2vec2-base", num_labels=len(LABELS)
        )
        downloaded_model_path = hf_hub_download(repo_id=REPO_ID, filename=MODEL_FILENAME)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.load_state_dict(torch.load(downloaded_model_path, map_location=device))
        model.to(device)
        model.eval()
        return processor, model, device
    except Exception as e:
        print(f"Failed to load model: {e}")
        return None, None, None

processor, model, device = load_model_and_processor()

def preprocess_audio(waveform, sample_rate):
    """Resamples, pads/truncates audio to TARGET_LENGTH, and processes for Wav2Vec2."""
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)

    if sample_rate != SAMPLING_RATE:
        resampler = T.Resample(orig_freq=sample_rate, new_freq=SAMPLING_RATE)
        waveform = resampler(waveform)
        sample_rate = SAMPLING_RATE

    current_length = waveform.shape[1]
    if current_length < TARGET_LENGTH:
        padding = torch.zeros(waveform.shape[0], TARGET_LENGTH - current_length)
        waveform = torch.cat([waveform, padding], dim=1)
    elif current_length > TARGET_LENGTH:
        waveform = waveform[:, :TARGET_LENGTH]

    processed_waveform = waveform.squeeze().numpy() if waveform.shape[0] == 1 else waveform.numpy()

    input_values = processor(
        processed_waveform,
        sampling_rate=sample_rate,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=TARGET_LENGTH,
    ).input_values

    return input_values.to(device)

def predict_audio(input_values):
    """Performs inference on the preprocessed audio input."""
    with torch.no_grad():
        logits = model(input_values).logits
        predicted_label_idx = torch.argmax(logits, dim=-1).item()
        probabilities = torch.softmax(logits, dim=-1)[0]
    return predicted_label_idx, probabilities

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_audio():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not file.filename.lower().endswith('.wav'):
        return jsonify({'error': 'Only WAV files are supported'}), 400

    try:
        # Save the uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Load and process the audio
        waveform, sample_rate = torchaudio.load(filepath)
        input_values = preprocess_audio(waveform, sample_rate)
        predicted_label_idx, probabilities = predict_audio(input_values)
        
        # Clean up the uploaded file
        os.remove(filepath)

        # Prepare response
        result = {
            'prediction': LABELS[predicted_label_idx],
            'confidence': float(probabilities[predicted_label_idx]),
            'probabilities': {
                'real': float(probabilities[0]),
                'fake': float(probabilities[1])
            }
        }
        
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/samples')
def get_samples():
    try:
        print(f"Looking for WAV files in: {SAMPLE_AUDIO_DIR}")
        # Get all .wav files from the sample directory
        sample_files = glob.glob(os.path.join(SAMPLE_AUDIO_DIR, "*.wav"))
        print(f"Found {len(sample_files)} WAV files")
        
        samples = []
        
        for file_path in sample_files:
            try:
                print(f"Processing file: {file_path}")
                file_name = os.path.basename(file_path)
                # Skip files with 'copy' in the name
                if 'copy' in file_name.lower():
                    continue
                    
                file_size = os.path.getsize(file_path)
                
                # Get duration using torchaudio
                try:
                    waveform, sample_rate = torchaudio.load(file_path)
                    duration = waveform.shape[1] / sample_rate
                except Exception as e:
                    print(f"Error loading audio {file_path}: {str(e)}")
                    duration = 0
                
                # Clean up the filename for display
                display_name = file_name.replace('_16k.wav_norm.wav_mono.wav_silence.wav', '.wav')
                
                samples.append({
                    'name': display_name,
                    'path': f'/sample/{file_name}',
                    'size': file_size,
                    'duration': duration
                })
                print(f"Successfully added {display_name}")
            except Exception as e:
                print(f"Error processing file {file_path}: {str(e)}")
                continue
        
        print(f"Returning {len(samples)} samples")
        return jsonify(samples)
    except Exception as e:
        print(f"Error in get_samples: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/sample/<path:filename>')
def serve_sample(filename):
    try:
        return send_from_directory(SAMPLE_AUDIO_DIR, filename)
    except Exception as e:
        print(f"Error serving file {filename}: {str(e)}")
        return jsonify({'error': str(e)}), 404

if __name__ == '__main__':
    app.run(debug=True) 