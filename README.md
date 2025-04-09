
# Audio Deepfake Detection

## Project Overview  
This project is designed to detect deepfake audio using Wav2Vec2, a state-of-the-art self-supervised model developed by Facebook AI for speech representation learning. With the rise of AI-generated content, deepfake audio presents serious threats in domains like cybersecurity, media, legal evidence, and personal privacy.
This system leverages the powerful Wav2Vec2 architecture fine-tuned on a labeled dataset consisting of both real human voices and synthetically generated audio samples. The goal is to automatically classify an audio clip as either real or deepfake with high accuracy.

## Features  
- **Deepfake Detection**: Identifies whether an audio sample is real or fake.  
- **Pre-trained Model Support**: Uses state-of-the-art deep learning models.  
- **User-Friendly Interface**: Simple script-based execution for ease of use.  
- **Scalable & Efficient**: Can be integrated into real-time applications.  

## 📂 Project Structure  
```
├── app/                 # Main application files
│   └── app.py           # Script to run inference
├── models/              # Pre-trained models
├── scripts/             # Utility scripts
├── data/                # Contains raw audio files (ignored in .gitignore)
├── upload/              # For storing temporary files (ignored in .gitignore)
├── requirements.txt     # Dependencies
├── training.log         # Training history
├── .gitignore           # Ignoring unnecessary files
└── README.md            # Project documentation (this file)
```

## Setup & Installation  
### 1️. Clone the Repository  
```
git clone https://github.com/lakshiitakalyanasundaram/Lakshiita_kalyanasundaram.git  
cd Lakshiita_kalyanasundaram
```
### 2️. Install Dependencies
```
pip install -r requirements.txt
 ```

### 3️. Run the Application
```
python app/app.py --input path/to/audio.wav
```

## **How It Works**
- **Preprocessing**: The audio file is preprocessed (e.g., noise reduction, feature extraction).

- **Model Inference**: The trained model classifies the audio as real or fake.

- **Output**: The result is displayed as real (✅) or deepfake (❌).

## **Accuracy & Performance**
- **Dataset**: Trained on a dataset of real and fake audio samples.

- **Model Used**: LSTM + CNN Hybrid.

- **Accuracy**: Achieved ~72% accuracy in testing.

## Use Cases

- **Media Verification**: Detect and verify the authenticity of voice recordings in journalism and broadcasting.

- **Cybersecurity**: Prevent voice spoofing attacks in authentication systems, especially in financial and biometric applications.

- **Forensic Analysis**: Assist law enforcement and legal investigations by identifying AI-generated speech in evidence materials.

- **AI Ethics & Policy Compliance**: Ensure responsible use of generative AI by identifying deepfake content, helping organizations comply with AI ethics policies.

- **AI-Generated Speech Detection**: Effectively identify synthetically generated voices from real human speech using deep learning.
---

## Dataset  

I have used the **3004lakshu/for-norm** dataset available on Hugging Face. It contains labeled samples of real and deepfake audio.

### How to Access the Dataset  
1. Install the Hugging Face datasets library:
   ```bash
   pip install datasets
   ```

2. Load the dataset in your script:
   ```python
   from datasets import load_dataset
   dataset = load_dataset("3004lakshu/Deepfake-Audio")
   ```

3. Explore the dataset:
   ```python
   print(dataset)
   ```

Link: [https://huggingface.co/datasets/3004lakshu/Deepfake-Audio](https://huggingface.co/datasets/3004lakshu/Deepfake-Audio)

---

## Model Details  

The classification model is built on top of the **Wav2Vec2** encoder. It extracts high-level audio embeddings from the input waveform. These embeddings are passed through a combination of **LSTM and CNN** layers, which capture temporal and local patterns in the speech signal.

This hybrid model architecture improves classification accuracy by combining both sequence and feature learning capabilities.

---

## Training Process  

1. **Preprocessing**:  
   - Audio normalization  
   - Silence trimming  
   - Conversion to 16kHz mono WAV

2. **Training**:  
   - Loss Function: Binary Cross Entropy  
   - Optimizer: Adam  
   - Epochs: 20  
   - Dataset Split: 80% train, 20% test  
   - Validation Accuracy: ~72%

---

## Fine-Tuned Model

This repository contains a **fine-tuned `.pth` model** using the `wav2vec2` architecture for audio deepfake detection.  
The model has been trained and optimized on a custom dataset for the task.

---

### How to Use

1. **Install the Hugging Face Hub library** (if you haven't already):

   ```bash
   pip install huggingface_hub
   huggingface-cli login

   ```
2. **Download the model:**
   ```
   from huggingface_hub import hf_hub_download
   model_path = hf_hub_download(
    repo_id="3004lakshu/wav2vec2_trained",
    filename="deepfake_model.pth"
   )
   ```
## Model Hub URL:
https://huggingface.co/3004lakshu/wav2vec2_trained

---

## Task Description  

To understand the goals, objectives, and evaluation criteria of this internship task, please refer to the detailed documentation below:

**Task Document (Google Docs):**  
[View Full Internship Task Document](https://docs.google.com/document/d/1A-vxewmCHlzjVEGx68Wg65CItiBBgoLNYx_Xd-3UcZk/edit?usp=sharing)

---

## Demo  

A screen recording of the Streamlit interface demonstrating the detection process has been included.

**Watch the demo here:** 

[Click here for the demo](https://drive.google.com/file/d/1Wl4rqkgmrAE0OWKcelehZdtTSFzeTu_8/view?usp=sharing)

---
## Contributing
Feel free to fork the repo, open an issue, or submit a pull request to improve the project!

## License
MIT License. Use it freely but give credits where due.

If this project helps you, give it a star on GitHub!
