# scripts/predict.py
import os
import torch
import librosa
import numpy as np
from scripts.train import TransformerClassifier

# Get the absolute path to the data directory
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(root_dir, "data")

# Extract labels from wav filenames, as in train.py
wav_files = [f for f in os.listdir(data_dir) if f.endswith('.wav')]
labels = sorted(list(set([f.split('-')[0] for f in wav_files])))

def predict_audio(file_path):
    y, sr = librosa.load(file_path, sr=None)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)
    log_mel = librosa.power_to_db(mel_spec)
    # Pad or truncate to fixed length (same as in preprocess.py)
    max_len = 432
    if log_mel.shape[1] < max_len:
        pad_width = max_len - log_mel.shape[1]
        log_mel = np.pad(log_mel, ((0, 0), (0, pad_width)), mode='constant')
    else:
        log_mel = log_mel[:, :max_len]
    input_tensor = torch.tensor(log_mel).unsqueeze(0).unsqueeze(0)

    model = TransformerClassifier(input_dim=128, n_classes=len(labels))
    model.load_state_dict(torch.load(os.path.join(root_dir, "models", "transformer_model.pth")))
    model.eval()

    with torch.no_grad():
        out = model(input_tensor)
        pred = torch.argmax(out, dim=1).item()
    return labels[pred]
