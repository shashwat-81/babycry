# scripts/preprocess.py
import torchaudio
import os
import torch
import librosa
import numpy as np
from torch.utils.data import Dataset

class BabyCryDataset(Dataset):
    def __init__(self, root_dir, labels, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data = []
        self.labels = labels
        
        # Scan for WAV files directly in root_dir
        for file in os.listdir(root_dir):
            if file.endswith('.wav'):
                label = file.split('-')[0]  # Extract label from filename
                if label in labels:
                    label_idx = labels.index(label)
                    self.data.append((os.path.join(root_dir, file), label_idx))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        file_path, label = self.data[idx]
        waveform, sr = torchaudio.load(file_path)
        waveform = waveform.mean(dim=0)  # mono
        mel_spec = librosa.feature.melspectrogram(y=waveform.numpy(), sr=sr)
        log_mel = librosa.power_to_db(mel_spec)

        # Pad or truncate to fixed length
        max_len = 432  # or another value you choose
        if log_mel.shape[1] < max_len:
            pad_width = max_len - log_mel.shape[1]
            log_mel = np.pad(log_mel, ((0, 0), (0, pad_width)), mode='constant')
        else:
            log_mel = log_mel[:, :max_len]

        return torch.tensor(log_mel).unsqueeze(0), label
