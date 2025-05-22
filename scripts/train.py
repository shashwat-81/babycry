# scripts/train.py
import torch
import torch.nn as nn
import torch.optim as optim
from scripts.preprocess import BabyCryDataset
from torch.utils.data import DataLoader
import os

class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, n_classes, nhead=4, num_layers=2):
        super().__init__()
        self.input_dim = input_dim
        self.encoder = nn.Linear(input_dim, 128)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=128, nhead=nhead),
            num_layers=num_layers
        )
        self.classifier = nn.Linear(128, n_classes)

    def forward(self, x):
        x = self.encoder(x.squeeze(1).permute(2, 0, 1))  # (seq_len, batch, features)
        x = self.transformer(x)
        out = self.classifier(x.mean(dim=0))
        return out

def train_model():
    # Get the absolute path to the data directory
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(root_dir, "data")
    models_dir = os.path.join(root_dir, "models")
    
    # Create models directory if it doesn't exist
    os.makedirs(models_dir, exist_ok=True)
    
    # Filter only WAV files and extract labels from filenames
    wav_files = [f for f in os.listdir(data_dir) if f.endswith('.wav')]
    if not wav_files:
        raise FileNotFoundError("No WAV files found in the data directory")
    
    # Assuming labels are encoded in filenames (modify this based on your naming convention)
    labels = sorted(list(set([f.split('-')[0] for f in wav_files])))
    print(f"Found {len(labels)} unique labels: {labels}")
    
    dataset = BabyCryDataset(data_dir, labels)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    model = TransformerClassifier(input_dim=128, n_classes=len(labels))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    print("Starting training...")
    for epoch in range(10):
        running_loss = 0.0
        for batch_idx, (X, y) in enumerate(dataloader):
            optimizer.zero_grad()
            output = model(X)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        epoch_loss = running_loss / len(dataloader)
        print(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}")

    model_path = os.path.join(models_dir, "transformer_model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    train_model()
