import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os

class AudioDataset(Dataset):
    def __init__(self, clean_files, noise_files, sample_length=16384):
        self.clean_files = clean_files
        self.noise_files = noise_files
        self.sample_length = sample_length
        
    def __len__(self):
        return len(self.clean_files)
    
    def __getitem__(self, idx):
        # Load clean audio
        clean, sr = torchaudio.load(self.clean_files[idx])
        # Randomly select noise
        noise_idx = np.random.randint(0, len(self.noise_files))
        noise, _ = torchaudio.load(self.noise_files[noise_idx])
        
        # Ensure same length and mix
        if clean.size(1) > self.sample_length:
            start = np.random.randint(0, clean.size(1) - self.sample_length)
            clean = clean[:, start:start + self.sample_length]
        else:
            clean = torch.nn.functional.pad(clean, (0, self.sample_length - clean.size(1)))
            
        if noise.size(1) != clean.size(1):
            noise = torch.nn.functional.pad(noise, (0, clean.size(1) - noise.size(1)))
        
        # Mix clean and noise
        noise_level = np.random.uniform(0.1, 0.5)
        noisy = clean + noise_level * noise
        
        return noisy, clean

class DenoisingAutoencoder(nn.Module):
    def __init__(self):
        super(DenoisingAutoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=32, stride=2, padding=15),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=32, stride=2, padding=15),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=32, stride=2, padding=15),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=32, stride=2, padding=15),
            nn.ReLU()
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(128, 64, kernel_size=32, stride=2, padding=15),
            nn.ReLU(),
            nn.ConvTranspose1d(64, 32, kernel_size=32, stride=2, padding=15),
            nn.ReLU(),
            nn.ConvTranspose1d(32, 16, kernel_size=32, stride=2, padding=15),
            nn.ReLU(),
            nn.ConvTranspose1d(16, 1, kernel_size=32, stride=2, padding=15),
            nn.Tanh()
        )
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class AudioDenoiser:
    def __init__(self, model_path=None, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = DenoisingAutoencoder().to(device)
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        
    def train(self, clean_files, noise_files, epochs=100, batch_size=32, save_path='models/denoiser.pth'):
        dataset = AudioDataset(clean_files, noise_files)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters())
        
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for noisy, clean in dataloader:
                noisy, clean = noisy.to(self.device), clean.to(self.device)
                
                optimizer.zero_grad()
                output = self.model(noisy)
                loss = criterion(output, clean)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            print(f'Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.6f}')
        
        # Save the model
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(self.model.state_dict(), save_path)
    
    def denoise(self, audio):
        self.model.eval()
        with torch.no_grad():
            # Convert numpy array to tensor
            audio_tensor = torch.FloatTensor(audio).unsqueeze(0).unsqueeze(0).to(self.device)
            
            # Process in chunks if audio is too long
            chunk_size = 16384
            if audio_tensor.size(-1) > chunk_size:
                denoised_chunks = []
                for i in range(0, audio_tensor.size(-1), chunk_size):
                    chunk = audio_tensor[:, :, i:i+chunk_size]
                    denoised_chunk = self.model(chunk)
                    denoised_chunks.append(denoised_chunk)
                denoised_audio = torch.cat(denoised_chunks, dim=2)
            else:
                denoised_audio = self.model(audio_tensor)
            
            return denoised_audio.squeeze().cpu().numpy() 