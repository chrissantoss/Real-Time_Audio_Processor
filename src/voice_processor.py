import numpy as np
from scipy import signal
import librosa
import scipy.ndimage
from scipy import ndimage
import os
import sys

# Add parent directory to Python path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.denoising_autoencoder import AudioDenoiser

class VoiceProcessor:
    def __init__(self, sample_rate):
        self.sr = sample_rate
        self.frame_length = 2048
        self.hop_length = 512
        self.denoiser = AudioDenoiser()
        
    def _normalize_audio(self, audio, target_peak=0.9):
        """Normalize audio to a target peak amplitude"""
        peak = np.max(np.abs(audio))
        if peak > 0:
            normalized = audio * (target_peak / peak)
            return normalized
        return audio
        
    def _spectral_subtraction(self, audio):
        """Basic spectral subtraction with adjusted parameters"""
        # Convert to frequency domain
        stft = librosa.stft(audio, n_fft=self.frame_length, hop_length=self.hop_length)
        mag = np.abs(stft)
        phase = np.angle(stft)
        
        # Estimate noise from more frames and use median instead of mean
        noise_estimate = np.median(mag[:, :20], axis=1, keepdims=True)
        
        # Add oversubtraction factor and flooring
        oversubtraction = 1.2  # Adjust between 1.0 and 2.0
        floor = 0.1  # Adjust between 0.01 and 0.2
        
        # Subtract noise spectrum with flooring
        mag_reduced = np.maximum(mag - oversubtraction * noise_estimate, floor * mag)
        
        # Reconstruct signal
        stft_reduced = mag_reduced * np.exp(1j * phase)
        return librosa.istft(stft_reduced, hop_length=self.hop_length)
    
    def reduce_noise(self, audio):
        """Enhanced noise reduction with less aggressive settings"""
        # Apply gentler spectral subtraction
        denoised = self._spectral_subtraction(audio)
        
        # Skip DAE for now as it might be causing artifacts
        # denoised = self.denoiser.denoise(spec_subtracted)
        
        return denoised
    
    def adjust_pitch(self, audio):
        # Basic pitch shifting using librosa
        return librosa.effects.pitch_shift(audio, sr=self.sr, n_steps=0)
    
    def detect_voice_activity(self, audio):
        # Simple energy-based VAD
        energy = librosa.feature.rms(y=audio, frame_length=self.frame_length, hop_length=self.hop_length)
        threshold = np.mean(energy) * 0.5
        return energy > threshold
    
    def adjust_tone(self, audio):
        """Implement multi-band equalization for tone adjustment"""
        # Define frequency bands
        bands = {
            'low': (20, 250),
            'mid': (250, 2000), 
            'high': (2000, 8000)
        }
        
        # Convert to frequency domain
        stft = librosa.stft(audio, n_fft=self.frame_length, hop_length=self.hop_length)
        freqs = librosa.fft_frequencies(sr=self.sr, n_fft=self.frame_length)
        
        # Apply gain to different frequency bands
        for band_name, (low, high) in bands.items():
            # Create band mask
            mask = (freqs >= low) & (freqs <= high)
            
            # Apply gain (adjust these values as needed)
            gains = {
                'low': 1.2,  # Boost bass
                'mid': 1.0,  # Keep mids neutral
                'high': 1.1  # Slightly boost highs
            }
            
            stft[mask] *= gains[band_name]
        
        # Convert back to time domain
        return librosa.istft(stft, hop_length=self.hop_length)
    
    def process(self, audio):
        """Process the audio with noise reduction and normalization"""
        # Apply noise reduction
        denoised = self.reduce_noise(audio)
        
        # Normalize the output to prevent low volume
        normalized = self._normalize_audio(denoised)
        
        return normalized 