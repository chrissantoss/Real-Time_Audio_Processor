import librosa
import soundfile as sf
import os

def load_audio(file_path):
    """Load audio file using librosa"""
    return librosa.load(file_path)

def save_audio(file_path, audio, sr):
    """Save audio file using soundfile"""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    sf.write(file_path, audio, sr) 