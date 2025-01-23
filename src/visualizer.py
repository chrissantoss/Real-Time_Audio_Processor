import matplotlib.pyplot as plt
import librosa.display
import numpy as np

class AudioVisualizer:
    def __init__(self):
        # Remove plt.ion() as we're handling the display in demo.py
        pass
        
    def plot_waveform(self, audio, sr):
        # Don't create new figure here, just plot in current axes
        librosa.display.waveshow(audio, sr=sr)
        plt.title('Waveform')
        
    def plot_spectrogram(self, audio, sr):
        D = librosa.amplitude_to_db(np.abs(librosa.stft(audio)))
        # Don't create new figure here, just plot in current axes
        librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='hz')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Spectrogram') 