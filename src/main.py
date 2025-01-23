import numpy as np
import librosa
import soundfile as sf
import os
from voice_processor import VoiceProcessor
from utils import load_audio, save_audio

def process_audio_file(input_file, output_file):
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found.")
        print("Please place an audio file in the data directory.")
        return False
        
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Load audio file
    audio, sr = load_audio(input_file)
    
    # Initialize voice processor
    processor = VoiceProcessor(sr)
    
    # Process audio
    enhanced_audio = processor.process(audio)
    
    # Save processed audio
    save_audio(output_file, enhanced_audio, sr)
    print(f"Successfully processed audio to: {output_file}")
    return True

if __name__ == "__main__":
    input_file = "data/sample.wav"
    output_file = "output/enhanced.wav"
    
    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)
    
    if not os.path.exists(input_file):
        print("\nNo sample.wav found. Recording a test audio file...")
        # Record 5 seconds of audio
        import sounddevice as sd
        duration = 5  # seconds
        fs = 44100  # Sample rate
        print("Recording 5 seconds... Say something!")
        recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
        sd.wait()
        sf.write(input_file, recording, fs)
        print(f"Saved test recording to {input_file}")
    
    process_audio_file(input_file, output_file) 