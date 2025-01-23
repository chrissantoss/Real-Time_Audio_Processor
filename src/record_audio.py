import soundfile as sf
import os
import pyaudio
import numpy as np
import time

def record_sample():
    # Create directory if it doesn't exist
    os.makedirs('data/test', exist_ok=True)
    
    # Recording parameters
    duration = 5  # seconds
    sample_rate = 44100
    channels = 1
    chunk = 1024
    
    # Initialize PyAudio
    p = pyaudio.PyAudio()
    
    print("Recording will start in 3 seconds...")
    for i in range(3, 0, -1):
        print(f"{i}...")
        time.sleep(1)
    
    print("Recording... Speak now!")
    
    # Open stream
    stream = p.open(
        format=pyaudio.paFloat32,
        channels=channels,
        rate=sample_rate,
        input=True,
        frames_per_buffer=chunk
    )
    
    # Record audio
    frames = []
    for i in range(0, int(sample_rate / chunk * duration)):
        data = stream.read(chunk)
        frames.append(np.frombuffer(data, dtype=np.float32))
    
    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    p.terminate()
    
    # Convert to numpy array and adjust gain
    recording = np.concatenate(frames) * 0.5  # Adjust gain here
    
    # Save the recording
    output_file = 'data/test/sample_speech.wav'
    sf.write(output_file, recording, sample_rate)
    
    print(f"\nRecording saved to: {output_file}")
    print(f"Duration: {duration} seconds")

if __name__ == "__main__":
    record_sample() 