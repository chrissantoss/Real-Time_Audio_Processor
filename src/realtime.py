import pyaudio
import numpy as np
import wave
from voice_processor import VoiceProcessor

CHUNK = 2048  # Increased chunk size to match FFT size
FORMAT = pyaudio.paFloat32
CHANNELS = 1
RATE = 44100

def process_realtime():
    p = pyaudio.PyAudio()
    processor = VoiceProcessor(RATE)
    
    stream = p.open(format=FORMAT,
                   channels=CHANNELS,
                   rate=RATE,
                   input=True,
                   output=True,
                   frames_per_buffer=CHUNK)
    
    print("* Recording and processing. Press Ctrl+C to stop.")
    
    try:
        while True:
            try:
                data = stream.read(CHUNK, exception_on_overflow=False)
                audio_data = np.frombuffer(data, dtype=np.float32)
                
                # Process audio
                enhanced = processor.process(audio_data)
                
                # Play processed audio
                stream.write(enhanced.astype(np.float32).tobytes())
            except OSError as e:
                print(f"Warning: {e}")
                continue

    except KeyboardInterrupt:
        print("\n* Stopped recording")
    finally:
        if stream.is_active():
            stream.stop_stream()
        stream.close()
        p.terminate()

if __name__ == "__main__":
    process_realtime() 