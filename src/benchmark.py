import time
import numpy as np
from voice_processor import VoiceProcessor
import soundfile as sf
import librosa

def measure_latency(processor, chunk_size=2048):
    """Measure processing latency for a single chunk"""
    # Generate test audio chunk
    test_chunk = np.random.randn(chunk_size)
    
    # Measure processing time
    start_time = time.time()
    processor.process(test_chunk)
    end_time = time.time()
    
    return (end_time - start_time) * 1000  # Convert to milliseconds

def benchmark_file_processing(processor, audio_file):
    """Benchmark full file processing"""
    # Load audio file
    audio, sr = librosa.load(audio_file)
    
    # Measure processing time
    start_time = time.time()
    processor.process(audio)
    end_time = time.time()
    
    duration = len(audio) / sr
    processing_time = end_time - start_time
    
    return {
        'duration': duration,
        'processing_time': processing_time,
        'realtime_factor': processing_time / duration
    }

def run_benchmarks():
    processor = VoiceProcessor(44100)
    
    # Test chunk latency
    latencies = [measure_latency(processor) for _ in range(100)]
    avg_latency = np.mean(latencies)
    max_latency = np.max(latencies)
    
    print(f"Average chunk processing latency: {avg_latency:.2f}ms")
    print(f"Maximum chunk processing latency: {max_latency:.2f}ms")
    
    # Test file processing if available
    try:
        results = benchmark_file_processing(processor, "data/sample.wav")
        print(f"\nFile processing results:")
        print(f"Audio duration: {results['duration']:.2f}s")
        print(f"Processing time: {results['processing_time']:.2f}s")
        print(f"Realtime factor: {results['realtime_factor']:.2f}x")
    except FileNotFoundError:
        print("\nNo test file found in data/sample.wav")

if __name__ == "__main__":
    run_benchmarks() 