import os
import soundfile as sf
import numpy as np
from voice_processor import VoiceProcessor
from denoising_autoencoder import AudioDenoiser
from visualizer import AudioVisualizer
import matplotlib.pyplot as plt
from scipy import signal
from scipy.stats import pearsonr

def create_test_data():
    """Create synthetic test data if no real data is available"""
    # Create data directories
    os.makedirs('data/clean', exist_ok=True)
    os.makedirs('data/noise', exist_ok=True)
    os.makedirs('data/test', exist_ok=True)
    
    # Generate more realistic synthetic speech
    duration = 3  # seconds
    sr = 44100
    t = np.linspace(0, duration, int(sr * duration))
    
    # Create a mix of frequencies typical for human voice (100-400 Hz fundamental)
    f0 = 150  # fundamental frequency
    harmonics = [1, 2, 3, 4, 5]  # multiple harmonics
    amplitudes = [1.0, 0.5, 0.3, 0.2, 0.1]  # decreasing amplitudes for harmonics
    
    clean = np.zeros_like(t)
    for freq_mult, amp in zip(harmonics, amplitudes):
        # Add frequency modulation to simulate natural voice variation
        freq_mod = 1 + 0.1 * np.sin(2 * np.pi * 2 * t)
        clean += amp * np.sin(2 * np.pi * f0 * freq_mult * t * freq_mod)
    
    # Add amplitude modulation to simulate syllables
    syllable_rate = 4  # syllables per second
    amplitude_mod = 0.5 + 0.5 * (np.sin(2 * np.pi * syllable_rate * t) ** 2)
    clean *= amplitude_mod
    
    # Normalize clean signal
    clean = clean / np.max(np.abs(clean)) * 0.9
    
    # Generate realistic noise (pink noise)
    noise = np.random.randn(len(clean))
    # Create pink noise by applying 1/f filter
    f = np.fft.fftfreq(len(noise))
    f = np.abs(f)
    f[0] = 1e-6  # Avoid division by zero
    pink_filter = 1 / np.sqrt(f)
    pink_filter = pink_filter / np.max(pink_filter)
    noise_fft = np.fft.fft(noise) * pink_filter
    noise = np.real(np.fft.ifft(noise_fft))
    
    # Scale noise
    noise = noise / np.max(np.abs(noise)) * 0.1  # 10% noise level
    
    # Save test files
    sf.write('data/clean/test_clean.wav', clean, sr)
    sf.write('data/noise/test_noise.wav', noise, sr)
    
    # Create noisy test file
    noisy = clean + noise
    sf.write('data/test/test_noisy.wav', noisy, sr)
    
    return 'data/test/test_noisy.wav'

def reduce_static(audio_data, sr):
    """Enhanced static removal with multi-band processing"""
    frame_length = 2048
    hop_length = frame_length // 4

    original_length = len(audio_data)

    pre_emphasis = 0.95
    audio_emphasized = np.append(audio_data[0], audio_data[1:] - pre_emphasis * audio_data[:-1])

    f, t, Zxx = signal.stft(audio_emphasized, 
                           fs=sr, 
                           nperseg=frame_length,
                           noverlap=hop_length,
                           window='hann')

    voice_bands = [
        (50, 180),
        (181, 300),
        (301, 800),
        (801, 1500),
        (1501, 2500),
        (2501, 4000)
    ]

    wiener_params = [
        {'alpha': 3.0, 'beta': 0.05, 'oversubtract': 3.5},
        {'alpha': 3.5, 'beta': 0.04, 'oversubtract': 4.0},
        {'alpha': 4.0, 'beta': 0.03, 'oversubtract': 4.5},
        {'alpha': 4.5, 'beta': 0.02, 'oversubtract': 5.0},
        {'alpha': 5.0, 'beta': 0.01, 'oversubtract': 5.5},
        {'alpha': 5.5, 'beta': 0.01, 'oversubtract': 6.0}
    ]

    mag = np.abs(Zxx)
    phase = np.angle(Zxx)

    def estimate_noise(magnitude, p=10, smooth_window=11):
        noise_est = np.percentile(magnitude, p, axis=1, keepdims=True)
        smoothed = signal.medfilt(noise_est.squeeze(), smooth_window)
        return np.maximum(smoothed[:, np.newaxis], 1e-8)

    mag_reduced = np.zeros_like(mag)
    for band_idx, ((low, high), params) in enumerate(zip(voice_bands, wiener_params)):
        band_mask = (f >= low) & (f <= high)
        band_mag = mag[band_mask]

        noise_estimate = estimate_noise(band_mag, 
                                        p=10 if band_idx < 2 else 20, 
                                        smooth_window=9)

        subtracted = np.maximum(
            band_mag - params['oversubtract'] * noise_estimate,
            params['beta'] * band_mag
        )

        wiener_gain = np.maximum(
            1 - (params['alpha'] * noise_estimate / (subtracted + 1e-8)),
            params['beta']
        )

        wiener_gain = signal.medfilt2d(wiener_gain, kernel_size=(5, 5))

        mag_reduced[band_mask] = band_mag * wiener_gain

    noise_floor = np.percentile(mag_reduced, 2, axis=1, keepdims=True)
    mag_reduced = np.maximum(mag_reduced - 1.0 * noise_floor, 0)

    Zxx_reduced = mag_reduced * np.exp(1j * phase)
    _, audio_reduced = signal.istft(Zxx_reduced, 
                                  fs=sr, 
                                  nperseg=frame_length,
                                  noverlap=hop_length,
                                  window='hann')

    audio_reduced = np.append(audio_reduced[0], 
                            audio_reduced[1:] + pre_emphasis * audio_reduced[:-1])

    if len(audio_reduced) > original_length:
        audio_reduced = audio_reduced[:original_length]
    elif len(audio_reduced) < original_length:
        audio_reduced = np.pad(audio_reduced, (0, original_length - len(audio_reduced)))

    def compute_mix_ratio(orig, proc, window_size=1024):
        orig_energy = np.array([np.mean(orig[i:i+window_size]**2) 
                              for i in range(0, len(orig), window_size)])
        proc_energy = np.array([np.mean(proc[i:i+window_size]**2) 
                              for i in range(0, len(proc), window_size)])
        ratio = np.minimum(proc_energy / (orig_energy + 1e-8), 1.0)
        ratio = np.repeat(ratio, window_size)[:len(orig)]
        return np.clip(0.1 + 0.8 * ratio, 0.1, 0.9)

    mix_ratio = compute_mix_ratio(audio_data, audio_reduced)
    audio_reduced = mix_ratio * audio_reduced + (1 - mix_ratio) * audio_data

    max_val = np.max(np.abs(audio_reduced))
    if max_val > 1e-8:
        audio_reduced = audio_reduced / max_val

    return audio_reduced


def calculate_audio_metrics(original, processed, sr):
    """Calculate audio quality metrics with robust error handling"""
    # Ensure inputs are finite and normalized
    original = np.nan_to_num(original, nan=0.0, posinf=1.0, neginf=-1.0)
    processed = np.nan_to_num(processed, nan=0.0, posinf=1.0, neginf=-1.0)
    
    # Signal-to-Noise Ratio (SNR) with safety checks
    def calculate_snr(signal, noise):
        signal_power = np.mean(signal ** 2)
        noise_power = np.mean(noise ** 2)
        if noise_power < 1e-10:  # Avoid division by zero
            return 40.0  # Return a reasonable maximum value
        return 10 * np.log10(max(signal_power, 1e-10) / max(noise_power, 1e-10))
    
    # Root Mean Square Error (RMSE)
    rmse = np.sqrt(np.mean((original - processed) ** 2))
    
    # Peak Signal-to-Noise Ratio (PSNR) with safety checks
    mse = np.mean((original - processed) ** 2)
    if mse < 1e-10:  # Avoid log of zero
        psnr = 40.0  # Return a reasonable maximum value
    else:
        psnr = 20 * np.log10(1.0 / np.sqrt(mse))
    
    # Cross-correlation with safety checks
    if np.std(original) < 1e-10 or np.std(processed) < 1e-10:
        correlation = 0.95 if np.allclose(original, processed, rtol=1e-5) else 0.0
    else:
        try:
            correlation, _ = pearsonr(original, processed)
            correlation = 0.0 if np.isnan(correlation) else correlation
        except:
            correlation = 0.0
    
    # Noise reduction with safety checks
    noise = original - processed
    noise_reduction = calculate_snr(processed, noise)
    
    return {
        'rmse': float(rmse),
        'psnr': float(psnr),
        'correlation': float(correlation),
        'noise_reduction': float(noise_reduction)
    }

def optimize_noise_reduction(audio_data, sr, target_db=10.0, max_iterations=20):
    """Iteratively optimize noise reduction until target dB is achieved"""
    current_audio = audio_data.copy()
    original_audio = audio_data.copy()
    iteration = 0
    
    # Process first iteration
    processed = reduce_static(current_audio, sr)
    best_audio = processed
    best_metrics = calculate_audio_metrics(original_audio, processed, sr)
    best_db = best_metrics['noise_reduction']
    
    print("\nStarting noise reduction optimization...")
    print(f"Target: {target_db} dB")
    print(f"Iteration 1: Noise Reduction = {best_db:.2f} dB")
    
    # Check if first iteration achieved target
    if best_db >= target_db:
        print(f"✓ Target noise reduction of {target_db} dB achieved!")
        return best_audio, best_metrics
    
    # Continue with additional iterations if needed
    current_audio = processed
    iteration = 1
    
    while iteration < max_iterations:
        processed = reduce_static(current_audio, sr)
        metrics = calculate_audio_metrics(original_audio, processed, sr)
        current_db = metrics['noise_reduction']
        
        print(f"Iteration {iteration + 1}: Noise Reduction = {current_db:.2f} dB")
        
        if current_db > best_db:
            best_audio = processed
            best_metrics = metrics
            best_db = current_db
            
            if best_db >= target_db:
                print(f"✓ Target noise reduction of {target_db} dB achieved!")
                return best_audio, best_metrics
                
            current_audio = best_audio
        else:
            print(f"× No further improvement after {iteration + 1} iterations")
            break
            
        iteration += 1
    
    print(f"\nBest achieved noise reduction: {best_db:.2f} dB")
    return best_audio, best_metrics

def run_demo():
    print("Starting Voice Processing Demo...")
    
    # Try to use a real audio file first
    test_file = 'data/test/sample.wav'
    if not os.path.exists(test_file):
        test_file = 'data/test/test_noisy.wav'
        if not os.path.exists(test_file):
            test_file = create_test_data()
            print("Created synthetic test data")
        else:
            print("Using existing synthetic test data")
    else:
        print("Using real speech sample")
    
    # Load audio
    audio, sr = sf.read(test_file)
    print(f"Loaded audio file: {test_file}")
    
    # Initialize processor and visualizer
    processor = VoiceProcessor(sr)
    visualizer = AudioVisualizer()
    
    # Train DAE if needed
    if not os.path.exists('models/denoiser.pth'):
        print("\nTraining Denoising Autoencoder...")
        clean_files = [f for f in os.listdir('data/clean') if f.endswith('.wav')]
        noise_files = [f for f in os.listdir('data/noise') if f.endswith('.wav')]
        
        if clean_files and noise_files:
            clean_files = [os.path.join('data/clean', f) for f in clean_files]
            noise_files = [os.path.join('data/noise', f) for f in noise_files]
            
            denoiser = AudioDenoiser()
            denoiser.train(clean_files, noise_files, epochs=5)  # Quick training for demo
            print("DAE training completed")
        else:
            print("Warning: No training data found, using untrained DAE")
    
    # Clear any existing plots
    plt.close('all')
    
    # Create single figure with subplots for all visualizations
    fig = plt.figure(figsize=(15, 10))
    
    # Visualize original audio
    print("\nVisualizing original audio...")
    plt.subplot(2, 2, 1)
    visualizer.plot_waveform(audio, sr)
    plt.subplot(2, 2, 2)
    visualizer.plot_spectrogram(audio, sr)
    
    # Process audio
    print("\nProcessing audio...")
    processed = processor.process(audio)
    
    # Optimize noise reduction with clear progress feedback
    print("\nOptimizing noise reduction...")
    processed, final_metrics = optimize_noise_reduction(processed, sr, target_db=10.0)
    
    # Safe normalization function
    def safe_normalize(audio):
        """Safely normalize audio with checks for invalid values"""
        # Clean any NaN or inf values
        audio = np.nan_to_num(audio, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Get maximum absolute value
        max_val = np.max(np.abs(audio))
        
        # Only normalize if we have non-zero values
        if max_val > 1e-10:
            audio = audio / max_val
        else:
            print("Warning: Audio signal is too quiet, applying minimum gain")
            audio = audio * 1e-3
        
        # Ensure bounds
        return np.clip(audio, -1.0, 1.0)
    
    # Apply safe normalization before visualization
    processed = safe_normalize(processed)
    
    # Verify signal validity
    if not np.all(np.isfinite(processed)):
        print("Warning: Invalid values detected after processing, applying cleanup")
        processed = np.nan_to_num(processed, nan=0.0, posinf=1.0, neginf=-1.0)
        processed = np.clip(processed, -1.0, 1.0)
    
    # Visualize processed audio
    print("\nVisualizing processed audio...")
    plt.subplot(2, 2, 3)
    visualizer.plot_waveform(processed, sr)
    plt.subplot(2, 2, 4)
    visualizer.plot_spectrogram(processed, sr)
    
    # Print metrics
    print("\nAudio Quality Metrics:")
    print(f"Root Mean Square Error (RMSE): {final_metrics['rmse']:.4f}")
    print(f"Peak Signal-to-Noise Ratio (PSNR): {final_metrics['psnr']:.2f} dB")
    print(f"Signal Correlation: {final_metrics['correlation']:.4f}")
    print(f"Noise Reduction: {final_metrics['noise_reduction']:.2f} dB")
    
    # Add metrics to visualization
    plt.figtext(0.02, 0.02, 
                f"Metrics:\nRMSE: {final_metrics['rmse']:.4f}\n"
                f"PSNR: {final_metrics['psnr']:.2f} dB\n"
                f"Correlation: {final_metrics['correlation']:.4f}\n"
                f"Noise Reduction: {final_metrics['noise_reduction']:.2f} dB",
                fontsize=8, bbox=dict(facecolor='white', alpha=0.8))
    
    # Save processed audio
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'processed_audio.wav')
    sf.write(output_file, processed, sr)
    print(f"\nProcessed audio saved to: {output_file}")
    
    print("\nDemo completed!")
    print("Close the visualization window to exit...")
    
    # Adjust layout and display all plots
    plt.tight_layout()
    plt.show(block=True)  # block=True ensures the window stays open

if __name__ == "__main__":
    run_demo() 