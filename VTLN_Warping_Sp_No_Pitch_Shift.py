import os
import pyworld as pw
import numpy as np
import librosa
import soundfile as sf


def process_audio(input_path, output_path):
    
    y, sr = librosa.load(input_path, sr=16000)  
    y = y.astype(np.float64)  

    # Step 1: WORLD Vocoder Analysis
    _f0, t = pw.harvest(y, sr)  # Pitch extraction
    sp = pw.cheaptrick(y, _f0, t, sr)  # Spectral envelope
    ap = pw.d4c(y, _f0, t, sr)  # Aperiodicity

    # Step 2: Warp the spectral envelope
    def warp_spectral_envelope(sp, warp_factor):
        """Warp spectral envelope along frequency axis."""
        num_bins = sp.shape[1]
        x_old = np.linspace(0, 1, num_bins)
        x_new = np.clip(x_old ** warp_factor, 0, 1)  # Frequency axis warping

        warped_sp = np.zeros_like(sp)
        for frame in range(sp.shape[0]):
            warped_sp[frame, :] = np.interp(x_old, x_new, sp[frame, :]) #used interpolation method

        return warped_sp

    
    sp_child = warp_spectral_envelope(sp, warp_factor=1.26)

    # Synthesize waveform from warped spectral envelope
    y_child = pw.synthesize(_f0, sp_child, ap, sr)

    # Save output with same filename in the target output directory
    output_file = os.path.join(output_path, os.path.basename(input_path))
    sf.write(output_file, y_child, sr)

# Function to process all .wav files in the input directory
def process_all_files(input_directory, output_directory):
    
    for filename in os.listdir(input_directory):
        if filename.endswith(".wav"):
            input_file_path = os.path.join(input_directory, filename)
            print(f"Processing {input_file_path}...")
            process_audio(input_file_path, output_directory)

input_dir = "/home/drsandipan/Desktop/VTLN-Experiment/" #Keep the input .wav files in this path 
output_dir = "/home/drsandipan/Desktop/VTLN-Experiment/" #Obtained output .wav files available to this path 
process_all_files(input_dir, output_dir)
