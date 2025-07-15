import os
import pyworld as pw
import numpy as np
import librosa
import soundfile as sf


def warp_spectral_envelope(sp, warp_factor):
    """Warp spectral envelope along frequency axis."""
    num_bins = sp.shape[1]
    x_old = np.linspace(0, 1, num_bins)
    x_new = np.clip(x_old ** warp_factor, 0, 1)  # Frequency axis warping

    warped_sp = np.zeros_like(sp)
    for frame in range(sp.shape[0]):
        warped_sp[frame, :] = np.interp(x_old, x_new, sp[frame, :])

    return warped_sp

def pitch_shift_f0(f0, semitones):
    """
    Shift pitch (F0) by a number of semitones.
    Semitones to multiplication factor: 2^(semitones/12)
    """
    factor = 2 ** (semitones / 12)
    f0_shifted = f0 * factor
    return f0_shifted

def process_audio_vtln_pitchshift_fixed_semitone(input_path, output_path, warp_factors, semitone_shift):
    
    y, sr = librosa.load(input_path, sr=16000)
    y = y.astype(np.float64)  # Required for pyworld

    # WORLD analysis
    f0, t = pw.harvest(y, sr)
    sp = pw.cheaptrick(y, f0, t, sr)
    ap = pw.d4c(y, f0, t, sr)


    base_filename = os.path.splitext(os.path.basename(input_path))[0]

  
    f0_shifted_base = pitch_shift_f0(f0, semitone_shift)
    f0_shifted_base[f0 == 0] = 0  # To keep unvoiced frames as zero

    for warp_factor in warp_factors:
        
        sp_warped = warp_spectral_envelope(sp, warp_factor)

        # Use the pitch-shifted f0 
        f0_shifted = f0_shifted_base

        
        # Synthesize audio
        y_warped = pw.synthesize(f0_shifted, sp_warped, ap, sr)

        
        output_filename = f"{base_filename}_{warp_factor:.2f}.wav"
        output_file_path = os.path.join(output_path, output_filename)

        sf.write(output_file_path, y_warped, sr)
        print(f"Saved warped audio: {output_file_path}")

def process_all_files_fixed_semitone(input_directory, output_directory, semitone_shift):
    
    warp_factors = [0.85]

    os.makedirs(output_directory, exist_ok=True)

    for filename in os.listdir(input_directory):
        if filename.endswith(".wav"):
            input_file_path = os.path.join(input_directory, filename)
            print(f"Processing {input_file_path}...")
            process_audio_vtln_pitchshift_fixed_semitone(input_file_path, output_directory, warp_factors, semitone_shift)


input_dir = "/home/drsandipan/Desktop/VTLN-Experiment/Pitch_shift_VTLN_Visualization/Original_audio/"
output_dir = "/home/drsandipan/Desktop/VTLN-Experiment/Pitch_shift_VTLN_Visualization/Pitch_shift_output_samples/-1/"
fixed_semitone_shift = -1.0  

process_all_files_fixed_semitone(input_dir, output_dir, fixed_semitone_shift)
