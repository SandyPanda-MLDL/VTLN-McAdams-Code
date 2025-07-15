import librosa
import soundfile as sf
from libtsm.pitchshift import pitch_shift
import parselmouth
import numpy as np

# === Pitch extraction function ===
def extract_f0_parselmouth(wav_path,
                           time_step=0.01,
                           f0_min=150,
                           f0_max=500,
                           silence_threshold=0.03,
                           voicing_threshold=0.5,
                           octave_cost=0.01,
                           octave_jump_cost=0.35,
                           voiced_unvoiced_cost=0.14,
                           max_candidates=15):
    snd = parselmouth.Sound(wav_path)
    pitch = parselmouth.praat.call(
        snd, "To Pitch (ac)...",
        time_step, f0_min, max_candidates, f0_max,
        silence_threshold, voicing_threshold,
        octave_cost, octave_jump_cost, voiced_unvoiced_cost, f0_max
    )
    f0_values = pitch.selected_array['frequency']
    times = pitch.xs()
    voiced_mask = f0_values > 0
    return f0_values, voiced_mask, times

# === Input and output paths ===
input_path = "/home/Sharedata/sandipan/Voice_Editing_VTLN/VTLN-Experiment/EAAI-Final-Dataset/Experiment_With_libtsm_library/3a1f_EN-OL-RC-234_1.wav"
output_path = "/home/Sharedata/sandipan/Voice_Editing_VTLN/VTLN-Experiment/EAAI-Final-Dataset/Experiment_With_libtsm_library/3a1f_EN-OL-RC-234_1_Pitch_shifted.wav"

# === Load original audio ===
x, sr = librosa.load(input_path, sr=None)

# === Pitch shift by -700 cents (i.e. -7 semitones) ===
shift_in_cents = -700
y = pitch_shift(x, p=shift_in_cents, Fs=sr, order='res-tsm')

# === Save shifted audio ===
sf.write(output_path, y, sr)
print(f" Saved pitch-shifted audio to: {output_path}")

# === Extract pitch from both original and shifted audio ===
f0_orig, mask_orig, _ = extract_f0_parselmouth(input_path)
f0_shifted, mask_shifted, _ = extract_f0_parselmouth(output_path)

# === Align and compute pitch shift in semitones (only on voiced frames) ===
voiced_mask = mask_orig & mask_shifted
f0_orig_voiced = f0_orig[voiced_mask]
f0_shifted_voiced = f0_shifted[voiced_mask]

# Avoid division by zero
eps = 1e-6
semitone_diff = 12 * np.log2((f0_shifted_voiced + eps) / (f0_orig_voiced + eps))

# === Report average shift ===
if len(semitone_diff) > 0:
    avg_shift = np.mean(semitone_diff)
    print(f" Average semitone shift between original and shifted: {avg_shift:.2f} semitones")
else:
    print("No voiced frames found for pitch comparison!")
