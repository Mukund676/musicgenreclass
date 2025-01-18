import librosa
import numpy as np

def extract_features(file_path):
    """
    Extract features from an audio file for genre classification.
    
    Parameters:
    - file_path: Path to the audio file.
    
    Returns:
    - features: A numpy array of the extracted features, aligned with the dataset.
    """
    # Load the audio file
    audio, sr = librosa.load(file_path, res_type='kaiser_fast')

    # Extract chroma_stft
    chroma_stft = librosa.feature.chroma_stft(y=audio, sr=sr)
    chroma_stft_mean = np.mean(chroma_stft)
    chroma_stft_var = np.var(chroma_stft)

    # Extract rms
    rms = librosa.feature.rms(y=audio)
    rms_mean = np.mean(rms)
    rms_var = np.var(rms)

    # Extract spectral centroid
    spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
    spectral_centroid_mean = np.mean(spectral_centroid)
    spectral_centroid_var = np.var(spectral_centroid)

    # Extract spectral bandwidth
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)
    spectral_bandwidth_mean = np.mean(spectral_bandwidth)
    spectral_bandwidth_var = np.var(spectral_bandwidth)

    # Extract rolloff
    rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
    rolloff_mean = np.mean(rolloff)
    rolloff_var = np.var(rolloff)

    # Extract zero crossing rate
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y=audio)
    zero_crossing_rate_mean = np.mean(zero_crossing_rate)
    zero_crossing_rate_var = np.var(zero_crossing_rate)

    # Extract harmony
    harmony = librosa.effects.harmonic(y=audio)
    harmony_mean = np.mean(harmony)
    harmony_var = np.var(harmony)

    # Extract percussive (perceptr)
    perceptr = librosa.effects.percussive(y=audio)
    perceptr_mean = np.mean(perceptr)
    perceptr_var = np.var(perceptr)

    # Extract tempo
    tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)

    # Extract MFCCs (20 coefficients)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)
    mfccs_mean = [np.mean(mfcc) for mfcc in mfccs]
    mfccs_var = [np.var(mfcc) for mfcc in mfccs]

    # Combine all features into a single flat feature vector
    features = np.array([
        chroma_stft_mean, chroma_stft_var,
        rms_mean, rms_var,
        spectral_centroid_mean, spectral_centroid_var,
        spectral_bandwidth_mean, spectral_bandwidth_var,
        rolloff_mean, rolloff_var,
        zero_crossing_rate_mean, zero_crossing_rate_var,
        harmony_mean, harmony_var,
        perceptr_mean, perceptr_var,
        tempo[0]
    ] + mfccs_mean + mfccs_var, dtype=np.float64).flatten()  # Ensure everything is flattened and casted to float

    return features
