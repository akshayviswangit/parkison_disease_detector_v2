import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import librosa
import pandas as pd
import os

def record_audio(duration=5, sample_rate=22050, output_file="user_audio.wav"):
    """
    Record audio from the user's microphone.
    
    Parameters:
        duration (int): Recording duration in seconds.
        sample_rate (int): Sampling rate of the audio.
        output_file (str): Path to save the recorded audio.
    """
    print("Recording...")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()  # Wait until recording is finished
    print("Recording complete.")

    # Save the recorded audio as a WAV file
    wav.write(output_file, sample_rate, (audio * 32767).astype(np.int16))  # Convert to 16-bit PCM format
    print(f"Audio saved to {output_file}")

def extract_features(file_path):
    """Extract features from an audio file."""
    # Load the audio file
    y, sr = librosa.load(file_path, sr=None)
    features = {}
    
    # 1. Fundamental frequency features
    f0, voiced_flag, voiced_probs = librosa.pyin(y, 
                                                fmin=librosa.note_to_hz('C2'), 
                                                fmax=librosa.note_to_hz('C7'))
    f0_cleaned = f0[voiced_flag]
    if len(f0_cleaned) > 0:
        features['MDVP:Fo(Hz)'] = np.mean(f0_cleaned)
        features['MDVP:Fhi(Hz)'] = np.max(f0_cleaned)
        features['MDVP:Flo(Hz)'] = np.min(f0_cleaned)
    else:
        features['MDVP:Fo(Hz)'] = 0.0
        features['MDVP:Fhi(Hz)'] = 0.0
        features['MDVP:Flo(Hz)'] = 0.0
    
    # 2. Jitter features
    y_frames = librosa.util.frame(y, frame_length=2048, hop_length=512)
    frame_means = np.mean(np.abs(y_frames), axis=0)
    jitter = np.diff(frame_means)
    features['MDVP:Jitter(%)'] = np.std(jitter) * 100
    features['MDVP:Jitter(Abs)'] = np.mean(np.abs(jitter))
    features['MDVP:RAP'] = np.mean(np.abs(np.diff(jitter)))
    features['MDVP:PPQ'] = np.percentile(np.abs(jitter), 25)
    features['Jitter:DDP'] = np.mean(np.abs(np.diff(np.diff(frame_means))))
    
    # 3. Shimmer features
    rms = librosa.feature.rms(y=y)[0]
    shimmer = np.diff(rms)
    features['MDVP:Shimmer'] = np.std(shimmer) * 100
    features['MDVP:Shimmer(dB)'] = librosa.amplitude_to_db(np.std(shimmer))
    features['Shimmer:APQ3'] = np.mean(np.abs(shimmer))
    features['Shimmer:APQ5'] = np.percentile(np.abs(shimmer), 75)
    features['MDVP:APQ'] = np.mean(shimmer)
    features['Shimmer:DDA'] = np.mean(np.abs(np.diff(shimmer)))
    
    # 4. Noise and harmonicity measures
    harmonics = librosa.effects.harmonic(y)
    noise = y - harmonics
    features['NHR'] = np.mean(np.abs(noise)) / np.mean(np.abs(harmonics))
    features['HNR'] = librosa.feature.spectral_rolloff(y=y)[0].mean()
    
    # 5. Nonlinear measures
    mfccs = librosa.feature.mfcc(y=y, sr=sr)
    features['RPDE'] = np.std(mfccs) / np.mean(mfccs)
    features['DFA'] = np.mean(np.abs(np.diff(mfccs, axis=1)))
    
    # 6. Spread and complexity measures
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    features['spread1'] = np.std(spec_cent)
    features['spread2'] = np.percentile(spec_cent, 75) - np.percentile(spec_cent, 25)
    features['D2'] = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    features['PPE'] = np.sum(np.abs(np.diff(mfccs, axis=1))) / mfccs.shape[1]
    
    # Ensure all values are finite
    for key in features:
        if not np.isfinite(features[key]):
            features[key] = 0.0
    
    return features

def save_features_to_csv(features, output_csv):
    """
    Save extracted features to a CSV file with proper scaling.
    """
    # Define the correct feature order
    feature_order = [
        'MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)', 'MDVP:Jitter(Abs)',
        'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP', 'MDVP:Shimmer', 'MDVP:Shimmer(dB)',
        'Shimmer:APQ3', 'Shimmer:APQ5', 'MDVP:APQ', 'Shimmer:DDA', 'NHR', 'HNR',
        'RPDE', 'DFA', 'spread1', 'spread2', 'D2', 'PPE'
    ]
    
    # Normalize features to expected ranges
    normalizers = {
        'MDVP:Fo(Hz)': lambda x: x / 1000,  # Normalize frequency to kHz
        'MDVP:Fhi(Hz)': lambda x: x / 1000,
        'MDVP:Flo(Hz)': lambda x: x / 1000,
        'MDVP:Jitter(%)': lambda x: min(x, 100),  # Cap percentage values
        'MDVP:Shimmer': lambda x: min(x, 100),
        'HNR': lambda x: x / 10000,  # Scale down high HNR values
        'RPDE': lambda x: abs(x),  # Ensure positive values
        'DFA': lambda x: abs(x),
        'spread1': lambda x: x / 100,
        'spread2': lambda x: x / 100,
        'D2': lambda x: x / 1000,
        'PPE': lambda x: min(abs(x), 100)
    }
    
    # Apply normalization
    normalized_features = features.copy()
    for key in normalized_features:
        if key in normalizers:
            normalized_features[key] = normalizers[key](features[key])
    
    # Create DataFrame with normalized values
    df = pd.DataFrame([normalized_features], columns=feature_order)
    
    # Add validation
    print("\nFeature Statistics:")
    print("-" * 50)
    for col in feature_order:
        value = df[col].iloc[0]
        print(f"{col}: {value:.6f}")
    
    # Save to CSV
    df.to_csv(output_csv, index=False, float_format='%.6f')
    print(f"\nFeatures saved to {output_csv}")

    assessment, risk_factors, risk_details = assess_parkinsons(normalized_features)
    print("\nParkinson's Assessment:")
    print(f"Result: {assessment}")
    print(f"Risk Factors Found: {risk_factors}/4")
    if risk_details:
        print("Detected Issues:")
        for detail in risk_details:
            print(f"- {detail}")

def process_audio(file_path, output_csv):
    """Process audio file and extract features with verification."""
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found.")
        return
    
    print(f"\nProcessing audio file: {file_path}")
    print("-" * 50)
    
    # Extract features
    features = extract_features(file_path)
    
    # Verify features
    zero_features = [k for k, v in features.items() if abs(v) < 1e-6]
    if zero_features:
        print("\nWarning: The following features have very low values:")
        for feat in zero_features:
            print(f"- {feat}")
    
    # Save features with normalization
    save_features_to_csv(features, output_csv)
    
    # Verify saved file
    try:
        df = pd.read_csv(output_csv)
        print("\nVerification:")
        print(f"- Features saved: {len(df.columns)}")
        print(f"- File size: {os.path.getsize(output_csv)} bytes")
        print(f"- Non-zero features: {(df != 0).sum().sum()}")
    except Exception as e:
        print(f"Error verifying saved file: {str(e)}")

    # Add this function after the existing functions, before main()
def assess_parkinsons(features):
    """
    Assess the likelihood of Parkinson's Disease based on voice features.
    Returns tuple of (assessment, risk_factors)
    """
    risk_factors = 0
    risk_details = []
    
    # Define thresholds based on research
    if features['MDVP:Jitter(%)'] > 1.0:
        risk_factors += 1
        risk_details.append("High Jitter percentage")
    if features['MDVP:Shimmer'] > 2.0:
        risk_factors += 1
        risk_details.append("Elevated Shimmer")
    if features['HNR'] < 0.6:
        risk_factors += 1
        risk_details.append("Low Harmonics to Noise Ratio")
    if features['PPE'] > 50.0:
        risk_factors += 1
        risk_details.append("High Pitch Period Entropy")
    
    has_parkinsons = risk_factors >= 3
    return "Yes" if has_parkinsons else "No", risk_factors, risk_details


def main():
    print("Parkinson's Disease Detection using Voice")
    print("Choose an option:")
    print("1. Record audio")
    print("2. Upload audio file")
    choice = input("Enter your choice (1 or 2): ").strip()
    
    if choice == "1":
        # Record audio
        output_audio_file = "user_audio.wav"
        record_audio(duration=5, output_file=output_audio_file)
        output_csv = "user_audio_features.csv"
        process_audio(output_audio_file, output_csv)
    
    elif choice == "2":
        # Upload audio file
        file_path = input("Enter the path to the audio file: ").strip()
        output_csv = "user_audio_features.csv"
        process_audio(file_path, output_csv)
    
    else:
        print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()