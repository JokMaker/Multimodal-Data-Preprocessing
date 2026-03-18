"""
Voiceprint Verification Model Pipeline
Formative 2 - Multimodal Data Preprocessing
Usage: python voice_model.py
"""

import os
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report

# ─── Configuration ────────────────────────────────────────────────────────────
MEMBERS      = ['sheryl', 'jok', 'innocent', 'vincent']
PHRASES      = [
    'yes_approve', 'yes_approve_1', 'yes_approve_2',
    'confirm_transaction', 'confirm_transaction_1', 'confirm_transaction_2'
]
SOUND_DIR    = 'sound'
FEATURES_DIR = 'features'
MODELS_DIR   = 'models'

os.makedirs(FEATURES_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# ─── Helper Functions ─────────────────────────────────────────────────────────

def find_audio(member, phrase):
    """Find audio file with any supported extension."""
    folder = f'{SOUND_DIR}/{member}'
    if not os.path.exists(folder):
        return None
    for filename in os.listdir(folder):
        name, ext = os.path.splitext(filename)
        if name.lower().strip() == phrase.lower():
            if ext.lower() in ['.wav', '.mp3', '.m4a', '.ogg', '.mp4']:
                return os.path.join(folder, filename)
    return None


def extract_audio_features(audio_path, member, phrase):
    """Extract MFCC and other audio features."""
    y, sr = librosa.load(audio_path, sr=22050)
    features = {'member': member, 'phrase': phrase}

    # MFCCs (13 coefficients)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    for i, val in enumerate(mfccs.mean(axis=1)):
        features[f'mfcc_{i+1}_mean'] = val
    for i, val in enumerate(mfccs.std(axis=1)):
        features[f'mfcc_{i+1}_std'] = val

    # Spectral Roll-off
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    features['spectral_rolloff_mean'] = rolloff.mean()
    features['spectral_rolloff_std']  = rolloff.std()

    # Energy (RMS)
    rms = librosa.feature.rms(y=y)
    features['rms_energy_mean'] = rms.mean()
    features['rms_energy_std']  = rms.std()

    # Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(y)
    features['zcr_mean'] = zcr.mean()

    # Chroma
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    features['chroma_mean'] = chroma.mean()

    return features

# ─── Main Pipeline ────────────────────────────────────────────────────────────

def main():
    print('=' * 60)
    print('  VOICEPRINT VERIFICATION MODEL PIPELINE')
    print('=' * 60)

    # Step 1: Extract features
    print('\n[1/4] Extracting audio features...')
    audio_features = []
    for m in MEMBERS:
        for p in PHRASES:
            path = find_audio(m, p)
            if path:
                feat = extract_audio_features(path, m, p)
                audio_features.append(feat)
                print(f'  Extracted: {m} - {p}')
            else:
                print(f'  Not found: {m} - {p}')

    aud_df = pd.DataFrame(audio_features)
    aud_df.to_csv(f'{FEATURES_DIR}/audio_features.csv', index=False)
    print(f'\n  audio_features.csv saved! Shape: {aud_df.shape}')
    print('\n  Member counts:')
    print(aud_df['member'].value_counts().to_string())

    # Step 2: Prepare data
    print('\n[2/4] Preparing training data...')
    le_aud = LabelEncoder()
    X = aud_df.drop(columns=['member', 'phrase']).values
    y = le_aud.fit_transform(aud_df['member'])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    scaler_aud = StandardScaler()
    X_train = scaler_aud.fit_transform(X_train)
    X_test  = scaler_aud.transform(X_test)
    print(f'  Train: {X_train.shape}, Test: {X_test.shape}')

    # Step 3: Train model
    print('\n[3/4] Training Logistic Regression model...')
    voice_model = LogisticRegression(max_iter=1000, random_state=42)
    voice_model.fit(X_train, y_train)

    # Step 4: Evaluate
    print('\n[4/4] Evaluating model...')
    y_pred = voice_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, average='weighted')

    print(f'\n  Accuracy : {acc:.4f}')
    print(f'  F1 Score : {f1:.4f}')
    print('\n  Classification Report:')
    unique_classes = np.unique(y_test)
    print(classification_report(
        y_test, y_pred,
        labels=unique_classes,
        target_names=le_aud.classes_[unique_classes]
    ))

    # Save models
    joblib.dump(voice_model, f'{MODELS_DIR}/voice_model.pkl')
    joblib.dump(scaler_aud,  f'{MODELS_DIR}/scaler_aud.pkl')
    joblib.dump(le_aud,      f'{MODELS_DIR}/le_aud.pkl')

    print('\n' + '=' * 60)
    print('  Models saved to models/ folder')
    print('  voice_model.pkl')
    print('  scaler_aud.pkl')
    print('  le_aud.pkl')
    print('=' * 60)


if __name__ == '__main__':
    main()
