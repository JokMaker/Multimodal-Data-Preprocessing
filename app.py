"""
User Identity & Product Recommendation System
CLI Application - Formative 2
Usage: python app.py
"""

import os
import sys
import numpy as np
import cv2
import librosa
import soundfile as sf
import joblib
import argparse
from PIL import Image, ExifTags

# ─── Configuration ────────────────────────────────────────────────────────────
MODELS_DIR   = 'models'
IMAGES_DIR   = 'images'
SOUND_DIR    = 'sound'
MEMBERS      = ['sheryl', 'jok', 'innocent', 'vincent']
CONFIDENCE_THRESHOLD = 0.70

# ─── Helper Functions ─────────────────────────────────────────────────────────

def find_image(member, expression):
    folder = f'{IMAGES_DIR}/{member}'
    if not os.path.exists(folder):
        return None
    for filename in os.listdir(folder):
        name, ext = os.path.splitext(filename)
        if name.lower().strip() == expression.lower():
            if ext.lower() in ['.jpg', '.jpeg', '.png', '.heic']:
                return os.path.join(folder, filename)
    return None


def find_audio(member, phrase):
    folder = f'{SOUND_DIR}/{member}'
    if not os.path.exists(folder):
        return None
    for filename in os.listdir(folder):
        name, ext = os.path.splitext(filename)
        if name.lower().strip() == phrase.lower():
            if ext.lower() in ['.wav', '.mp3', '.m4a', '.ogg', '.mp4']:
                return os.path.join(folder, filename)
    return None


def load_image(image_path, member=None):
    img = Image.open(image_path).convert('RGB')
    try:
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation':
                break
        exif = img._getexif()
        if exif and orientation in exif:
            if exif[orientation] == 3:
                img = img.rotate(180, expand=True)
            elif exif[orientation] == 6:
                img = img.rotate(270, expand=True)
            elif exif[orientation] == 8:
                img = img.rotate(90, expand=True)
    except:
        pass
    if member == 'vincent':
        img = img.rotate(90, expand=True)
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

# ─── Feature Extraction ───────────────────────────────────────────────────────

def extract_image_features(img_path, member=None):
    img = load_image(img_path, member)
    img_resized = cv2.resize(img, (64, 64))
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    hist_b = cv2.calcHist([img_resized], [0], None, [16], [0, 256]).flatten()
    hist_g = cv2.calcHist([img_resized], [1], None, [16], [0, 256]).flatten()
    hist_r = cv2.calcHist([img_resized], [2], None, [16], [0, 256]).flatten()
    mean_b = img_resized[:, :, 0].mean()
    mean_g = img_resized[:, :, 1].mean()
    mean_r = img_resized[:, :, 2].mean()
    embedding = gray.flatten()[:64]
    features = [mean_b, mean_g, mean_r]
    features += list(hist_b) + list(hist_g) + list(hist_r) + list(embedding)
    return np.array(features)


def extract_audio_features(audio_path):
    y, sr = librosa.load(audio_path, sr=22050)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    features = (list(mfccs.mean(axis=1)) + list(mfccs.std(axis=1)) +
                [librosa.feature.spectral_rolloff(y=y, sr=sr).mean(),
                 librosa.feature.spectral_rolloff(y=y, sr=sr).std(),
                 librosa.feature.rms(y=y).mean(),
                 librosa.feature.rms(y=y).std(),
                 librosa.feature.zero_crossing_rate(y).mean(),
                 librosa.feature.chroma_stft(y=y, sr=sr).mean()])
    return np.array(features)

# ─── Model Loading ────────────────────────────────────────────────────────────

def load_models():
    try:
        face_model    = joblib.load(f'{MODELS_DIR}/face_model.pkl')
        voice_model   = joblib.load(f'{MODELS_DIR}/voice_model.pkl')
        product_model = joblib.load(f'{MODELS_DIR}/product_model.pkl')
        scaler_img    = joblib.load(f'{MODELS_DIR}/scaler_img.pkl')
        scaler_aud    = joblib.load(f'{MODELS_DIR}/scaler_aud.pkl')
        scaler_prod   = joblib.load(f'{MODELS_DIR}/scaler_prod.pkl')
        le_img        = joblib.load(f'{MODELS_DIR}/le_img.pkl')
        le_aud        = joblib.load(f'{MODELS_DIR}/le_aud.pkl')
        le_prod       = joblib.load(f'{MODELS_DIR}/le_prod.pkl')
        feature_cols  = joblib.load(f'{MODELS_DIR}/feature_cols.pkl')
        return (face_model, voice_model, product_model,
                scaler_img, scaler_aud, scaler_prod,
                le_img, le_aud, le_prod, feature_cols)
    except FileNotFoundError as e:
        print(f'[ERROR] Model not found: {e}')
        print('Please run the Jupyter notebook first to train and save models.')
        sys.exit(1)


def verify_face(image_path, face_model, scaler_img, le_img, member=None):
    X = extract_image_features(image_path, member).reshape(1, -1)
    X_scaled = scaler_img.transform(X)
    pred  = face_model.predict(X_scaled)[0]
    proba = face_model.predict_proba(X_scaled).max()
    return le_img.inverse_transform([pred])[0], proba


def verify_voice(audio_path, voice_model, scaler_aud, le_aud):
    X = extract_audio_features(audio_path).reshape(1, -1)
    X_scaled = scaler_aud.transform(X)
    pred  = voice_model.predict(X_scaled)[0]
    proba = voice_model.predict_proba(X_scaled).max()
    return le_aud.inverse_transform([pred])[0], proba


def predict_product(customer_data, product_model, scaler_prod, le_prod, feature_cols):
    X = np.array([customer_data.get(c, 0) for c in feature_cols]).reshape(1, -1)
    X_scaled = scaler_prod.transform(X)
    pred = product_model.predict(X_scaled)[0]
    return le_prod.inverse_transform([pred])[0]

# ─── Simulation Flows ─────────────────────────────────────────────────────────

def run_full_transaction(image_path, audio_path, customer_data, models):
    (face_model, voice_model, product_model,
     scaler_img, scaler_aud, scaler_prod,
     le_img, le_aud, le_prod, feature_cols) = models

    print('\n' + '=' * 60)
    print('   USER IDENTITY & PRODUCT RECOMMENDATION SYSTEM')
    print('=' * 60)

    print('\n[STEP 1/3] Facial Recognition...')
    try:
        member, conf = verify_face(image_path, face_model, scaler_img, le_img)
        print(f'  Detected  : {member}')
        print(f'  Confidence: {conf:.2%}')
    except Exception as e:
        print(f'  [ERROR] {e}')
        print('  ACCESS DENIED')
        return

    if conf < CONFIDENCE_THRESHOLD:
        print(f'  LOW CONFIDENCE ({conf:.2%} < {CONFIDENCE_THRESHOLD:.0%})')
        print('\n' + '-' * 60)
        print('  ACCESS DENIED - FACE NOT RECOGNIZED')
        print('-' * 60)
        return
    print(f'  FACE VERIFIED: {member}')

    print('\n[STEP 2/3] Voiceprint Verification...')
    try:
        v_member, v_conf = verify_voice(audio_path, voice_model, scaler_aud, le_aud)
        print(f'  Detected  : {v_member}')
        print(f'  Confidence: {v_conf:.2%}')
    except Exception as e:
        print(f'  [ERROR] {e}')
        print('  ACCESS DENIED')
        return

    if v_conf < CONFIDENCE_THRESHOLD:
        print(f'  LOW CONFIDENCE ({v_conf:.2%} < {CONFIDENCE_THRESHOLD:.0%})')
        print('\n' + '-' * 60)
        print('  ACCESS DENIED - VOICE NOT RECOGNIZED')
        print('-' * 60)
        return
    print(f'  VOICE VERIFIED: {v_member}')

    if member != v_member:
        print('  IDENTITY MISMATCH!')
        print('\n' + '-' * 60)
        print('  ACCESS DENIED - IDENTITY MISMATCH')
        print('-' * 60)
        return

    print('\n[STEP 3/3] Product Recommendation...')
    product = predict_product(customer_data, product_model, scaler_prod, le_prod, feature_cols)
    print('\n' + '=' * 60)
    print(f'  RECOMMENDED PRODUCT: {product.upper()}')
    print('=' * 60)
    print('  TRANSACTION COMPLETE - ACCESS GRANTED')
    print('=' * 60 + '\n')


def run_unauthorized_simulation(models):
    (face_model, voice_model, _,
     scaler_img, scaler_aud, _,
     le_img, le_aud, _, _) = models

    print('\n' + '=' * 60)
    print('   SIMULATION: UNAUTHORIZED ACCESS ATTEMPT')
    print('=' * 60)

    unknown_face  = 'images/unknown_face.jpg'
    unknown_audio = 'sound/unknown_voice.wav'
    cv2.imwrite(unknown_face, np.random.randint(0, 40, (224, 224, 3), dtype=np.uint8))
    sf.write(unknown_audio, 0.005 * np.random.randn(22050 * 2), 22050)

    print('\n[STEP 1] Scanning unknown face...')
    member, conf = verify_face(unknown_face, face_model, scaler_img, le_img)
    print(f'  Best match : {member}')
    print(f'  Confidence : {conf:.2%}')

    if conf < CONFIDENCE_THRESHOLD:
        print(f'  LOW CONFIDENCE ({conf:.2%} < {CONFIDENCE_THRESHOLD:.0%})')
        print('\n' + '-' * 60)
        print('  ACCESS DENIED - UNAUTHORIZED USER')
        print('  SECURITY ALERT: Attempt has been logged!')
        print('-' * 60 + '\n')
        return

    print('\n[STEP 2] Verifying voice...')
    v_member, v_conf = verify_voice(unknown_audio, voice_model, scaler_aud, le_aud)
    print(f'  Best match : {v_member}')
    print(f'  Confidence : {v_conf:.2%}')
    print('\n' + '-' * 60)
    print('  ACCESS DENIED - UNAUTHORIZED USER')
    print('  SECURITY ALERT: Attempt has been logged!')
    print('-' * 60 + '\n')

# ─── Interactive CLI Menu ─────────────────────────────────────────────────────

def interactive_menu(models):
    customer_data = {
        'engagement_score': 75,
        'purchase_interest_score': 4.2,
        'purchase_amount': 350.0,
        'customer_rating': 3.5,
        'purchase_month': 3,
        'purchase_dayofweek': 1,
        'value_per_rating': 75.0,
        'engagement_x_interest': 315.0,
        'social_media_platform_enc': 2,
        'review_sentiment_enc': 1
    }

    while True:
        print('\n' + '-' * 60)
        print('  MAIN MENU')
        print('-' * 60)
        print('  1. Run authorized transaction (sheryl)')
        print('  2. Run unauthorized access simulation')
        print('  3. Test with custom image & audio paths')
        print('  4. Exit')
        print('-' * 60)

        choice = input('  Select option [1-4]: ').strip()

        if choice == '1':
            img_path = find_image('sheryl', 'smiling')
            aud_path = find_audio('sheryl', 'yes_approve')
            run_full_transaction(img_path, aud_path, customer_data, models)
        elif choice == '2':
            run_unauthorized_simulation(models)
        elif choice == '3':
            img_path = input('  Enter image path: ').strip()
            aud_path = input('  Enter audio path: ').strip()
            run_full_transaction(img_path, aud_path, customer_data, models)
        elif choice == '4':
            print('\n  Goodbye!\n')
            break
        else:
            print('  Invalid option. Try again.')

# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Multimodal Identity & Product Recommendation System')
    parser.add_argument('--image',  type=str, help='Path to face image')
    parser.add_argument('--audio',  type=str, help='Path to voice audio')
    parser.add_argument('--unauth', action='store_true', help='Run unauthorized simulation')
    args = parser.parse_args()

    models = load_models()

    if args.unauth:
        run_unauthorized_simulation(models)
    elif args.image and args.audio:
        customer_data = {
            'engagement_score': 75,
            'purchase_interest_score': 4.2,
            'purchase_amount': 350.0,
            'customer_rating': 3.5,
            'purchase_month': 3,
            'purchase_dayofweek': 1,
            'value_per_rating': 75.0,
            'engagement_x_interest': 315.0,
            'social_media_platform_enc': 2,
            'review_sentiment_enc': 1
        }
        run_full_transaction(args.image, args.audio, customer_data, models)
    else:
        interactive_menu(models)
