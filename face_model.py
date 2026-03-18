"""
Face Recognition Model Pipeline
Formative 2 - Multimodal Data Preprocessing
Usage: python face_model.py
"""

import os
import numpy as np
import pandas as pd
import cv2
import joblib
from PIL import Image, ExifTags
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report

# ─── Configuration ────────────────────────────────────────────────────────────
MEMBERS     = ['sheryl', 'jok', 'innocent', 'vincent']
EXPRESSIONS = ['neutral', 'smiling', 'surprised']
IMAGES_DIR  = 'images'
FEATURES_DIR = 'features'
MODELS_DIR  = 'models'

os.makedirs(FEATURES_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# ─── Helper Functions ─────────────────────────────────────────────────────────

def find_image(member, expression):
    """Find image file with any supported extension."""
    folder = f'{IMAGES_DIR}/{member}'
    if not os.path.exists(folder):
        return None
    for filename in os.listdir(folder):
        name, ext = os.path.splitext(filename)
        if name.lower().strip() == expression.lower():
            if ext.lower() in ['.jpg', '.jpeg', '.png', '.heic']:
                return os.path.join(folder, filename)
    return None


def load_image(member, expression):
    """Load image with EXIF rotation fix."""
    path = find_image(member, expression)
    if path is None:
        print(f'Image not found: {member} - {expression}')
        return None, None

    img = Image.open(path).convert('RGB')

    # Fix iPhone rotation
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

    # Fix vincent rotation manually
    if member == 'vincent':
        img = img.rotate(90, expand=True)

    img_array = np.array(img)
    return cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR), path


def extract_image_features(img_path, member, expression):
    """Extract color histogram and pixel embedding features."""
    pil_img = Image.open(img_path).convert('RGB')

    # Fix rotation
    try:
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation':
                break
        exif = pil_img._getexif()
        if exif and orientation in exif:
            if exif[orientation] == 3:
                pil_img = pil_img.rotate(180, expand=True)
            elif exif[orientation] == 6:
                pil_img = pil_img.rotate(270, expand=True)
            elif exif[orientation] == 8:
                pil_img = pil_img.rotate(90, expand=True)
    except:
        pass

    if member == 'vincent':
        pil_img = pil_img.rotate(90, expand=True)

    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    img_resized = cv2.resize(img, (64, 64))
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

    # Color histograms
    hist_b = cv2.calcHist([img_resized], [0], None, [16], [0, 256]).flatten()
    hist_g = cv2.calcHist([img_resized], [1], None, [16], [0, 256]).flatten()
    hist_r = cv2.calcHist([img_resized], [2], None, [16], [0, 256]).flatten()

    mean_b = img_resized[:, :, 0].mean()
    mean_g = img_resized[:, :, 1].mean()
    mean_r = img_resized[:, :, 2].mean()
    embedding = gray.flatten()[:64]

    features = {
        'member': member,
        'expression': expression,
        'mean_b': mean_b, 'mean_g': mean_g, 'mean_r': mean_r,
    }
    for i, v in enumerate(hist_b): features[f'hist_b_{i}'] = v
    for i, v in enumerate(hist_g): features[f'hist_g_{i}'] = v
    for i, v in enumerate(hist_r): features[f'hist_r_{i}'] = v
    for i, v in enumerate(embedding): features[f'emb_{i}'] = v

    return features

# ─── Main Pipeline ────────────────────────────────────────────────────────────

def main():
    print('=' * 60)
    print('  FACE RECOGNITION MODEL PIPELINE')
    print('=' * 60)

    # Step 1: Extract features
    print('\n[1/4] Extracting image features...')
    image_features = []
    for m in MEMBERS:
        for e in EXPRESSIONS:
            _, path = load_image(m, e)
            if path:
                features = extract_image_features(path, m, e)
                image_features.append(features)
                print(f'  Extracted: {m} - {e}')

    img_df = pd.DataFrame(image_features)
    img_df.to_csv(f'{FEATURES_DIR}/image_features.csv', index=False)
    print(f'  image_features.csv saved! Shape: {img_df.shape}')

    # Step 2: Prepare data
    print('\n[2/4] Preparing training data...')
    le_img = LabelEncoder()
    X = img_df.drop(columns=['member', 'expression']).values
    y = le_img.fit_transform(img_df['member'])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    scaler_img = StandardScaler()
    X_train = scaler_img.fit_transform(X_train)
    X_test  = scaler_img.transform(X_test)
    print(f'  Train: {X_train.shape}, Test: {X_test.shape}')

    # Step 3: Train model
    print('\n[3/4] Training Random Forest model...')
    face_model = RandomForestClassifier(n_estimators=100, random_state=42)
    face_model.fit(X_train, y_train)

    # Step 4: Evaluate
    print('\n[4/4] Evaluating model...')
    y_pred = face_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, average='weighted')

    print(f'\n  Accuracy : {acc:.4f}')
    print(f'  F1 Score : {f1:.4f}')
    print('\n  Classification Report:')
    unique_classes = np.unique(y_test)
    print(classification_report(
        y_test, y_pred,
        labels=unique_classes,
        target_names=le_img.classes_[unique_classes]
    ))

    # Save models
    joblib.dump(face_model, f'{MODELS_DIR}/face_model.pkl')
    joblib.dump(scaler_img, f'{MODELS_DIR}/scaler_img.pkl')
    joblib.dump(le_img,     f'{MODELS_DIR}/le_img.pkl')

    print('\n' + '=' * 60)
    print('  Models saved to models/ folder')
    print('  face_model.pkl')
    print('  scaler_img.pkl')
    print('  le_img.pkl')
    print('=' * 60)


if __name__ == '__main__':
    main()
