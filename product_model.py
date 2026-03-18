"""
Product Recommendation Model Pipeline
Formative 2 - Multimodal Data Preprocessing
Usage: python product_model.py
"""

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from xgboost import XGBClassifier

# ─── Configuration ────────────────────────────────────────────────────────────
DATA_DIR     = 'data'
MODELS_DIR   = 'models'
FEATURES_DIR = 'features'

os.makedirs(MODELS_DIR, exist_ok=True)

# ─── Main Pipeline ────────────────────────────────────────────────────────────

def main():
    print('=' * 60)
    print('  PRODUCT RECOMMENDATION MODEL PIPELINE')
    print('=' * 60)

    # Step 1: Load merged dataset
    print('\n[1/4] Loading merged dataset...')
    merged = pd.read_csv(f'{DATA_DIR}/merged_dataset.csv')
    print(f'  Shape: {merged.shape}')
    print(f'  Columns: {merged.columns.tolist()}')

    # Step 2: Feature engineering
    print('\n[2/4] Engineering features...')
    le = LabelEncoder()

    # Encode categoricals
    cat_cols = ['social_media_platform', 'review_sentiment', 'product_category']
    for col in cat_cols:
        if col in merged.columns:
            merged[col + '_enc'] = le.fit_transform(merged[col].astype(str))

    # Date features
    if 'purchase_date' in merged.columns:
        merged['purchase_date'] = pd.to_datetime(merged['purchase_date'])
        merged['purchase_month']     = merged['purchase_date'].dt.month
        merged['purchase_dayofweek'] = merged['purchase_date'].dt.dayofweek

    # Engineered features
    if 'purchase_amount' in merged.columns and 'customer_rating' in merged.columns:
        merged['value_per_rating'] = merged['purchase_amount'] / (merged['customer_rating'] + 1)
    if 'engagement_score' in merged.columns and 'purchase_interest_score' in merged.columns:
        merged['engagement_x_interest'] = merged['engagement_score'] * merged['purchase_interest_score']

    # Feature columns
    feature_cols = [
        'engagement_score',
        'purchase_interest_score',
        'purchase_amount',
        'customer_rating',
        'purchase_month',
        'purchase_dayofweek',
        'value_per_rating',
        'engagement_x_interest',
        'social_media_platform_enc',
        'review_sentiment_enc',
    ]

    # Keep only available columns
    feature_cols = [c for c in feature_cols if c in merged.columns]
    print(f'  Features used: {feature_cols}')

    # Step 3: Train model
    print('\n[3/4] Training XGBoost model...')
    le_prod = LabelEncoder()
    X = merged[feature_cols].values
    y = le_prod.fit_transform(merged['product_category'])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler_prod = StandardScaler()
    X_train = scaler_prod.fit_transform(X_train)
    X_test  = scaler_prod.transform(X_test)

    product_model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        use_label_encoder=False,
        eval_metric='mlogloss',
        random_state=42
    )
    product_model.fit(X_train, y_train)

    # Step 4: Evaluate
    print('\n[4/4] Evaluating model...')
    y_pred = product_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, average='weighted')

    print(f'\n  Accuracy : {acc:.4f}')
    print(f'  F1 Score : {f1:.4f}')
    print('\n  Classification Report:')
    print(classification_report(y_test, y_pred, target_names=le_prod.classes_))

    # Save models
    joblib.dump(product_model, f'{MODELS_DIR}/product_model.pkl')
    joblib.dump(scaler_prod,   f'{MODELS_DIR}/scaler_prod.pkl')
    joblib.dump(le_prod,       f'{MODELS_DIR}/le_prod.pkl')
    joblib.dump(feature_cols,  f'{MODELS_DIR}/feature_cols.pkl')

    print('\n' + '=' * 60)
    print('  Models saved to models/ folder')
    print('  product_model.pkl')
    print('  scaler_prod.pkl')
    print('  le_prod.pkl')
    print('  feature_cols.pkl')
    print('=' * 60)


if __name__ == '__main__':
    main()
