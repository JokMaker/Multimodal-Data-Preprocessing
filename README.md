# Formative 2: Multimodal Data Preprocessing
## User Identity & Product Recommendation System

---

## Group Members
| Member | Contribution |
|--------|-------------|
| Sheryl | Task 1: Data loading, EDA, data cleaning, merge, feature engineering |
| Jok | Task 2: Image collection, augmentation, feature extraction, face model, system simulation, GitHub |
| Innocent | Task 3: Audio collection, augmentation, feature extraction, voice model |
| Vincent | Task 4: Product recommendation model, evaluation, model saving |

---

## Repository Structure
```
├── multimodal_pipeline.ipynb
├── app.py
├── face_model.py
├── voice_model.py
├── product_model.py
├── requirements.txt
├── data/
│   ├── customer_social_profiles.csv
│   ├── customer_transactions.csv
│   └── merged_dataset.csv
├── images/
│   ├── sheryl/  (neutral.jpg, smiling.jpg, surprised.jpg)
│   ├── jok/
│   ├── innocent/
│   └── vincent/
├── sound/
│   ├── sheryl/  (yes_approve.wav, confirm_transaction.wav, ...)
│   ├── jok/
│   ├── innocent/
│   └── vincent/
├── features/
│   ├── image_features.csv
│   └── audio_features.csv
├── models/
│   ├── face_model.pkl
│   ├── voice_model.pkl
│   ├── product_model.pkl
│   └── (scalers and encoders)
└── augmented/
    ├── images/
    └── sound/
```

---

## Setup & Installation
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## How to Run

### Step 1: Clone the repository
```bash
git clone https://github.com/JokMaker/Multimodal-Data-Preprocessing.git
cd Multimodal-Data-Preprocessing
```

### Step 2: Run the Jupyter Notebook
```bash
jupyter notebook multimodal_pipeline.ipynb
```
Run all cells from top to bottom. This will:
- Merge and clean the datasets
- Augment images and audio
- Extract features to CSV files
- Train all 3 models
- Run system simulations

### Step 3: Run the CLI App
```bash
# Interactive menu
python app.py

# Direct authorized transaction
python app.py --image images/sheryl/smiling.jpg --audio sound/sheryl/yes_approve.wav

# Unauthorized simulation
python app.py --unauth
```

---

## System Flow
```
User Input
    │
    ▼
[Face Scan] ──── FAIL ──→ ACCESS DENIED
    │ PASS
    ▼
[Voice Check] ── FAIL ──→ ACCESS DENIED
    │ PASS
    ▼
[Identity Match Check] ── FAIL ──→ ACCESS DENIED
    │ PASS
    ▼
[Product Recommendation]
    │
    ▼
  RESULT DISPLAYED
```

---

## Models Used

| Model | Algorithm | Purpose |
|-------|-----------|---------|
| Face Recognition | Random Forest | Identify member from face image |
| Voiceprint Verification | Logistic Regression | Verify member identity from voice |
| Product Recommendation | XGBoost | Predict product from customer profile |

---

## Model Performance

| Model | Accuracy | F1 Score |
|-------|----------|----------|
| Face Recognition | 1.0000 | 1.0000 |
| Voiceprint Verification | 0.8750 | 0.8750 |
| Product Recommendation | 0.5349 | 0.5173 |

---

## Demo Video
https://drive.google.com/drive/folders/15nqE0bCxTmPU2XZ4ygOoQ5wsm9iL8Vqg?usp=sharing

## Report Link
https://docs.google.com/document/d/1N08t9xReuPNMh2tc94JI5Cn1tIJvejH0/edit