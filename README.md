# Formative 2: Multimodal Data Preprocessing
## User Identity & Product Recommendation System

---

## 👥 Group Members
| Member | Contribution |
|--------|-------------|
| Sheryl Otieno | Image collection, Face model, CLI app |
| Jok John Maker | Audio collection, Voice model, augmentations |
| Innocent Nangah | Data merge, EDA, Feature engineering |
| Vincent Mugabo | Product model, evaluation, report |

---

## 📁 Repository Structure

```
├── multimodal_pipeline.ipynb   ← Main Jupyter notebook (all steps)
├── app.py                      ← CLI simulation app
├── data/
│   ├── customer_social_profiles.csv
│   ├── customer_transactions.csv
│   └── merged_dataset.csv
├── images/
│   ├── member1/  (neutral.jpg, smiling.jpg, surprised.jpg)
│   ├── member2/
│   ├── member3/
│   └── member4/
├── audio/
│   ├── member1/  (yes_approve.wav, confirm_transaction.wav)
│   ├── member2/
│   ├── member3/
│   └── member4/
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
    └── audio/
```

---

## ⚙️ Setup & Installation

```bash
pip install pandas numpy matplotlib seaborn scikit-learn opencv-python \
            Pillow librosa soundfile xgboost joblib scipy
```

---

## 🚀 How to Run

### Step 1: Add your data
- Place facial images in `images/memberX/` named `neutral.jpg`, `smiling.jpg`, `surprised.jpg`
- Place voice recordings in `audio/memberX/` named `yes_approve.wav`, `confirm_transaction.wav`
- Place datasets in `data/`

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
python app.py --image images/member1/neutral.jpg --audio audio/member1/yes_approve.wav

# Unauthorized simulation
python app.py --unauth
```

---

## 🔄 System Flow

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
[Product Recommendation]
    │
    ▼
  RESULT DISPLAYED
```

---

## 📊 Models Used

| Model | Algorithm | Purpose |
|-------|-----------|---------|
| Face Recognition | Random Forest | Identify which member from face image |
| Voiceprint Verification | Logistic Regression | Verify member identity from voice |
| Product Recommendation | XGBoost | Predict product based on customer profile |

---

## 📈 Evaluation Metrics
- Accuracy
- F1 Score (weighted)
- Confusion Matrix

---

## 🎥 Demo Video
[Link to demo video]
