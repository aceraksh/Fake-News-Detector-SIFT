# 📰 AI Fake News Credibility Detector

A multi-layer fake news detection system built with **Streamlit**, **FastAPI**, and **scikit-learn**. The system combines ML classification with rule-based credibility analysis to score news articles.

---

## 🚀 Features

- **ML Layer** — Logistic Regression trained on TF-IDF features (WELFake/Kaggle dataset)
- **Linguistic Layer** — Detects sensational/clickbait language patterns
- **Structural Layer** — Checks article length and structure
- **Entity Layer** — Identifies credible named entities and citations
- **Realtime Layer** — Placeholder for future API-based fact verification
- **Streamlit UI** — Clean interface with credibility score and verdict
- **FastAPI Backend** — REST endpoint for programmatic access

---

## 📁 Project Structure

```
fake-news-project/
├── app.py              # Streamlit frontend
├── predict.py          # Core multi-layer prediction logic
├── api.py              # FastAPI backend
├── train_model.py      # Model training pipeline
├── news_fetcher.py     # NewsAPI integration
├── load_model.py       # Model loading utility
├── models/             # Trained model files (see setup)
│   ├── logistic_model.pkl
│   ├── naive_model.pkl
│   └── vectorizer.pkl
├── requirements.txt
└── .gitignore
```

---

## ⚙️ Setup

### 1. Clone the repository
```bash
git clone https://github.com/your-username/fake-news-detector.git
cd fake-news-detector
```

### 2. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate      # Linux/Mac
venv\Scripts\activate         # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download trained models
The `.pkl` model files are too large for GitHub. Download them from the link below and place them in the `models/` folder:

> 📦 [Download models from Google Drive](#) *(replace with your actual link)*

### 5. Run the Streamlit app
```bash
streamlit run app.py
```

### 6. Run the FastAPI backend (optional)
```bash
uvicorn api:app --reload
```
API docs available at: `http://127.0.0.1:8000/docs`

---

## 📊 Scoring Logic

| Layer         | Weight | Description                          |
|---------------|--------|--------------------------------------|
| ML (real_prob)| 50%    | Logistic Regression prediction       |
| Linguistic    | 15%    | Sensational language detection       |
| Entity        | 15%    | Named entity / citation presence     |
| Realtime      | 10%    | API fact-check (placeholder)         |
| Structural    | 10%    | Article length analysis              |

**Verdict thresholds:**
- `< 0.35` → ⚠️ Highly Suspicious — Likely FAKE
- `0.35–0.50` → 🔶 Misleading — Needs Verification
- `0.50–0.65` → 🔷 Uncertain — Possibly Real
- `> 0.65` → ✅ Likely REAL NEWS

---

## 🏋️ Training Your Own Model

Download the [WELFake dataset](https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification) and place `Fake.csv` and `True.csv` in the root directory, then run:

```bash
python train_model.py
```

---

## 🔑 NewsAPI Key

To use `news_fetcher.py`, replace the `API_KEY` in `news_fetcher.py` with your own key from [newsapi.org](https://newsapi.org/).

---

## 📌 Tech Stack

- Python 3.10+
- scikit-learn
- Streamlit
- FastAPI + Uvicorn
- joblib
- pandas

---

## 👤 Author

**Rakshana** — [24BCS220, Kumaraguru College of Technology]
