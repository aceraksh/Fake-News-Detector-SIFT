🔍 Sift — Separate Signal from Noise

A multi-layer fake news detection system. Sift combines a trained machine learning
classifier with rule-based credibility checks to score how trustworthy a piece of
news text is, and explains why in plain language.


🚀 Features


ML Layer — Logistic Regression trained on TF-IDF features (WELFake dataset, 44,919 articles)
Linguistic Layer — Flags sensational / clickbait language patterns
Structural Layer — Checks article length and structure
Entity Layer — Looks for credible named entities and sourcing language
Realtime Layer — Placeholder for future live fact-check API
Sift UI (sift.html) — Standalone web interface with a credibility gauge, signal/noise breakdown, and reasoning panel
FastAPI Backend (api.py) — REST endpoint (/predict) for programmatic access
Streamlit App (app.py, optional) — Alternative UI with a PostgreSQL-backed analytics dashboard



📁 Project Structure

sift/
├── sift.html            # Standalone frontend (Scanner, Activity, Methodology)
├── app.py                # Streamlit frontend (optional, needs Postgres)
├── predict.py            # Core multi-layer scoring logic
├── api.py                 # FastAPI backend serving /predict
├── train_model.py         # Model training pipeline
├── load_model.py           # Model loading utility
├── news_fetcher.py          # NewsAPI integration (requires your own API key)
├── db.py                      # PostgreSQL helpers for the Streamlit dashboard
├── init_db.sql                 # DB schema for predictions + retrain logs
├── docker-compose.yml            # Postgres + pgAdmin + Airflow stack (optional)
├── retrain_dag.py                 # Weekly Airflow retraining DAG (optional)
├── models/                          # Trained model files (generated, not committed)
│   ├── logistic_model.pkl
│   ├── naive_model.pkl
│   └── vectorizer.pkl
├── requirements.txt
└── .gitignore


⚙️ Setup

1. Clone the repository

bashgit clone https://github.com/<your-username>/sift.git
cd sift

2. Create a virtual environment

bashpython -m venv venv
source venv/bin/activate      # Linux/Mac
venv\Scripts\activate         # Windows

3. Install dependencies

bashpip install -r requirements.txt

4. Get the training data

Download Fake.csv and True.csv from the
WELFake dataset on Kaggle
and place them in the project root. These are not committed to the repo
(see .gitignore).

5. Train the model

bashmkdir models
python train_model.py

This generates models/logistic_model.pkl, models/naive_model.pkl, and
models/vectorizer.pkl. These are also gitignored — too large for GitHub.

6. Run the FastAPI backend

bashuvicorn api:app --port 8000

API docs available at http://127.0.0.1:8000/docs

7. Open the Sift UI

Open sift.html directly in a browser, or serve it:

bashpython -m http.server 5500

then visit http://localhost:5500/sift.html.


📊 Scoring Logic

LayerWeightDescriptionML (real_prob)50%Logistic Regression predictionLinguistic15%Sensational language detectionEntity15%Named entity / citation presenceStructural10%Article length analysisRealtime10%Live fact-check (placeholder)

Verdict thresholds:


< 0.35 → ⚠️ Highly Suspicious — Likely FAKE
0.35–0.50 → 🔶 Misleading — Needs Verification
0.50–0.65 → 🔷 Uncertain — Possibly Real
> 0.65 → ✅ Likely REAL NEWS



🔑 NewsAPI Key

news_fetcher.py requires a NewsAPI key. Do not hardcode
it — set it as an environment variable instead:

bashexport NEWSAPI_KEY="9574ead008e44845ba97d56d2686ee3c"     # Linux/Mac
$env:NEWSAPI_KEY="9574ead008e44845ba97d56d2686ee3c"       # Windows PowerShell

and load it in news_fetcher.py with os.environ["NEWSAPI_KEY"].


📌 Tech Stack


Python 3.10+
scikit-learn, pandas, joblib
FastAPI + Uvicorn
HTML / Tailwind CSS / vanilla JS (Sift UI)
Streamlit (optional dashboard)
PostgreSQL + Airflow (optional data pipeline)



👤 Author

Rakshana — 24BCS220, Kumaraguru College of Technology
