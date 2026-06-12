from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import functions from predict.py
from predict import (
    ml_layer,
    linguistic_layer,
    structural_layer,
    entity_layer,
    realtime_layer,
    compute_final_score,
    get_verdict
)

# ======================================================
# FASTAPI APP
# ======================================================

app = FastAPI(
    title="Fake News Detection API",
    description="AI system to detect fake news using ML + credibility layers",
    version="1.0"
)

# ======================================================
# CORS
# ======================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ======================================================
# INPUT SCHEMA
# ======================================================

class NewsInput(BaseModel):
    text: str

# ======================================================
# HEALTH CHECK
# ======================================================

@app.get("/")
def home():
    return {"message": "Fake News Detection API is running"}

# ======================================================
# PREDICTION ENDPOINT
# ======================================================

@app.post("/predict")
def predict(news: NewsInput):

    # ML Layer
    fake_prob, real_prob = ml_layer(news.text)

    # Other layers
    ling_score = linguistic_layer(news.text)

    struct_score, word_count = structural_layer(news.text)

    entity_score = entity_layer(news.text)

    realtime_score = realtime_layer(news.text)

    # Final score
    final_score = compute_final_score(
        fake_prob,
        ling_score,
        struct_score,
        entity_score,
        realtime_score,
        word_count
    )

    # Verdict + reasoning
    verdict, reasons = get_verdict(
        final_score,
        fake_prob,
        word_count,
        entity_score
    )

    return {
        "verdict": verdict,
        "fake_probability": round(fake_prob, 3),
        "real_probability": round(real_prob, 3),
        "final_score": round(final_score, 3),
        "word_count": word_count,
        "reasons": reasons
    }