import joblib
import re

# -------------------
# LOAD MODELS
# -------------------
lr_model = joblib.load("models/logistic_model.pkl")
nb_model = joblib.load("models/naive_model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")


# -------------------
# TEXT CLEANING
# -------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z ]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# -------------------
# ML LAYER
# -------------------
def ml_layer(text):
    text = clean_text(text)
    vect = vectorizer.transform([text])
    prob = lr_model.predict_proba(vect)[0]
    fake_prob = prob[0]
    real_prob = prob[1]
    return fake_prob, real_prob


# -------------------
# LINGUISTIC LAYER
# -------------------
def linguistic_layer(text):
    suspicious_words = [
        "shocking", "unbelievable", "breaking", "click here",
        "you won't believe", "miracle", "secret", "hoax",
        "conspiracy", "exposed", "mainstream media", "they don't want you to know"
    ]
    text_lower = text.lower()
    word_count = max(len(text.split()), 1)
    hits = sum(1 for w in suspicious_words if w in text_lower)

    # Normalize by text length — short texts with 1 hit shouldn't be penalized as much
    hit_ratio = hits / (word_count / 10)

    if hit_ratio >= 0.5:
        return 0.2
    elif hit_ratio >= 0.2:
        return 0.5
    else:
        return 0.8


# -------------------
# STRUCTURAL LAYER
# -------------------
def structural_layer(text):
    words = text.split()
    word_count = len(words)

    if word_count < 10:
        struct_score = 0.2
    elif word_count < 30:
        struct_score = 0.5
    else:
        struct_score = 0.8

    return struct_score, word_count


# -------------------
# ENTITY LAYER
# -------------------
def entity_layer(text):
    """
    Checks for named-entity-like signals in text.
    Returns 0.8 if strong entities found, 0.5 if moderate, 0.6 as neutral default
    (instead of punishing 0.2 for missing entities — most real news may not mention
    government/india but can still be legitimate).
    """
    strong_entities = [
        "india", "government", "minister", "police", "university",
        "company", "president", "court", "parliament", "cbi", "rbi",
        "supreme court", "hospital", "army", "officer"
    ]
    moderate_entities = [
        "said", "according", "reported", "official", "source",
        "announced", "confirmed", "stated", "study", "research"
    ]

    text_lower = text.lower()

    strong_hits = sum(1 for e in strong_entities if e in text_lower)
    moderate_hits = sum(1 for e in moderate_entities if e in text_lower)

    if strong_hits >= 2 or (strong_hits >= 1 and moderate_hits >= 1):
        return 0.85
    elif strong_hits == 1 or moderate_hits >= 2:
        return 0.75
    elif moderate_hits == 1:
        return 0.6
    else:
        # Neutral — don't penalize just for missing entities
        return 0.5


# -------------------
# REALTIME LAYER
# -------------------
def realtime_layer(text):
    # Placeholder for future API-based verification
    return 0.5


# -------------------
# FINAL SCORE
# -------------------
def compute_final_score(fake_prob, ling_score, struct_score, entity_score, realtime_score, word_count):
    real_prob = 1 - fake_prob

    score = (
        real_prob      * 0.50 +
        ling_score     * 0.15 +
        struct_score   * 0.10 +
        entity_score   * 0.15 +
        realtime_score * 0.10
    )

    return round(score, 4)


# -------------------
# VERDICT
# -------------------
def get_verdict(score, fake_prob, word_count, entity_score):
    reasons = []

    if fake_prob > 0.75:
        reasons.append("ML model strongly predicts FAKE news")
    elif fake_prob > 0.55:
        reasons.append("ML model leans toward FAKE news")

    if word_count < 10:
        reasons.append("News article is very short — insufficient content to analyze")
    elif word_count < 30:
        reasons.append("News article is relatively short")

    if entity_score < 0.5:
        reasons.append("No named entities or credible references detected")
    elif entity_score >= 0.75:
        reasons.append("Credible named entities detected")

    if score < 0.35:
        verdict = "⚠️ Highly Suspicious — Likely FAKE"
    elif score < 0.50:
        verdict = "🔶 Misleading — Needs Verification"
    elif score < 0.65:
        verdict = "🔷 Uncertain — Possibly Real"
    else:
        verdict = "✅ Likely REAL NEWS"

    if not reasons:
        reasons.append("No strong indicators of fake or real content found")

    return verdict, reasons
