-- ======================================================
-- FAKE NEWS DETECTOR — Database Init Script
-- Auto-runs when PostgreSQL container starts for first time
-- ======================================================

-- Create a separate Airflow database
CREATE DATABASE airflow;

-- Switch to fakenews DB (already created via POSTGRES_DB env var)
\connect fakenews;

-- Predictions table
CREATE TABLE IF NOT EXISTS predictions (
    id            SERIAL PRIMARY KEY,
    text          TEXT,
    fake_prob     FLOAT,
    real_prob     FLOAT,
    final_score   FLOAT,
    verdict       TEXT,
    word_count    INT,
    created_at    TIMESTAMP DEFAULT NOW()
);

-- Retrain tracking table
CREATE TABLE IF NOT EXISTS retrain_log (
    id            SERIAL PRIMARY KEY,
    run_at        TIMESTAMP DEFAULT NOW(),
    accuracy      FLOAT,
    samples_used  INT,
    notes         TEXT
);

-- Index for fast time-based queries on dashboard
CREATE INDEX IF NOT EXISTS idx_predictions_created_at ON predictions(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_predictions_verdict ON predictions(verdict);
