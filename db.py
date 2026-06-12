import psycopg2
import pandas as pd
from datetime import datetime

# ======================================================
# CONNECTION
# ======================================================

def get_connection():
    return psycopg2.connect(
        host="localhost",
        port=5432,
        dbname="fakenews",
        user="postgres",
        password="fakenews123"
    )


# ======================================================
# SETUP — run once to create tables
# ======================================================

def init_db():
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
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
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS retrain_log (
            id            SERIAL PRIMARY KEY,
            run_at        TIMESTAMP DEFAULT NOW(),
            accuracy      FLOAT,
            samples_used  INT,
            notes         TEXT
        );
    """)

    conn.commit()
    cur.close()
    conn.close()
    print("✔ Database initialized.")


# ======================================================
# LOG A PREDICTION
# ======================================================

def log_prediction(text, fake_prob, real_prob, final_score, verdict, word_count):
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO predictions (text, fake_prob, real_prob, final_score, verdict, word_count, created_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """, (
            text[:500],         # cap at 500 chars
            float(fake_prob),
            float(real_prob),
            float(final_score),
            verdict,
            int(word_count),
            datetime.now()
        ))
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        print(f"[db] log_prediction failed: {e}")


# ======================================================
# FETCH STATS FOR DASHBOARD
# ======================================================

def get_verdict_counts():
    conn = get_connection()
    df = pd.read_sql("""
        SELECT verdict, COUNT(*) AS count
        FROM predictions
        GROUP BY verdict
        ORDER BY count DESC
    """, conn)
    conn.close()
    return df


def get_hourly_trend():
    conn = get_connection()
    df = pd.read_sql("""
        SELECT
            DATE_TRUNC('hour', created_at) AS hour,
            COUNT(*) AS total,
            SUM(CASE WHEN verdict LIKE '%FAKE%' THEN 1 ELSE 0 END) AS fake_count
        FROM predictions
        WHERE created_at >= NOW() - INTERVAL '24 hours'
        GROUP BY hour
        ORDER BY hour
    """, conn)
    conn.close()
    return df


def get_recent_predictions(limit=20):
    conn = get_connection()
    df = pd.read_sql(f"""
        SELECT id, LEFT(text, 80) AS preview, fake_prob, final_score, verdict, created_at
        FROM predictions
        ORDER BY created_at DESC
        LIMIT {limit}
    """, conn)
    conn.close()
    return df


def get_total_analyzed():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM predictions")
    count = cur.fetchone()[0]
    cur.close()
    conn.close()
    return count


# ======================================================
# LOG A RETRAIN RUN
# ======================================================

def log_retrain(accuracy, samples_used, notes=""):
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO retrain_log (run_at, accuracy, samples_used, notes)
            VALUES (%s, %s, %s, %s)
        """, (datetime.now(), float(accuracy), int(samples_used), notes))
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        print(f"[db] log_retrain failed: {e}")


def get_retrain_history():
    conn = get_connection()
    df = pd.read_sql("""
        SELECT run_at, accuracy, samples_used, notes
        FROM retrain_log
        ORDER BY run_at DESC
    """, conn)
    conn.close()
    return df


# ======================================================
# ENTRYPOINT — init tables when run directly
# ======================================================

if __name__ == "__main__":
    init_db()
