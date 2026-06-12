"""
retrain_dag.py
--------------
Weekly pipeline for the Fake News Detector.

Schedule:  Every Sunday at midnight
Tasks:
  1. check_data    — verify enough new data exists to retrain
  2. retrain_model — run train_model.py, log metrics to MLflow + DB
  3. notify        — print summary (extend to email/Slack later)
"""

from airflow import DAG
from airflow.operators.python import PythonOperator, ShortCircuitOperator
from datetime import datetime, timedelta
import os
import sys

# -------------------------------------------------------
# Project root is mounted at /opt/airflow/project
# -------------------------------------------------------
PROJECT_ROOT = "/opt/airflow/project"
sys.path.insert(0, PROJECT_ROOT)

# -------------------------------------------------------
# DEFAULT ARGS
# -------------------------------------------------------
default_args = {
    "owner": "rakshana",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
}

# -------------------------------------------------------
# TASK FUNCTIONS
# -------------------------------------------------------

def check_data_volume(**context):
    """
    Returns True if there are >= 50 new predictions in the DB since last retrain.
    If not enough data, short-circuits the DAG (skips retrain).
    """
    import psycopg2

    conn = psycopg2.connect(
        host="postgres", port=5432,
        dbname="fakenews", user="postgres", password="fakenews123"
    )
    cur = conn.cursor()

    # Get last retrain time
    cur.execute("SELECT MAX(run_at) FROM retrain_log")
    last_run = cur.fetchone()[0]

    if last_run is None:
        last_run = datetime(2000, 1, 1)

    # Count new predictions since last retrain
    cur.execute(
        "SELECT COUNT(*) FROM predictions WHERE created_at > %s",
        (last_run,)
    )
    new_count = cur.fetchone()[0]
    cur.close()
    conn.close()

    print(f"[check_data] New predictions since last retrain: {new_count}")
    context["ti"].xcom_push(key="new_count", value=new_count)

    return new_count >= 50   # ShortCircuitOperator: False = skip downstream


def retrain_model(**context):
    """
    Retrains the model and logs metrics to MLflow + retrain_log table.
    """
    import subprocess
    import mlflow
    import psycopg2
    from datetime import datetime

    os.chdir(PROJECT_ROOT)

    print("[retrain] Starting model retraining...")

    # Run the training script as subprocess
    result = subprocess.run(
        ["python", os.path.join(PROJECT_ROOT, "train_model.py")],
        capture_output=True, text=True
    )

    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        raise Exception("train_model.py failed. Check logs above.")

    # Parse accuracy from stdout (train_model.py prints "Accuracy: X.XX")
    accuracy = None
    for line in result.stdout.splitlines():
        if "Logistic Regression Accuracy:" in line:
            try:
                accuracy = float(line.split(":")[-1].strip())
            except ValueError:
                pass

    new_count = context["ti"].xcom_pull(key="new_count")

    # Log to MLflow
    mlflow.set_tracking_uri(f"file://{PROJECT_ROOT}/mlruns")
    mlflow.set_experiment("fake_news_detector")
    with mlflow.start_run(run_name=f"weekly_retrain_{datetime.now().strftime('%Y%m%d')}"):
        if accuracy:
            mlflow.log_metric("accuracy", accuracy)
        mlflow.log_param("trigger", "airflow_weekly")
        mlflow.log_param("new_samples", new_count or 0)

    # Log to DB
    conn = psycopg2.connect(
        host="postgres", port=5432,
        dbname="fakenews", user="postgres", password="fakenews123"
    )
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO retrain_log (run_at, accuracy, samples_used, notes)
        VALUES (%s, %s, %s, %s)
    """, (
        datetime.now(),
        accuracy or 0.0,
        new_count or 0,
        "Scheduled weekly retrain via Airflow"
    ))
    conn.commit()
    cur.close()
    conn.close()

    print(f"[retrain] ✔ Retrain complete. Accuracy: {accuracy}")
    context["ti"].xcom_push(key="accuracy", value=accuracy)


def notify(**context):
    """
    Prints a summary. Extend this to send email or Slack notification.
    """
    accuracy = context["ti"].xcom_pull(task_ids="retrain_model", key="accuracy")
    new_count = context["ti"].xcom_pull(task_ids="check_data", key="new_count")

    print("=" * 50)
    print("  FAKE NEWS DETECTOR — Weekly Retrain Summary")
    print("=" * 50)
    print(f"  New samples used : {new_count}")
    print(f"  New accuracy     : {accuracy}")
    print(f"  Status           : ✔ Success")
    print("=" * 50)


# -------------------------------------------------------
# DAG DEFINITION
# -------------------------------------------------------

with DAG(
    dag_id="fake_news_weekly_retrain",
    description="Weekly retrain pipeline for the Fake News Detector",
    schedule_interval="0 0 * * 0",    # Every Sunday midnight
    start_date=datetime(2024, 1, 1),
    catchup=False,
    default_args=default_args,
    tags=["fake-news", "ml", "retrain"],
) as dag:

    check_data = ShortCircuitOperator(
        task_id="check_data",
        python_callable=check_data_volume,
        provide_context=True,
    )

    retrain = PythonOperator(
        task_id="retrain_model",
        python_callable=retrain_model,
        provide_context=True,
    )

    notify_task = PythonOperator(
        task_id="notify",
        python_callable=notify,
        provide_context=True,
    )

    # Pipeline order
    check_data >> retrain >> notify_task
