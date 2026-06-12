import pandas as pd
import re
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score


print("Loading datasets...")

fake_df = pd.read_csv("Fake.csv")
real_df = pd.read_csv("True.csv")

fake_df["label"] = 0
real_df["label"] = 1

df = pd.concat([fake_df, real_df])

print("Dataset combined.")

print("\nClass distribution:")
print(df["label"].value_counts())


# -------------------------
# TEXT CLEANING
# -------------------------

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z ]", "", text)
    return text


df["text"] = df["title"] + " " + df["text"]
df["text"] = df["text"].apply(clean_text)


# -------------------------
# REMOVE DATASET LEAKAGE WORDS
# -------------------------

leak_words = ["reuters", "washington", "said"]

def remove_leak_words(text):
    for word in leak_words:
        text = text.replace(word, "")
    return text


df["text"] = df["text"].apply(remove_leak_words)


# -------------------------
# TRAIN TEST SPLIT
# -------------------------

X = df["text"]
y = df["label"]

print("\nSplitting dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# -------------------------
# TF-IDF VECTORIZATION
# -------------------------

print("\nVectorizing text with TF-IDF...")

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_df=0.75,
    min_df=5,
    ngram_range=(1,2)
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)


# -------------------------
# MODEL 1: LOGISTIC REGRESSION
# -------------------------

print("\nTraining Logistic Regression...")

lr_model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

lr_model.fit(X_train_vec, y_train)

lr_pred = lr_model.predict(X_test_vec)

lr_acc = accuracy_score(y_test, lr_pred)

print("\nLogistic Regression Results")
print("Accuracy:", lr_acc)
print(classification_report(y_test, lr_pred))


# -------------------------
# MODEL 2: NAIVE BAYES
# -------------------------

print("\nTraining Multinomial Naive Bayes...")

nb_model = MultinomialNB()

nb_model.fit(X_train_vec, y_train)

nb_pred = nb_model.predict(X_test_vec)

nb_acc = accuracy_score(y_test, nb_pred)

print("\nNaive Bayes Results")
print("Accuracy:", nb_acc)
print(classification_report(y_test, nb_pred))


# -------------------------
# MODEL COMPARISON
# -------------------------

print("\nModel Comparison")
print("---------------------------")
print("Logistic Regression Accuracy:", lr_acc)
print("Naive Bayes Accuracy:", nb_acc)


# -------------------------
# SAVE BOTH MODELS
# -------------------------

print("\nSaving trained models...")

joblib.dump(lr_model, "models/logistic_model.pkl")
joblib.dump(nb_model, "models/naive_model.pkl")
joblib.dump(vectorizer, "models/vectorizer.pkl")

print("\n✔ Logistic Regression model saved")
print("✔ Naive Bayes model saved")
print("✔ Vectorizer saved")

print("\nTraining pipeline completed successfully.")