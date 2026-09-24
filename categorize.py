"""
Auto-categorization model: classifies a transaction into a spending category
based purely on the raw (noisy) merchant description string.

Pipeline: character n-gram TF-IDF -> Linear SVM (calibrated for probabilities).
Character n-grams (not word n-grams) are used deliberately: bank merchant
strings are full of truncations, codes, and inconsistent spacing
("AMZN MKTP US*2K3"), so sub-word patterns generalize far better than
whole-word tokens.
"""
import re
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

MODEL_PATH = "data/categorizer.joblib"


def clean_text(s: str) -> str:
    s = s.upper()
    s = re.sub(r"[^A-Z ]", " ", s)   # strip digits/codes/punctuation noise
    s = re.sub(r"\s+", " ", s).strip()
    return s


def train(df: pd.DataFrame, test_size=0.2, random_state=42, model_path=MODEL_PATH):
    X = df["description"].apply(clean_text)
    y = df["category"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=1)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    base_clf = LinearSVC(class_weight="balanced", random_state=random_state)
    clf = CalibratedClassifierCV(base_clf, cv=3)
    clf.fit(X_train_vec, y_train)

    preds = clf.predict(X_test_vec)
    acc = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds, zero_division=0)

    joblib.dump({"vectorizer": vectorizer, "clf": clf}, model_path)
    return acc, report


def load_model(model_path=MODEL_PATH):
    return joblib.load(model_path)


def predict(descriptions, model=None):
    """Predict category + confidence for a list of raw descriptions."""
    if model is None:
        model = load_model()
    vectorizer, clf = model["vectorizer"], model["clf"]
    cleaned = [clean_text(d) for d in descriptions]
    X = vectorizer.transform(cleaned)
    preds = clf.predict(X)
    probs = clf.predict_proba(X).max(axis=1)
    return list(zip(preds, probs))


if __name__ == "__main__":
    df = pd.read_csv("data/transactions.csv")
    acc, report = train(df)
    print(f"Test accuracy: {acc:.3f}\n")
    print(report)

    # quick sanity check on a few unseen-style descriptions
    samples = ["SWIGGY BLR ORDER#4471", "AMZN MKTP INDIA", "UBER TRIP REF9981", "NETFLIX SUBSCRIP"]
    for desc, (cat, conf) in zip(samples, predict(samples)):
        print(f"{desc:30s} -> {cat:20s} (confidence {conf:.2f})")
