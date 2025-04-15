import csv
import pickle
import random
import re
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def clean_text(text):
    """
    Remove extra punctuation, quotes, etc.
    Simple approach:
      - Lowercase
      - Remove non-alphanumeric except spaces
    """
    text = text.lower()
    # remove quotes, punctuation except spaces
    text = re.sub(r"[^a-z0-9\s]+", "", text)
    return text.strip()

def balance_dataset(texts, labels):
    """
    Ensures all labels have the same number of training examples
    by duplicating minority samples.
    """
    label_counts = Counter(labels)
    max_samples = max(label_counts.values())

    new_texts, new_labels = [], []
    for label in label_counts:
        label_texts = [t for (t, l) in zip(texts, labels) if l == label]
        while len(label_texts) < max_samples:
            label_texts.append(random.choice(label_texts))
        new_texts.extend(label_texts)
        new_labels.extend([label] * len(label_texts))

    return new_texts, new_labels

def train_sound_effect_model(csv_path="train/holmes_excerpt_dataset.csv",
                             model_path="train/sound_effect_model.pkl"):
    texts = []
    labels = []

    # 1. Load CSV data
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_text = row["Text"]
            label = row["Effect"].strip().lower()
            # Clean the text
            cleaned = clean_text(raw_text)
            texts.append(cleaned)
            labels.append(label)

    # 2. Show label distribution before balancing
    print("Label Distribution Before:", Counter(labels))

    # 3. Balance dataset
    texts, labels = balance_dataset(texts, labels)

    # 4. Show label distribution after balancing
    print("Label Distribution After:", Counter(labels))

    # 5. Train-test split
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=0.2, random_state=42, stratify=labels
        )
    except ValueError:
        print("⚠ Not enough samples per label to stratify. Using random split.")
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=0.2, random_state=42
        )

    # 6. Pipeline: TF-IDF (word-level) + Logistic Regression
    #    Use a simpler ngram_range=(1,2).
    model_pipeline = Pipeline([
        ("vectorizer", TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            token_pattern=r"\b[a-z0-9]+\b"  # ensures we only pick up alphanumeric tokens
        )),
        ("classifier", LogisticRegression(
            max_iter=5000,
            solver='saga',
            class_weight='balanced'
        ))
    ])

    # 7. Train
    model_pipeline.fit(X_train, y_train)

    # 8. Evaluate
    y_pred = model_pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {acc:.2f}")

    with open(model_path, "wb") as out_f:
        pickle.dump(model_pipeline, out_f)
    print(f"✅ Trained model saved to {model_path}")

if __name__ == "__main__":
    train_sound_effect_model()
