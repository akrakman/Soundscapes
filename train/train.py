import csv
import pickle
import random
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def balance_dataset(texts, labels):
    """
    Ensures all labels have the same number of training examples.
    If a label is underrepresented, existing examples are duplicated.
    """
    label_counts = Counter(labels)
    max_samples = max(label_counts.values())  # Get the largest label count

    new_texts, new_labels = [], []
    for label in label_counts.keys():
        label_texts = [t for t, l in zip(texts, labels) if l == label]
        
        while len(label_texts) < max_samples:
            label_texts.append(random.choice(label_texts))  # Randomly duplicate data
            
        new_texts.extend(label_texts)
        new_labels.extend([label] * len(label_texts))

    return new_texts, new_labels

def train_sound_effect_model(csv_path: str = "holmes_excerpt_dataset.csv", model_path: str = "sound_effect_model.pkl") -> None:
    """
    Trains a Logistic Regression model on the Holmes excerpt dataset.
    Uses TF-IDF vectorization, balanced dataset, and class weighting.
    Saves the trained model pipeline to 'model_path'.
    """
    texts = []
    labels = []

    # 1. Load CSV data
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            texts.append(row["Text"].strip())
            labels.append(row["Effect"].strip().lower())  # Ensure lowercase labels

    # 2. Check label distribution
    label_counts = Counter(labels)
    print("\nLabel Distribution Before Fix:", label_counts)

    # 3. Balance dataset
    texts, labels = balance_dataset(texts, labels)

    # 4. Verify new label distribution
    label_counts = Counter(labels)
    print("\nLabel Distribution After Fix:", label_counts)

    # 5. Train-test split (Use stratify only if all labels have 2+ occurrences)
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=0.2, random_state=42, stratify=labels
        )
    except ValueError:
        print("⚠️ Not enough samples per label! Using random split instead.")
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=0.2, random_state=42
        )

    # 6. Create a pipeline: (TF-IDF + Logistic Regression)
    model_pipeline = Pipeline([
        ("vectorizer", TfidfVectorizer(ngram_range=(1, 3), analyzer="char_wb")),  # Use both word & character n-grams
        ("classifier", LogisticRegression(max_iter=5000, solver='saga', class_weight='balanced'))  # Robust solver
    ])

    # 7. Train the model
    model_pipeline.fit(X_train, y_train)

    # 8. Evaluate on test set
    y_pred = model_pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nModel accuracy: {acc:.2f}")

    # 9. Save the trained pipeline to disk
    with open(model_path, "wb") as out_f:
        pickle.dump(model_pipeline, out_f)
    print(f"✅ Trained model saved to {model_path}")

if __name__ == "__main__":
    # Train the model using only Holmes excerpt data
    train_sound_effect_model("holmes_excerpt_dataset.csv", "sound_effect_model.pkl")
