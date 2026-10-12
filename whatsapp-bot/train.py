"""Train the spam model and save it to model.joblib.

Run:  python train.py
"""
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB, MultinomialNB

from text_utils import transform_text

HERE = Path(__file__).parent
DATA_FILE = HERE.parent / "email-spam-classifier" / "spam.csv"
MODEL_FILE = HERE / "model.joblib"


def load_data():
    df = pd.read_csv(DATA_FILE, encoding="latin1")[["v1", "v2"]]
    df.columns = ["label", "text"]
    df = df.drop_duplicates()
    df["target"] = (df["label"] == "spam").astype(int)
    df["clean"] = df["text"].apply(transform_text)
    return df


def main():
    df = load_data()
    print(f"{len(df)} messages, {df['target'].sum()} spam ({df['target'].mean():.1%})")

    train_text, test_text, y_train, y_test = train_test_split(
        df["clean"], df["target"], test_size=0.2, random_state=2
    )

    # The vectorizer learns its vocabulary from the training messages only.
    # The notebook fitted it on all messages before splitting, which lets
    # information about the test set leak into training.
    vectorizer = TfidfVectorizer(max_features=3000)
    X_train = vectorizer.fit_transform(train_text)
    X_test = vectorizer.transform(test_text)

    models = {"MultinomialNB": MultinomialNB(), "BernoulliNB": BernoulliNB()}
    scores = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        predicted = model.predict(X_test)
        precision = precision_score(y_test, predicted)
        recall = recall_score(y_test, predicted)
        scores[name] = (precision, recall)
        print(f"\n{name}")
        print(f"  accuracy  {accuracy_score(y_test, predicted):.3f}")
        print(f"  precision {precision:.3f}")
        print(f"  recall    {recall:.3f}")
        print(f"  confusion matrix [[ham ok, ham flagged], [spam missed, spam caught]]")
        print(f"  {confusion_matrix(y_test, predicted).tolist()}")

    # Precision first (never flag a real message), then recall (catch more spam)
    best = max(scores, key=scores.get)
    joblib.dump({"vectorizer": vectorizer, "model": models[best], "name": best}, MODEL_FILE)
    print(f"\nSaved {best} to {MODEL_FILE.name}")


if __name__ == "__main__":
    main()
