"""Train the spam model and save it to model.joblib.

Run:  python train.py
"""
from pathlib import Path

import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.pipeline import FeatureUnion, Pipeline

from text_utils import MASK_TOKENS, clean_text

HERE = Path(__file__).parent
DATA_FILE = HERE / "data" / "messages.csv"
OUTSIDE_FILE = HERE / "data" / "handwritten_test.csv"
MODEL_FILE = HERE / "model.joblib"

INDIAN_LANGUAGES = {"hinglish": "Indian Hinglish", "en": "Indian English", "hi": "Hindi (Devanagari)"}


def load_data(path):
    # keep_default_na: a message that just says "NA" or "null" is text, not a missing value
    df = pd.read_csv(path, keep_default_na=False)
    df["target"] = (df["label"] == "spam").astype(int)
    return df


def group_name(row):
    if row["source"] == "uci_sms":
        return "English SMS (UK, Singapore)"
    if row["source"] == "hinglish_chat":
        return "Hinglish everyday chat"
    return INDIAN_LANGUAGES.get(row["language"], "Other Indian code-mixed")


def build_model(classifier):
    # Whole words and pairs of words carry the meaning. Pieces of words (2 to 5
    # letters) let the model match Hinglish however it is spelled: "karein",
    # "kare" and "karo" all share "kar".
    features = FeatureUnion([
        ("words", TfidfVectorizer(preprocessor=clean_text, token_pattern=r"\S+", ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
        ("pieces", TfidfVectorizer(preprocessor=clean_text, analyzer="char_wb", ngram_range=(2, 5), min_df=3, sublinear_tf=True)),
    ])
    return Pipeline([("features", features), ("classifier", classifier)])


def print_row(name, y, predicted):
    flagged = int(((predicted == 1) & (y == 0)).sum())
    missed = int(((predicted == 0) & (y == 1)).sum())
    # Precision and recall mean nothing for a group with no spam in it
    precision = f"{precision_score(y, predicted, zero_division=0):.3f}" if y.sum() else "-"
    recall = f"{recall_score(y, predicted):.3f}" if y.sum() else "-"
    print(f"  {name:28} {len(y):>8} {precision:>9} {recall:>7} {flagged:>4} of {(y == 0).sum():<5} {missed:>3} of {y.sum():<4}")


def print_groups(model, df, groups):
    print(f"  {'':28} {'messages':>8} {'precision':>9} {'recall':>7} {'real flagged':>13} {'spam missed':>11}")
    predicted = pd.Series(model.predict(df["text"]), index=df.index)
    for name, group in df.groupby(groups, sort=True):
        print_row(name, group["target"], predicted[group.index])


def unseen_scam_types(df, classifier):
    """How well does the model catch a kind of scam it has never seen?

    For each scam topic, train on everything except that topic and test on that
    topic alone. The ordinary test set cannot answer this, because it shares its
    topics (and its writers) with the training set.
    """
    print(f"  {'':28} {'messages':>8} {'precision':>9} {'recall':>7} {'real flagged':>13} {'spam missed':>11}")
    indian = df[(df["source"] == "indian_scam_sms") & (df["topic"] != "everyday")]
    for topic, held_out in indian.groupby("topic"):
        rest = df.drop(held_out.index)
        model = build_model(clone(classifier)).fit(rest["text"], rest["target"])
        print_row(topic, held_out["target"], model.predict(held_out["text"]))


def main():
    df = load_data(DATA_FILE)
    df["group"] = df.apply(group_name, axis=1)
    print(f"{len(df)} messages, {df['target'].sum()} spam ({df['target'].mean():.1%})")

    # Stratify so every source keeps its share of spam and ham in both halves
    train, test = train_test_split(
        df, test_size=0.2, random_state=2, stratify=df["source"] + df["label"]
    )

    classifiers = {"BernoulliNB": BernoulliNB(), "LogisticRegression": LogisticRegression(C=10, max_iter=3000)}
    models, scores = {}, {}
    for name, classifier in classifiers.items():
        # The vectorizers sit inside the model, so they learn their vocabulary
        # from the training messages only and nothing leaks from the test set.
        models[name] = build_model(classifier).fit(train["text"], train["target"])
        predicted = models[name].predict(test["text"])
        precision = precision_score(test["target"], predicted)
        recall = recall_score(test["target"], predicted)
        scores[name] = (precision, recall)
        print(f"\n{name}")
        print(f"  accuracy  {accuracy_score(test['target'], predicted):.3f}")
        print(f"  precision {precision:.3f}")
        print(f"  recall    {recall:.3f}")
        print(f"  confusion matrix [[ham ok, ham flagged], [spam missed, spam caught]]")
        print(f"  {confusion_matrix(test['target'], predicted).tolist()}")

    # Precision first (never flag a real message), then recall (catch more spam)
    best = max(scores, key=scores.get)

    print(f"\n{best} on the test set, by kind of message")
    print_groups(models[best], test, "group")

    outside = load_data(OUTSIDE_FILE)
    print(f"\n{best} on hand-written Indian messages it never trained on")
    print_groups(models[best], outside.assign(group="Hand-written (outside check)"), "group")

    print(f"\n{best} on a scam type left out of training")
    unseen_scam_types(df, classifiers[best])

    words = models[best].named_steps["features"].transformer_list[0][1].vocabulary_
    known_words = {word for word in words if " " not in word} - MASK_TOKENS
    joblib.dump({"model": models[best], "known_words": known_words, "name": best}, MODEL_FILE)
    print(f"\nSaved {best} to {MODEL_FILE.name}")


if __name__ == "__main__":
    main()
