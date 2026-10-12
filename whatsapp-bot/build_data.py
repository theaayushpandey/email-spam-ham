"""Download the Indian and Hinglish datasets and build the training file.

Run:  python build_data.py

Writes data/messages.csv, which train.py trains and tests on, and
data/handwritten_test.csv, which is never trained on and is only used as an
outside check. Downloads are kept in data/raw/ and reused on the next run.
"""
import csv
import html
import io
import json
import re
import urllib.request
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
DATA = HERE / "data"
RAW = DATA / "raw"
UCI_FILE = HERE.parent / "email-spam-classifier" / "spam.csv"

# Each file is pinned to one commit, so a rebuild gives the same data even if
# the author changes or deletes the dataset later.
URL = "https://huggingface.co/datasets/{repo}/resolve/{commit}/{path}"
SOURCES = {
    "indian_scam_sms.csv": {
        "repo": "Ridham115/indian-scam-sms-synthetic-audited",
        "commit": "45e92c38ba20e973ef3ff5d1893a0a710f622018",
        "path": "data/synthetic_audited.csv",
    },
    "hinglish_chat.csv": {
        "repo": "Abhishekcr448/Hinglish-Everyday-Conversations-1M",
        "commit": "cf040b1b80857c6038eb73ddad441f0b435e02a6",
        "path": "hinglish_conversations.csv",
        # The file is 185 MB. The first 1.5 MB holds far more lines than we sample.
        "first_bytes": 1_500_000,
    },
    "handwritten.json": {
        "repo": "bolewara/hinglish-scam-text-dataset",
        "commit": "743799229923fc80fd7e9f2c764b5e9bd4ab3c9a",
        "path": "data.json",
    },
}
CHAT_SAMPLE = 1200
COLUMNS = ["text", "label", "language", "source", "topic"]


def download(name):
    target = RAW / name
    if target.exists():
        return target
    source = SOURCES[name]
    request = urllib.request.Request(URL.format(**source))
    if "first_bytes" in source:
        request.add_header("Range", f"bytes=0-{source['first_bytes'] - 1}")
    print(f"Downloading {source['repo']}")
    with urllib.request.urlopen(request, timeout=120) as response:
        target.write_bytes(response.read())
    return target


def letters_only(text):
    """A message with everything but letters and digits removed, for spotting copies."""
    return re.sub(r"[^a-z0-9]", "", str(text).lower())


def unescape(text):
    """Undo the HTML escaping in the UCI messages. Many real messages there had
    their numbers replaced with "<#>" (stored as "&lt;#&gt;"): make it a number again."""
    return html.unescape(text).replace("<#>", "0")


def load_uci():
    """The original English SMS data (UK and Singapore, collected before 2012)."""
    df = pd.read_csv(UCI_FILE, encoding="latin1")
    # About 50 messages contain a comma the file does not quote properly, so the
    # rest of the message lands in the extra columns. Join it back.
    text = df.iloc[:, 1:].fillna("").agg(",".join, axis=1).str.rstrip(",")
    # The file's encoding is damaged: the pound sign reads as "å£" and other
    # symbols as stray accented letters. Left in, "å" becomes a spam word.
    text = text.str.replace("å£", "£", regex=False)
    text = text.str.replace(r"[\x80-\xa2\xa4-\xff]", "", regex=True)
    df = pd.DataFrame({"label": df["v1"], "text": text.map(unescape)}).drop_duplicates()
    return df.assign(language="en", source="uci_sms", topic="")


def load_indian_scam_sms():
    """Scam SMS and their genuine look-alikes in Indian English, Hinglish, Hindi
    and four other code-mixed styles. Written by language models, not real traffic."""
    df = pd.read_csv(download("indian_scam_sms.csv"))
    df["label"] = df["label"].map({"scam": "spam", "safe": "ham"})
    return df.assign(source="indian_scam_sms")


def load_hinglish_chat():
    """Ordinary Hinglish chat lines, so the model also sees Hinglish that is not
    about banks, bills or parcels. Also written by a language model."""
    raw = download("hinglish_chat.csv").read_bytes().decode("utf-8", errors="ignore")
    raw = raw[: raw.rfind("\n")]  # the download stops mid-file, so drop the cut-off last row
    lines = [cell.strip() for row in list(csv.reader(io.StringIO(raw)))[1:] for cell in row]
    lines = pd.Series(lines).drop_duplicates()
    lines = lines[lines.str.len() >= 12].sample(CHAT_SAMPLE, random_state=7)
    return pd.DataFrame({"text": lines, "label": "ham", "language": "hinglish", "source": "hinglish_chat", "topic": "everyday"})


def load_handwritten(training_messages):
    """Hand-written Indian messages to check the model on, from a different author.

    Most of this dataset is a copy of the UCI data, so keep only the rows that
    are not in our training file.
    """
    rows = json.loads(download("handwritten.json").read_text(encoding="utf-8"))["rows"]
    df = pd.DataFrame(rows).drop_duplicates("text")

    # Compare the first 20 letters: the copies differ from ours in punctuation and endings
    def start(text):
        return letters_only(unescape(text))[:20]

    df = df[~df["text"].map(start).isin(set(training_messages.map(start)))]
    df["label"] = df["label"].map({1: "spam", 0: "ham"})
    return df[["text", "label"]]


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    messages = pd.concat([load_uci(), load_indian_scam_sms(), load_hinglish_chat()], ignore_index=True)
    messages = messages[COLUMNS].drop_duplicates("text")
    messages.to_csv(DATA / "messages.csv", index=False)
    print(f"\ndata/messages.csv: {len(messages)} messages")
    print(pd.crosstab(messages["source"], messages["label"]).to_string())

    handwritten = load_handwritten(messages["text"])
    handwritten.to_csv(DATA / "handwritten_test.csv", index=False)
    print(f"\ndata/handwritten_test.csv: {len(handwritten)} messages, {handwritten['label'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
