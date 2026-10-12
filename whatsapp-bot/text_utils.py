"""Text cleaning shared by training and prediction.

Both must clean text the same way. If prediction cleaned text differently from
training, the words would not match the vocabulary the model learned.
"""
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

for package, path in [("punkt_tab", "tokenizers/punkt_tab"), ("stopwords", "corpora/stopwords")]:
    try:
        nltk.data.find(path)
    except LookupError:
        nltk.download(package, quiet=True)

_stemmer = PorterStemmer()
_stop_words = set(stopwords.words("english"))


def transform_text(text):
    """Same steps as the notebook: lowercase, tokenize, keep letters and digits,
    drop stop words, stem."""
    tokens = nltk.word_tokenize(text.lower())
    tokens = [t for t in tokens if t.isalnum()]
    tokens = [t for t in tokens if t not in _stop_words]
    return " ".join(_stemmer.stem(t) for t in tokens)
