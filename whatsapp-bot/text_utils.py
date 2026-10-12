"""Text cleaning shared by training and prediction.

Both must clean text the same way. The cleaning is stored inside the saved
model (as the vectorizers' preprocessor), so prediction cannot drift from
training.
"""
import re

# Stand-ins for things whose exact value should not matter: a scam is a scam
# whichever number it tells you to call. Replacing links also stops the model
# from memorising which web addresses appeared in the safe training messages.
_URL = re.compile(
    r"(?:https?://|www\.)\S+"
    r"|\b[\w-]+(?:\.[\w-]+)*\.(?:com|in|net|org|co|ly|xyz|info|io|app|site|online|top)\b(?:/\S*)?"
)
# 10 or more digits in a row, or an Indian mobile written as two groups of five
_PHONE = re.compile(r"(?<!\d)(?:\+?\d{10,}|(?:\+?91[\s-]?)?\d{5}[\s-]\d{5})(?!\d)")
_MONEY = re.compile(r"[₹£$€]|\brs\b\.?|\binr\b")
_NUMBER = re.compile(r"\d+(?:[.,:/-]\d+)*")
# Python does not count Indic vowel signs as letters, so keep those scripts by range
_PUNCTUATION = re.compile(r"[^\wऀ-෿]+")

MASK_TOKENS = {"urltoken", "phonetoken", "moneytoken", "numtoken"}


def clean_text(text):
    """Lowercase, replace links, phone numbers, currency signs and numbers with
    placeholder words, and drop punctuation.

    There is no stop word removal or stemming, as the notebook had: both only
    know English and would mangle Hinglish.
    """
    text = text.lower()
    text = _URL.sub(" urltoken ", text)
    text = _PHONE.sub(" phonetoken ", text)
    text = _MONEY.sub(" moneytoken ", text)
    text = _NUMBER.sub(" numtoken ", text)
    return _PUNCTUATION.sub(" ", text).strip()
