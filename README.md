# Spam or ham: SMS spam detection, from a notebook to a WhatsApp bot

A machine learning project that tells spam messages from genuine ones ("ham"). It has two parts:

1. **A notebook** that explores 5,572 real SMS messages and builds a first spam classifier with Naive Bayes.
2. **A WhatsApp bot** that puts a classifier to work. Forward it a message you are unsure about, in English or Hinglish (Hindi typed in English letters), and it replies whether the message looks like spam.

Despite the repo's name, the data is SMS and chat messages, not email.

```
You:  Aapko 25 lakh ki lottery lagi hai KBC se. Whatsapp karein 9123456780
Bot:  Likely SPAM (99% sure). Don't click links or share an OTP.

You:  bhai kal milte hai 5 baje
Bot:  Looks safe (spam chance 1%).
```

## What is in the repo

| Path | What it is |
| --- | --- |
| `email-spam-classifier/project.ipynb` | The notebook, with every output and chart |
| `email-spam-classifier/SpamDetection.ipynb` | A working copy of the same analysis, with a correlation heatmap added |
| `email-spam-classifier/spam.csv` | The UCI SMS Spam Collection: 5,572 labelled English SMS |
| `whatsapp-bot/` | The bot: data builder, training script and web server. It has [its own README](whatsapp-bot/README.md) with the full results and the WhatsApp setup steps |

## Features

### Part 1: the notebook

The notebook walks through a complete text classification project, from the raw file to a working classifier.

- **Data cleaning.** Drops three mostly empty columns, renames the rest, turns the `ham`/`spam` labels into 0 and 1, and removes 403 duplicate messages. That leaves 5,169 messages: 4,516 ham and 653 spam.
- **Exploring the data.** A pie chart shows how unbalanced the data is (about 13% spam). Each message gets a character count, a word count and a sentence count, compared between spam and ham with histograms and a pair plot. The clearest finding: spam is about twice as long, 138 characters on average against 70.
- **Text preprocessing.** Every message is lowercased, split into words, stripped of punctuation and English stop words ("the", "is", "and"), and stemmed, so "winning", "wins" and "win" count as one word.
- **Word clouds and top words.** One word cloud for spam and one for ham, plus bar charts of the 30 most common words in each. Spam is full of "call", "free" and "txt".
- **Three models compared.** The cleaned text becomes TF-IDF numbers (the 3,000 most common words), the data is split 80/20, and three kinds of Naive Bayes are trained and scored:

  | Model | Accuracy | Precision |
  | --- | --- | --- |
  | Gaussian Naive Bayes | 86.9% | 50.7% |
  | Multinomial Naive Bayes | 97.1% | 100% |
  | Bernoulli Naive Bayes | 98.4% | 99.2% |

  Precision matters most here: flagging a real message as spam is worse than letting one spam through. These scores are slightly generous, because TF-IDF was fitted on all the messages before the split. The bot's training script fits it on the training messages only.
- **Try it yourself.** The last cell is a loop: type a message, get back SPAM or NOT SPAM.

That last cell also shows where the notebook stops. It judges "you won one lakh rupees" as not spam, because its data is English SMS from the UK and Singapore, collected before 2012. It has never seen "KYC" or "UPI", and almost none of its 5,572 messages are in Hinglish. Part 2 is the answer to that.

### Part 2: the WhatsApp bot

**What you get as a user**

- **Ask on WhatsApp.** Forward any message to the bot's number and it replies in the same chat. Nothing to install.
- **Three verdicts, not two.** "Likely SPAM" when the model is at least 70% sure, "Looks safe" when the spam chance is 30% or less, and "Not sure about this one" in between. A model that must always pick a side is wrong more often than one allowed to say it doesn't know.
- **A fourth answer when it has nothing to go on.** If a message has no words the model learned, or is nothing but a link or a phone number, the bot says "I can't judge this one" instead of guessing.
- **English and Hinglish.** It catches "Aapka KYC expire ho gaya hai, turant update karein warna account block ho jayega" followed by a link (99% sure) as well as "Free entry in a weekly competition to win FA Cup tickets. Text WIN to 87121" (93%). Hindi in Devanagari and Bengali, Marathi, Tamil or Telugu mixed with English get a rough verdict.
- **A confidence figure with every verdict.** It is never shown as 0% or 100%, because the model is never certain.
- **Safety advice in the reply.** Spam verdicts end with "Don't click links or share an OTP."
- **Reply WRONG to correct it.** The misjudged message is saved with the right label for the next retraining. The sender's phone number is never written to disk.
- **Says hello.** "hi", "hello" or "help" gets a short explanation of how to use it.

More real replies:

```
You:  Your OTP is 482913. Do not share it with anyone.
Bot:  Looks safe (spam chance 1%).

You:  Mera naya number 9876543210 hai, save kar lena
Bot:  Not sure about this one (spam chance 44%). Treat it with care.

You:  https://bit.ly/3xYz
Bot:  I can't judge this one: it has no words I learned from. Be careful with links and never share an OTP.
```

Every verdict also ends with "This is an automated guess. Reply WRONG if I got it wrong."

**What you get as a developer**

- **A JSON API.** `POST /predict` with `{"text": "..."}` returns `{"label": "spam", "spam_probability": 0.992}`. The label is `spam`, `ham`, `unsure` or `unknown`. Other apps can use the model without WhatsApp.
- **A Twilio webhook.** `POST /whatsapp` takes Twilio's form fields and answers in TwiML, the XML Twilio turns into a WhatsApp message.
- **A health check.** `GET /` reports that the server is up and which model it loaded.
- **Input limits.** Empty requests get a 400 and messages over 2,000 characters a 413, with a readable error.

**How the model is built**

- **Indian and Hinglish training data.** 7,948 messages from three sources: the notebook's 5,168 English SMS, 1,580 Indian scam messages with their genuine look-alikes (real-looking bank, courier, bill, job and loan messages), and 1,200 lines of everyday Hinglish chat.
- **A repeatable data build.** `build_data.py` downloads each source at a fixed version, so a rebuild gives the same file byte for byte. It also repairs the original CSV, where about 50 messages were cut off at their first comma.
- **Cleaning that works for Hinglish.** English stop word lists and stemmers mangle Hinglish, so they are gone. Links, phone numbers, currency signs and numbers are replaced with placeholder words instead: a scam is a scam whichever number it tells you to call.
- **Words plus pieces of words.** People spell Hinglish however they like. Besides whole words, the model reads pieces of 2 to 5 letters, so "karein", "kare" and "karo" all still match.
- **A model chosen by measurement.** `train.py` trains Naive Bayes and logistic regression on the same data and keeps the one with the higher precision. Logistic regression flagged 15 real messages in the test set and Naive Bayes 127.
- **Three levels of testing.** Good scores on a test set mean little when the test messages come from the same source as the training ones. So `train.py` also tests on hand-written messages from a different author, and on whole scam types it was never shown.

## How the bot works

```
You on WhatsApp --> Twilio --> POST /whatsapp on the Flask server
                                   |  clean the text (same steps as training)
                                   |  words and word pieces (TF-IDF) -> logistic regression -> spam probability
You on WhatsApp <-- Twilio <-- reply as TwiML (XML)
```

## Results

What retraining on Indian and Hinglish data changed. Both models were scored on the same messages: a test set of 1,590 and the 85 hand-written ones.

| | English-only model | Current model |
| --- | --- | --- |
| Hinglish scams missed | 35 of 64 | 3 of 64 |
| Genuine Hinglish messages flagged | 11 of 61 | 7 of 61 |
| Hand-written Indian scams missed (never trained on) | 31 of 44 | 6 of 44 |
| Hand-written genuine messages flagged | 1 of 41 | 0 of 41 |
| English spam missed | 26 of 130 | 10 of 130 |
| Genuine English messages flagged | 3 of 903 | 3 of 903 |

Overall on the test set: 97.7% accuracy, 94.7% precision, 92.4% recall.

The harder test is a scam type the model has never seen. Trained without one type and tested on it alone, it catches between 67% (electricity bill scams) and 96% (lottery scams), and flags between 6% and 23% of the genuine look-alikes. That is the better guide to how it handles next month's new scam.

The full tables are in the [bot's README](whatsapp-bot/README.md).

## Run it

```bash
git clone https://github.com/theaayushpandey/email-spam-ham.git
cd email-spam-ham
```

**The notebook**

```bash
pip install notebook numpy pandas matplotlib seaborn nltk scikit-learn wordcloud
jupyter notebook email-spam-classifier/project.ipynb
```

**The bot** (tested with Python 3.14)

```bash
cd whatsapp-bot
python -m venv .venv
.venv\Scripts\activate          # macOS and Linux: source .venv/bin/activate
pip install -r requirements.txt
python train.py                 # trains the model and prints the results
python app.py                   # starts the server on port 5001
```

Then try it without WhatsApp:

```bash
curl -X POST http://localhost:5001/predict -H "Content-Type: application/json" -d "{\"text\": \"Aapka KYC expire ho gaya hai, turant update karein warna account block ho jayega: http://sbi-kyc.xyz\"}"
```

```json
{"label": "spam", "spam_probability": 0.992}
```

To connect it to WhatsApp through Twilio's free sandbox, follow the five steps in the [bot's README](whatsapp-bot/README.md#connect-it-to-whatsapp). To rebuild the training data from its sources, run `python build_data.py` before `train.py`.

## Limitations

This is a prototype, and the bot's verdict is a guess, not a guarantee.

- **Most of the Indian and Hinglish training data was written by language models**, not collected from real phones. No large public set of real, labelled Hinglish spam exists. Real accuracy on real WhatsApp messages is not known, and the WRONG reply is there to collect the real examples that would fix this.
- **Short fragments and heavily shortened Hinglish are still missed.** "Congratulations! You won one lakh rupees. Call now to claim your prize." is caught at 91%, but the bare "you won one lakh rupees" is judged safe. So is the KYC message above once its threat and link are cut off, and so is "apka kyc expir ho gya h, turnt update kro".
- **Genuine bank, bill and courier messages are the hardest case.** They look like the scams that imitate them.
- **It leans on phone numbers.** In the training data they appear almost only in scams, so a friend sending a new number gets "not sure".
- **It does not inspect links.** It cannot tell `sbi.co.in` from `sbi-kyc.xyz`. It judges the words around the link.
- **It runs on Twilio's sandbox**, which only talks to phone numbers that have joined it, and the webhook does not yet check that requests really come from Twilio.

More in the [bot's README](whatsapp-bot/README.md#known-limitations).

## Built with

Python, pandas, scikit-learn and Flask for the bot. Jupyter, NLTK, matplotlib, seaborn and wordcloud for the notebook. Twilio's WhatsApp sandbox for messaging.

## Data and credits

- [UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) by Tiago Almeida and José María Gómez Hidalgo.
- [Indian Scam SMS (synthetic, audited)](https://huggingface.co/datasets/Ridham115/indian-scam-sms-synthetic-audited) by Ridham115, CC BY 4.0.
- [Hinglish Everyday Conversations](https://huggingface.co/datasets/Abhishekcr448/Hinglish-Everyday-Conversations-1M) by Abhishek Khatri, MIT.
- [Hinglish & English Financial Scam/Fraud Text Dataset](https://huggingface.co/datasets/bolewara/hinglish-scam-text-dataset) by Anuj Bolewar, CC BY 4.0. Used for testing only.

What was changed in each source is listed in the [bot's README](whatsapp-bot/README.md#training-data).
