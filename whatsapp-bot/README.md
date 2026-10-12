# WhatsApp spam checker

Forward a suspicious SMS or WhatsApp message to a WhatsApp number and get back "likely spam", "looks safe" or "not sure", with how confident the model is. It reads English and Hinglish (Hindi typed in English letters). Reply WRONG when it misjudges a message and that message is saved for retraining.

This is a prototype. It runs on Twilio's WhatsApp sandbox, which only talks to phone numbers that have joined the sandbox.

## How it works

```
You on WhatsApp --> Twilio --> POST /whatsapp on this Flask server
                                   |  clean the text (same steps as training)
                                   |  words and word pieces (TF-IDF) -> logistic regression -> spam probability
You on WhatsApp <-- Twilio <-- reply as TwiML (XML)
```

| File | What it does |
| --- | --- |
| `build_data.py` | Downloads the Indian and Hinglish datasets and builds `data/messages.csv` |
| `text_utils.py` | Text cleaning used by both training and prediction |
| `train.py` | Trains the models, prints the test results, saves the best one to `model.joblib` |
| `app.py` | The Flask server: `/predict` for JSON, `/whatsapp` for Twilio |

## Training data

The notebook's data is English SMS from the UK and Singapore, collected before 2012. It has no "KYC" or "UPI" in it and almost no Hinglish. `build_data.py` adds Indian and Hinglish messages to it.

| Source | Messages | What it is | Licence |
| --- | --- | --- | --- |
| [UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) (`../email-spam-classifier/spam.csv`) | 5,168 | Real English SMS, UK and Singapore | See the UCI page |
| [Ridham115/indian-scam-sms-synthetic-audited](https://huggingface.co/datasets/Ridham115/indian-scam-sms-synthetic-audited) | 1,580 | Scam SMS and their genuine look-alikes (bank, courier, bill, job, loan). 639 Hinglish, 262 Indian English, 167 Hindi, 512 Bengali, Marathi, Tamil or Telugu mixed with English | CC BY 4.0 |
| [Abhishekcr448/Hinglish-Everyday-Conversations-1M](https://huggingface.co/datasets/Abhishekcr448/Hinglish-Everyday-Conversations-1M) | 1,200 sampled | Everyday Hinglish chat lines, all safe, so the model also sees Hinglish that is not about banks or bills | MIT |
| [bolewara/hinglish-scam-text-dataset](https://huggingface.co/datasets/bolewara/hinglish-scam-text-dataset) | 85 | Hand-written Indian scam and normal messages. Never trained on, only tested on | CC BY 4.0 |

**The two middle datasets were written by language models, not collected from real phones.** There is no large public set of real, labelled Hinglish spam. The Ridham115 messages were generated with Gemini and a Groq-hosted model, and its dataset card notes that those providers' terms apply to the text as well as the CC BY licence.

Changes made to the sources: labels renamed to `spam` and `ham`, duplicates removed, the UCI file's damaged characters and cut-off messages repaired, and the 85 hand-written messages are what is left of the bolewara dataset after removing its copy of the UCI data.

Looked at and not used:

- `CloveAI/india-spam-sms`: 20,010 rows, but they are 127 sentence templates with random names and numbers filled in. Adding it did not change the results.
- `Shade63/scam-classification-multiclass`: 14,000 rows with no licence.
- `karanverma19/Indian_Multilingual_Scam_Message_Dataset`: 120 rows, 49 of them different, with "immediately" or "jaldi kare" added at random to scam and genuine messages alike.

## Results

Test set: 1,590 messages the model did not train on, 290 of them spam.

| Model | Accuracy | Precision | Recall | Real messages flagged | Spam missed |
| --- | --- | --- | --- | --- | --- |
| Bernoulli Naive Bayes | 91.3% | 68.6% | 95.9% | 127 | 12 |
| Logistic regression (used by the bot) | 97.7% | 94.7% | 92.4% | 15 | 22 |

Logistic regression on the same test set, by kind of message:

| Kind of message | Messages | Real messages flagged | Spam missed |
| --- | --- | --- | --- |
| English SMS (UK, Singapore) | 1,033 | 3 of 903 | 10 of 130 |
| Indian Hinglish | 125 | 7 of 61 | 3 of 64 |
| Indian English | 53 | 1 of 26 | 1 of 27 |
| Hindi (Devanagari) | 40 | 2 of 19 | 4 of 21 |
| Other Indian code-mixed | 99 | 2 of 51 | 4 of 48 |
| Hinglish everyday chat | 240 | 0 of 240 | no spam in this group |

These numbers flatter the model on the Indian rows, because the test messages were written by the same language models as the training messages. Two harder checks:

- **Hand-written messages from a different author** (85, never trained on): 0 of 41 real messages flagged, 6 of 44 scams missed.
- **A scam type the model has never seen.** Train without one scam topic, then test on that topic alone. It catches between 67% (electricity bill) and 96% (lottery) of the unseen scams, and flags between 6% and 23% of the genuine look-alikes.

The second check is the closest thing here to "a new scam appears next month". Real accuracy on real WhatsApp traffic is not known.

For comparison, the previous version (English stop words and stemming, Naive Bayes, UCI data only) was scored once on the same test messages before it was replaced. It missed 35 of the 64 Hinglish scams and 31 of the 44 hand-written ones, and missed 26 of 130 on English against 10 now.

## Run it

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python build_data.py
python train.py
python app.py
```

`build_data.py` downloads about 2.4 MB. Skip it if `data/messages.csv` is already there.

Test it without WhatsApp:

```bash
curl -X POST http://localhost:5001/predict -H "Content-Type: application/json" -d "{\"text\": \"Aapka KYC expire ho gaya hai, turant update karein warna account block ho jayega: http://sbi-kyc.xyz\"}"
```

## Connect it to WhatsApp

1. Create a free Twilio account and open Messaging, then Try it out, then Send a WhatsApp message. That is the sandbox.
2. From your phone, send the join code shown there to the sandbox number.
3. The server needs a public address. Run a tunnel such as `ngrok http 5001` and copy the https address it gives.
4. In the sandbox settings, set "When a message comes in" to `https://<your-tunnel-address>/whatsapp`, method POST.
5. Send any message to the sandbox number.

## Known limitations

- The Indian and Hinglish training data is synthetic. Real scams are messier and change over time. The WRONG reply exists to collect real ones, and that is the only route to a model that can be trusted on Hinglish.
- Heavily shortened Hinglish is still missed. "apka kyc expir ho gya h, turnt update kro" is judged safe, because the training messages are spelled out in full.
- Short fragments are missed too. "Aapka KYC expire ho gaya hai, turant update karein" on its own is judged safe (27%); with "warna account block ho jayega" and a link it is caught (99%).
- Genuine bank, bill and courier messages are the hardest case: 7 of 61 genuine Hinglish ones were flagged in the test.
- The model leans on phone numbers. In the training data they appear almost only in scams, so a friend sending "mera naya number 98..." gets "not sure".
- Links are replaced with a placeholder before the model sees them, so it cannot tell `sbi.co.in` from `sbi-kyc.xyz`. It judges the words around the link.
- Hindi in Devanagari and the regional code-mixed styles have only 125 to 167 messages each. Treat those verdicts as rough.
- The confidence figure is how sure the model is about one message. It is not the model's accuracy.
- Messages with no known words, or with nothing but a link or a phone number, get "I can't judge this one" instead of a guess.
- The webhook does not check Twilio's signature, so anyone who knows the address can call it.
- Feedback is saved to a CSV file but retraining on it is still a manual step.
