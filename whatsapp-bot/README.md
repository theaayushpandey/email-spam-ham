# WhatsApp spam checker

Forward a suspicious SMS to a WhatsApp number and get back "likely spam", "looks safe" or "not sure", with how confident the model is. Reply WRONG when it misjudges a message and that message is saved for retraining.

This is a prototype. It runs on Twilio's WhatsApp sandbox, which only talks to phone numbers that have joined the sandbox.

## How it works

```
You on WhatsApp --> Twilio --> POST /whatsapp on this Flask server
                                   |  clean the text (same steps as training)
                                   |  TF-IDF vector -> Naive Bayes -> spam probability
You on WhatsApp <-- Twilio <-- reply as TwiML (XML)
```

| File | What it does |
| --- | --- |
| `text_utils.py` | Text cleaning used by both training and prediction |
| `train.py` | Trains the models, prints the test results, saves the best one to `model.joblib` |
| `app.py` | The Flask server: `/predict` for JSON, `/whatsapp` for Twilio |

## Results on the test set (1,034 messages, 138 of them spam)

| Model | Accuracy | Precision | Recall | Real messages flagged | Spam missed |
| --- | --- | --- | --- | --- | --- |
| Multinomial Naive Bayes | 96.7% | 100% | 75.4% | 0 | 34 |
| Bernoulli Naive Bayes (used by the bot) | 98.1% | 100% | 85.5% | 0 | 20 |

These are a little lower than the notebook's numbers. The notebook fitted TF-IDF on all messages before splitting into train and test, so the test set leaked into training. `train.py` fits it on the training messages only.

## Run it

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python train.py
python app.py
```

Test it without WhatsApp:

```bash
curl -X POST http://localhost:5001/predict -H "Content-Type: application/json" -d "{\"text\": \"Free prize! Call now to claim\"}"
```

## Connect it to WhatsApp

1. Create a free Twilio account and open Messaging, then Try it out, then Send a WhatsApp message. That is the sandbox.
2. From your phone, send the join code shown there to the sandbox number.
3. The server needs a public address. Run a tunnel such as `ngrok http 5001` and copy the https address it gives.
4. In the sandbox settings, set "When a message comes in" to `https://<your-tunnel-address>/whatsapp`, method POST.
5. Send any message to the sandbox number.

## Known limitations

- The training data is English SMS from the UK and Singapore, collected more than a decade ago. Indian scam wording ("lakh", "KYC", "UPI") is mostly missing, so messages like "you won one lakh rupees" are judged safe. The WRONG reply exists to collect exactly these.
- The confidence figure comes from Naive Bayes, which is overconfident. It is not the model's accuracy.
- Messages with no known words (for example Hindi or Hinglish) get "I can't judge this one" instead of a guess.
- The webhook does not check Twilio's signature, so anyone who knows the address can call it.
- Feedback is saved to a CSV file but retraining on it is still a manual step.
