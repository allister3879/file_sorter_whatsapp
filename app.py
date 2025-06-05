from fastapi import FastAPI, File, UploadFile
import pandas as pd
import re
from deep_translator import GoogleTranslator
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
import torch.nn.functional as F
import uvicorn

app = FastAPI()

# Load models once
toxic_tokenizer = AutoTokenizer.from_pretrained("unitary/toxic-bert")
toxic_model = AutoModelForSequenceClassification.from_pretrained("unitary/toxic-bert")
deal_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Labels
toxic_labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
threat_keywords = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult']
deal_labels = ["suspicious deal", "casual talk", "transaction", "threat", "meeting", "neutral"]

def classify_combined(text, toxic_threshold=0.5, suspicious_threshold=0.6):
    inputs = toxic_tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        logits = toxic_model(**inputs).logits
        probs = F.sigmoid(logits)[0]
    toxic_scores = {label: float(probs[i]) for i, label in enumerate(toxic_labels)}
    toxic_threat = any(toxic_scores[label] > toxic_threshold for label in threat_keywords)

    try:
        deal_result = deal_classifier(text, deal_labels)
        top_label = deal_result['labels'][0]
        top_score = deal_result['scores'][0]
        suspicious = top_label == "suspicious deal" and top_score > suspicious_threshold
    except:
        suspicious = False

    if toxic_threat or suspicious:
        return 'threat'
    else:
        return 'neutral'

@app.post("/analyze/")
async def analyze_chat(file: UploadFile = File(...)):
    content = await file.read()
    lines = content.decode('utf-8').splitlines()

    pattern = r'^\[(\d{2}/\d{2}/\d{4}),\s(\d{2}:\d{2}:\d{2})\]\s(.+?):\s?(.*)'
    data = []
    for line in lines:
        match = re.match(pattern, line)
        if match:
            date, time, sender, message = match.groups()
            data.append([date, time, sender.strip(), message.strip()])

    df = pd.DataFrame(data, columns=['Date', 'Time', 'Sender', 'Message'])
    df['Translated'] = df['Message'].apply(lambda x: GoogleTranslator(source='auto', target='en').translate(x))
    df['FinalLabel'] = df['Translated'].apply(classify_combined)

    result = df.to_dict(orient='records')
    return {"result": result}
