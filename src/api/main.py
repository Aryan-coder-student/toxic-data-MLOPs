# src/api/main.py
from fastapi import FastAPI
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import yaml
import uvicorn

app = FastAPI()

with open("config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

model = BertForSequenceClassification.from_pretrained(config["model"]["output_dir"])
tokenizer = BertTokenizer.from_pretrained(config["model"]["output_dir"])
model.eval()

@app.post("/predict")
async def predict(comment: str):
    encoding = tokenizer(comment, return_tensors="pt", truncation=True, padding=True, max_length=config["model"]["max_len"])
    with torch.no_grad():
        outputs = model(**encoding)
        probs = torch.sigmoid(outputs.logits).tolist()[0]
    labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    return {label: prob for label, prob in zip(labels, probs)}

if __name__ == "__main__":
    uvicorn.run(app, host=config["api"]["host"], port=config["api"]["port"])