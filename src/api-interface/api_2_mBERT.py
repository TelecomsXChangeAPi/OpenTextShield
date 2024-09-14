import time
from typing import Optional
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime
from fastapi.responses import FileResponse
import torch
from transformers import AutoTokenizer, BertForSequenceClassification, BertConfig
import fasttext
import csv

app = FastAPI()

# Allowed IPs
ALLOWED_IPS = {"ANY", "127.0.0.1", "localhost", "10.0.0.1"}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Path to your models
bert_model_path = {
    "bert-base-uncased": "/home/ots/OpenTextShield/src/BERT/training/bert-mlx-apple-silicon/bert_ots_model_1.5.pth",
    "bert-base-multilingual-cased": "/home/ots/OpenTextShield/src/mBERT/training/mbert-mlx-apple-silicon/mbert_ots_model_2.1.pth"
}

# Load models dynamically based on the model type
models = {}

for model_name, model_path in bert_model_path.items():
    config = BertConfig.from_pretrained(model_name, num_labels=3)  # Adjust num_labels as per your model
    model = BertForSequenceClassification(config)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'), weights_only=True))  # Set weights_only=True
    model.eval()
    models[model_name] = model.to(torch.device("cpu"))  # Assuming CPU for simplicity

# Load FastText model
fasttext_model_path = "/home/ots/OpenTextShield/src/FastText/training/ots_fastext_model_v2.1.bin"
fasttext_model = fasttext.load_model(fasttext_model_path)

# Tokenizers for each BERT model
tokenizers = {
    model_name: AutoTokenizer.from_pretrained(model_name)
    for model_name in bert_model_path.keys()
}

class SMS(BaseModel):
    text: str
    model: str  # "bert" or "fasttext"
    bert_version: Optional[str] = "bert-base-uncased"  # "bert-base-uncased" or "bert-base-multilingual-cased"

class Feedback(BaseModel):
    content: str
    feedback: str
    thumbs_up: bool
    thumbs_down: bool
    user_id: Optional[str] = None
    model: str  # "bert" or "fasttext"
    bert_version: Optional[str] = "bert-base-uncased"

def preprocess_text(text, tokenizer, max_len=128):
    return tokenizer.encode_plus(
        text, add_special_tokens=True, max_length=max_len,
        padding='max_length', return_attention_mask=True,
        return_tensors='pt', truncation=True
    )


def write_feedback(feedback_data, model_name):
    file_name = f"feedback_{model_name}.csv"
    with open(file_name, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        if file.tell() == 0:
            writer.writerow(["Timestamp", "UserID", "Content", "Feedback", "Thumbs Up", "Thumbs Down"])
        writer.writerow(feedback_data)

def verify_ip_address(request: Request):
    if "ANY" in ALLOWED_IPS:
        return request.client.host  # Bypass IP check and allow any IP
    client_host = request.client.host
    if client_host not in ALLOWED_IPS:
        raise HTTPException(status_code=403, detail="Access denied")
    return client_host

@app.post("/predict/", dependencies=[Depends(verify_ip_address)])
async def predict_sms(sms: SMS):
    start_time = time.time()

    if not sms.text:
        raise HTTPException(status_code=400, detail="Text is empty")

    if sms.model == "bert":
        bert_version = sms.bert_version if sms.bert_version in models else "bert-base-uncased"
        tokenizer = tokenizers[bert_version]
        model = models[bert_version]

        inputs = preprocess_text(sms.text, tokenizer)
        inputs = {k: v.to(torch.device("cpu")) for k, v in inputs.items()}  # Ensure inputs are on the correct device
        with torch.no_grad():
            outputs = model(**inputs)
            prediction = torch.argmax(outputs.logits, dim=1).item()
        label_map = {0: 'ham', 1: 'spam', 2: 'phishing'}
        label = label_map[prediction]
        probability = torch.nn.functional.softmax(outputs.logits, dim=1).max().item()
        model_info = {"Model_Name": "OTS_mBERT", "Model_Version": bert_version}
    elif sms.model == "fasttext":
        label, probability = fasttext_model.predict(sms.text, k=1)
        label = label[0].replace('__label__', '')
        probability = probability[0]
        model_info = {
            "Model_Name": "OTS_fasttext",
            "Model_Version": "1.9.0",
            "Model_Author": "TelecomsXChange (TCXC)",
            "Last_Training": "2024-03-20"
        }
    else:
        raise HTTPException(status_code=400, detail="Invalid model type")

    end_time = time.time()
    return {
        "label": label,
        "probability": probability,
        "processing_time": end_time - start_time,
        **model_info,
        "Model_Author": "TelecomsXChange (TCXC)",
        "Last_Training": "2024-03-20"
    }

# Feedback and download-feedback routes remain unchanged

