import time
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime
from fastapi.responses import FileResponse
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import BertForSequenceClassification 
import csv

app = FastAPI()

# Add CORSMiddleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Load the trained BERT model and tokenizer
model_path = "../training/bert_sms_spam_phishing_model"  # Update this path
model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
print("Model loaded successfully")

# Ensure model is in evaluation mode
model.eval()

class Feedback(BaseModel):
    content: str
    feedback: str
    thumbs_up: bool
    thumbs_down: bool
    user_id: Optional[str] = None

@app.post("/feedback-loop/")
async def feedback_loop(feedback: Feedback):
    thumbs_up = 'Yes' if feedback.thumbs_up else 'No'
    thumbs_down = 'Yes' if feedback.thumbs_down else 'No'
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open("feedback.csv", mode="a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        if file.tell() == 0:
            writer.writerow([
                "Timestamp", "UserID", "Content", "Feedback", "Thumbs Up", "Thumbs Down"
            ])
        writer.writerow([
            timestamp, feedback.user_id, feedback.content, feedback.feedback, thumbs_up, thumbs_down
        ])
    return {"message": "Feedback received"}

class SMS(BaseModel):
    text: str

def preprocess_text(text, tokenizer, max_len=128):
    inputs = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=max_len,
        padding='max_length',
        return_attention_mask=True,
        return_tensors='pt',
        truncation=True
    )
    return inputs

@app.post("/predict/")
async def predict_sms(sms: SMS):
    start_time = time.time()

    text = sms.text
    if not text:
        raise HTTPException(status_code=400, detail="Text is empty")

    inputs = preprocess_text(text, tokenizer)
    with torch.no_grad():
        outputs = model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=1).item()

    end_time = time.time()
    processing_time = end_time - start_time

    label_map = {0: 'ham', 1: 'spam', 2: 'phishing'}
    label = label_map[prediction]

    return {
        "label": label,
        "processing_time": processing_time,
        "Model_Name": "OTS_BERT_SMS_Classifier",
        "Model_Version": "1.1.4",
        "Model_Author": "TelecomsXChange (TCXC)",
        "Last_Training": "2023-12-21"  # Update accordingly
    }

@app.get("/download-feedback/")
async def download_feedback():
    file_path = "feedback.csv"
    return FileResponse(file_path, media_type='text/csv', filename=file_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
