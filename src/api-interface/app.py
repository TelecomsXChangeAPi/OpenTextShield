import time
from typing import Optional
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime
from fastapi.responses import FileResponse
import torch
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig
import fasttext
import csv

app = FastAPI()

# Allowed IPs
ALLOWED_IPS = {"127.0.0.1", "localhost", "10.0.0.1"}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Update the model path to the one you want to use
bert_model_path = "../BERT/training/bert-mlx-apple-silicon/bert_ots_model_1.5.pth"

# Load BERT model with updated approach
config = BertConfig.from_pretrained('bert-base-uncased', num_labels=3)  # Adjust num_labels as per your model
bert_model = BertForSequenceClassification(config)
bert_model.load_state_dict(torch.load(bert_model_path, map_location=torch.device('cpu')))
bert_model.eval()

# Specify the device
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
bert_model = bert_model.to(device)

bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Load FastText model
fasttext_model_path = "../FastText/training/ots_sms_model_v1.1.bin"
fasttext_model = fasttext.load_model(fasttext_model_path)

class SMS(BaseModel):
    text: str
    model: str  # "bert" or "fasttext"

class Feedback(BaseModel):
    content: str
    feedback: str
    thumbs_up: bool
    thumbs_down: bool
    user_id: Optional[str] = None
    model: str  # "bert" or "fasttext"

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
        inputs = preprocess_text(sms.text, bert_tokenizer)
        inputs = {k: v.to(device) for k, v in inputs.items()}  # Ensure inputs are moved to the correct device
        with torch.no_grad():
            outputs = bert_model(**inputs)
            prediction = torch.argmax(outputs.logits, dim=1).item()
        label_map = {0: 'ham', 1: 'spam', 2: 'phishing'}
        label = label_map[prediction]
        probability = torch.nn.functional.softmax(outputs.logits, dim=1).max().item()
        model_info = {"Model_Name": "OTS_bert", "Model_Version": "1.5"}
    elif sms.model == "fasttext":
        label, probability = fasttext_model.predict(sms.text, k=1)
        label = label[0].replace('__label__', '')
        probability = probability[0]
        model_info = {
            "Model_Name": "OTS_fasttext",
            "Model_Version": "1.1.4",
            "Model_Author": "TelecomsXChange (TCXC)",
            "Last_Training": "2023-12-21"
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
        "Last_Training": "2024-03-11"  # Update accordingly
    }

@app.post("/feedback-loop/", dependencies=[Depends(verify_ip_address)])
async def feedback_loop(feedback: Feedback):
    thumbs_up = 'Yes' if feedback.thumbs_up else 'No'
    thumbs_down = 'Yes' if feedback.thumbs_down else 'No'
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    feedback_data = [timestamp, feedback.user_id, feedback.content, feedback.feedback, thumbs_up, thumbs_down]

    if feedback.model in ["bert", "fasttext"]:
        write_feedback(feedback_data, feedback.model)
    else:
        raise HTTPException(status_code=400, detail="Invalid model type")

    return {"message": "Feedback received"}

@app.get("/download-feedback/{model_name}", dependencies=[Depends(verify_ip_address)])
async def download_feedback(model_name: str):
    if model_name in ["bert", "fasttext"]:
        file_path = f"feedback_{model_name}.csv"
    else:
        raise HTTPException(status_code=400, detail="Invalid model name")
    return FileResponse(file_path, media_type='text/csv', filename=file_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
