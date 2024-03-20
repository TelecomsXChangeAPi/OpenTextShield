import torch
from transformers import BertTokenizer, BertForSequenceClassification
import time
import random

def load_model(model_path):
    model = BertForSequenceClassification.from_pretrained(model_path, num_labels=3, local_files_only=True)
    return model

def preprocess_text(text, tokenizer, max_len=128):
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=max_len,
        return_token_type_ids=False,
        padding='max_length',
        return_attention_mask=True,
        return_tensors='pt',
        truncation=True
    )
    return encoding

def predict(text, model, tokenizer):
    model.eval()
    with torch.no_grad():
        inputs = preprocess_text(text, tokenizer)
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        outputs = model(input_ids, attention_mask=attention_mask)
        prediction = torch.argmax(outputs.logits, dim=1).item()
    return prediction

def generate_random_text(base_text, index):
    return f"{base_text} - Message {index} - Random {random.randint(1, 10000)}"

def main():
    model_path = '~/programming/OpenTextShield/src/BERT/training/bert_sms_spam_phishing_model'
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = load_model(model_path)

    # Generate unique sample texts
    base_text = "Sample SMS text"
    sample_texts = [generate_random_text(base_text, i) for i in range(500)]

    # Stress test with progress logging
    start_time = time.time()

    for i, text in enumerate(sample_texts):
        predict(text, model, tokenizer)
        if (i + 1) % 50 == 0:
            print(f"Processed {i + 1} messages...")

    end_time = time.time()
    total_time = end_time - start_time
    print(f"Processed 500 messages in {total_time:.2f} seconds")

if __name__ == '__main__':
    main()
