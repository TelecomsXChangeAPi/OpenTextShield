import torch
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig
import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

# Setup logging
logging.basicConfig(filename='prediction_logs.log', level=logging.INFO,
                    format='%(asctime)s - Request: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

def load_model(model_path, device):
    config = BertConfig.from_pretrained('bert-base-uncased', num_labels=3)
    model = BertForSequenceClassification(config)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # Set the model to evaluation mode
    model.to(device)  # Move the model to the MPS device or CPU
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

def predict(text, model, tokenizer, device):
    with torch.no_grad():
        inputs = preprocess_text(text, tokenizer)
        inputs = {k: v.to(device) for k, v in inputs.items()}  # Ensure inputs are on the same device as the model
        outputs = model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=1).item()
    # Log the request and prediction result
    logging.info(f"Text: {text} | Prediction: {prediction}")
    return prediction

def generate_random_text(base_text, index):
    return f"{base_text} - Message {index} - Random {random.randint(1, 10000)}"

def main():
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    model_path = '/Users/ameedjamous/programming/OpenTextShield/src/BERT/training/mlx-bert/bert_sms_spam_phishing_model_gpu.pth'
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = load_model(model_path, device)

    base_text = "Sample SMS text"
    sample_texts = [generate_random_text(base_text, i) for i in range(1000)]  # Adjusted to 1000 messages

    executor = ThreadPoolExecutor(max_workers=10)  # Adjust based on your machine

    start_time = time.time()

    tasks = [(text, model, tokenizer, device) for text in sample_texts]
    futures = [executor.submit(predict, *task) for task in tasks]

    for i, future in enumerate(as_completed(futures), 1):
        _ = future.result()  # We already logged the result in predict function
        if i % 50 == 0:
            print(f"Processed {i} messages...")

    end_time = time.time()
    total_time = end_time - start_time
    print(f"Processed {len(sample_texts)} messages in {total_time:.2f} seconds")

    executor.shutdown()

if __name__ == '__main__':
    main()
