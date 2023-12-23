import torch
from transformers import BertTokenizer, BertForSequenceClassification
import time

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
    start_time = time.time()
    model.eval()
    with torch.no_grad():
        inputs = preprocess_text(text, tokenizer)
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        outputs = model(input_ids, attention_mask=attention_mask)
        prediction = torch.argmax(outputs.logits, dim=1).item()
    end_time = time.time()
    processing_time = end_time - start_time
    return prediction, processing_time

def main():
    model_path = '/Users/ameedjamous/programming/OpenTextShield/src/BERT/training/bert_sms_spam_phishing_model'
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Load the trained model
    model = load_model(model_path)

    # Sample text to classify
    sample_text = "Free entry in 2 a weekly competition to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate). T&C's apply 08452810075over18's, Free entry in 2 a weekly competition to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate). T&C's apply 08452810075over18's, Free entry in 2 a weekly competition to win FA Cup final tkts 21st May 3838329832983092823098320983209823902389028239038329083290. Text FA to 87121 to receive entry question(std txt rate). T&C's apply 08452810075over18's"

    # Get prediction and processing time
    prediction, processing_time = predict(sample_text, model, tokenizer)

    # Convert numerical prediction back to label
    label_map = {0: 'ham', 1: 'spam', 2: 'phishing'}
    print(f"The provided text is predicted as: {label_map[prediction]}")

    # Determine the emoji based on processing time
    emoji = "😊" if processing_time <= 0.2 else "😔"
    print(f"Processing time: {processing_time:.2f} seconds {emoji}")

if __name__ == '__main__':
    main()
