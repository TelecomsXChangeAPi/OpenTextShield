import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import chardet

# Dataset Class
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        label = self.labels[item]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )

        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Data Loader Function
def create_data_loader(df, tokenizer, max_len, batch_size):
    ds = TextDataset(
        texts=df.text.to_numpy(),
        labels=df.label.to_numpy(),
        tokenizer=tokenizer,
        max_len=max_len
    )

    return DataLoader(ds, batch_size=batch_size, num_workers=4)

def main():
    # Detect encoding
    with open('sms_spam_phishing_dataset.csv', 'rb') as f:
        result = chardet.detect(f.read())
        file_encoding = result['encoding']

    print("Detected encoding:", file_encoding)

    # Load Dataset
    df = pd.read_csv('sms_spam_phishing_dataset.csv', encoding=file_encoding)
    df['label'] = df['label'].map({'ham': 0, 'spam': 1, 'phishing': 2})  # Convert labels to numerical

    # Parameters
    BATCH_SIZE = 16
    MAX_LEN = 128
    EPOCHS = 3

    # Split Data
    train_df, test_df = train_test_split(df, test_size=0.1)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Create Data Loaders
    train_data_loader = create_data_loader(train_df, tokenizer, MAX_LEN, BATCH_SIZE)
    test_data_loader = create_data_loader(test_df, tokenizer, MAX_LEN, BATCH_SIZE)

    # Load BERT Model
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Training Loop
    for epoch in range(EPOCHS):
        model.train()
        for batch in train_data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        print(f"Epoch {epoch + 1}/{EPOCHS} completed.")

    # Evaluate
    model.eval()
    predictions, true_labels = [], []
    for batch in test_data_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
        
        logits = outputs.logits
        predictions.extend(torch.argmax(logits, dim=1).tolist())
        true_labels.extend(labels.tolist())

    accuracy = accuracy_score(true_labels, predictions)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

    # Save the Model
    model.save_pretrained('bert_sms_spam_phishing_model')

if __name__ == '__main__':
    main()
