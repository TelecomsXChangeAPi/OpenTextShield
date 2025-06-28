import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import chardet
import time
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

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

def create_data_loader(df, tokenizer, max_len, batch_size, num_workers=0):
    ds = TextDataset(
        texts=df.text.to_numpy(),
        labels=df.label.to_numpy(),
        tokenizer=tokenizer,
        max_len=max_len
    )
    return DataLoader(ds, batch_size=batch_size, num_workers=num_workers)

def main():
    # Detect encoding
    with open('dataset/sms_spam_phishing_dataset_v2.1.csv', 'rb') as f:
        result = chardet.detect(f.read())
        file_encoding = result['encoding']

    logger.info(f"Detected file encoding: {file_encoding}")

    # Load Dataset
    df = pd.read_csv('dataset/sms_spam_phishing_dataset_v2.1.csv', encoding=file_encoding)
    df['label'] = df['label'].map({'ham': 0, 'spam': 1, 'phishing': 2})  # Convert labels to numerical
    logger.info("Dataset loaded and labels converted to numerical values.")

    # Parameters
    BATCH_SIZE = 16
    MAX_LEN = 128
    EPOCHS = 3

    # Split Data
    train_df, test_df = train_test_split(df, test_size=0.1)
    logger.info(f"Data split into train and test sets. Train size: {len(train_df)}, Test size: {len(test_df)}")

    # Update to use mBERT
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    logger.info("Tokenizer loaded from bert-base-multilingual-cased.")

    # Create Data Loaders
    train_data_loader = create_data_loader(train_df, tokenizer, MAX_LEN, BATCH_SIZE)
    test_data_loader = create_data_loader(test_df, tokenizer, MAX_LEN, BATCH_SIZE)
    logger.info("Data loaders created for training and testing datasets.")

    # Update to use mBERT
    model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased', num_labels=3)
    logger.info("Model loaded from bert-base-multilingual-cased with 3 output labels.")

    # Device detection logic updated for better clarity
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    logger.info(f"Using device: {device}")
    model.to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    logger.info("Optimizer initialized with learning rate 2e-5.")

    # Start time tracking
    start_time = time.time()
    logger.info("Training started.")

    # Training Loop
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for batch_idx, batch in enumerate(train_data_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            running_loss += loss.item()
            if batch_idx % 100 == 0:
                logger.info(f"Epoch {epoch + 1}/{EPOCHS}, Batch {batch_idx}/{len(train_data_loader)}, Loss: {loss.item():.4f}")

        logger.info(f"Epoch {epoch + 1}/{EPOCHS} completed. Average Loss: {running_loss / len(train_data_loader):.4f}")

    # End time tracking
    end_time = time.time()
    training_time = end_time - start_time
    logger.info(f"Training completed in: {training_time // 60:.0f} minutes and {training_time % 60:.0f} seconds.")

    # Evaluate
    logger.info("Evaluation started.")
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
    logger.info(f"Test Accuracy: {accuracy * 100:.2f}%")

    # Save the Model
    model_path = 'mbert_ots_model_2.1.pth'
    torch.save(model.state_dict(), model_path)
    logger.info(f"Model saved to {model_path}")

if __name__ == '__main__':
    main()
