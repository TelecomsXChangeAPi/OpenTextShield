import fasttext
import pandas as pd


# Try different encodings if UTF-8 does not work
try:
    data = pd.read_csv('sms_spam_phishing_dataset.csv', encoding='utf-8')
except UnicodeDecodeError:
    data = pd.read_csv('sms_spam_phishing_dataset.csv', encoding='ISO-8859-1')  # Try latin1 encoding


# Preprocess data: format as fastText expects (each line: "__label__<label> <text>")
data['ft_format'] = data.apply(lambda row: f'__label__{row["Label"]} {row["Message"]}', axis=1)

# Save preprocessed data
data['ft_format'].to_csv('ft_data.txt', index=False, header=False)

# Train a supervised model
model = fasttext.train_supervised(input='ft_data.txt', epoch=25, lr=1.0, wordNgrams=2)

# Save the model
model.save_model('ots_sms_model_v1.bin')

