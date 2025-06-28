"""
This script is designed to leverage OpenAI's GPT models to translate an English dataset into different languages.
It addresses situations where there is a lack of organic datasets available in the target language. By utilizing
OpenAI's advanced language models, this script facilitates the creation of translated datasets that can be used
for various applications such as training machine learning models, data analysis, and more in languages where
data might be scarce or unavailable.
"""

import pandas as pd
import time
from openai import OpenAI
from tqdm import tqdm  # For displaying the progress bar

client = OpenAI(api_key='API-KEY-HERE')

def translate_text(text, target_language="es"):
    """
    Translate text to the target language using OpenAI's Translation API.
    """
    try:
        response = client.chat.completions.create(model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": f"You are a professional multilingual Translator, Translate the following text to Spanish language text, if text does not have a meaning in any language then leave it unchanged"},
            {"role": "user", "content": text}
        ])
        return response.choices[0].message.content
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


# Load your dataset
df = pd.read_csv('dataset/sms_spam_phishing_dataset_v1.6.csv')

# Prepare a list to hold translated rows
translated_rows = []

start_time = time.time()  # Start time measurement

# Iterate through the dataset with a progress bar
for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Translating"):
    translated_text = translate_text(row['text'], "es")
    if translated_text:
        translated_rows.append({'text': translated_text, 'label': row['label']})

end_time = time.time()  # End time measurement
total_time = end_time - start_time  # Calculate total time taken

# Convert translated rows to a DataFrame
translated_df = pd.DataFrame(translated_rows)

# Save the translated DataFrame to a new CSV file with UTF-8-BOM encoding
translated_df.to_csv('translated_dataset1_6_spanish.csv', index=False, encoding='utf-8-sig')

# Print the total time taken
print(f"Completed translation in {total_time:.2f} seconds.")
