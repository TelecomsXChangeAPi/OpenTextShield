"""
This script is designed to leverage OpenAI's GPT models to translate an English dataset into different languages.
It addresses situations where there is a lack of organic datasets available in the target language. By utilizing
OpenAI's advanced language models, this script facilitates the creation of translated datasets that can be used
for various applications such as training machine learning models, data analysis, and more in languages where
data might be scarce or unavailable.
"""

import pandas as pd
from openai import OpenAI

client = OpenAI(api_key='API_KEY')

def translate_text(text, target_language="id"):  # Change default target language to Indonesian
    """
    Translate text to the target language using OpenAI's Translation API.
    """
    try:
        response = client.chat.completions.create(model="gpt-3.5-turbo",  # Use the latest suitable model for translation
        messages=[
            {"role": "system", "content": "Translate the following English text to Indonesian."},  # Specify Indonesian translation
            {"role": "user", "content": text}
        ])
        return response.choices[0].message.content
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

# Load your dataset
df = pd.read_csv('sms_spam_phishing_dataset_v1.6.csv')

# Prepare a list to hold translated rows
translated_rows = []

for index, row in df.iterrows():
    translated_text = translate_text(row['text'], "id")  # Ensure the target_language is set correctly
    if translated_text:
        translated_rows.append({'text': translated_text, 'label': row['label']})

# Convert translated rows to a DataFrame
translated_df = pd.DataFrame(translated_rows)

# Since you now want only the translated data in the output,
# we skip appending the original DataFrame and directly work with the translated DataFrame.

# Save the translated DataFrame to a new CSV file with UTF-8-BOM encoding
translated_df.to_csv('translated_dataset1_6_indonesian.csv', index=False, encoding='utf-8-sig')