import pandas as pd

# Load the dataset
df = pd.read_csv('translated_dataset1_6_indonesian.csv', usecols=['text', 'label'])

# Drop rows with any missing values in 'text' or 'label' columns
df.dropna(subset=['text', 'label'], inplace=True)

# Filter out rows where 'label' is not one of the desired values
df = df[df['label'].isin(['ham', 'spam', 'phishing'])]

# Save the cleaned data to a new CSV file
df.to_csv('cleaned_translated_dataset1_6_indonesian.csv', index=False)

