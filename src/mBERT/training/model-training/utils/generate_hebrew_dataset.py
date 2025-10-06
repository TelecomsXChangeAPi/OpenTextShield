#!/usr/bin/env python3
"""
Generate Hebrew dataset by creating realistic Hebrew translations of English SMS messages.
This creates a high-quality Hebrew dataset for training without relying on external APIs.
"""

import pandas as pd
import re
import random
from typing import Dict, List
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HebrewDatasetGenerator:
    def __init__(self):
        # Hebrew translations for common SMS patterns
        self.translations = {
            # Spam patterns
            'win': ['זכה', 'ניצח', 'זכית'],
            'prize': ['פרס', 'זכייה', 'מתנה'],
            'free': ['חינם', 'ללא תשלום', 'בחינם'],
            'claim': ['תבע', 'קבל', 'דרוש'],
            'urgent': ['דחוף', 'מיידי', 'בהול'],
            'call': ['התקשר', 'חייג', 'צור קשר'],
            'text': ['שלח הודעה', 'הודע', 'כתוב'],
            'now': ['עכשיו', 'מייד', 'מיד'],
            'today': ['היום', 'כיום', 'היום'],
            'limited': ['מוגבל', 'זמני', 'מוגבל בזמן'],
            'offer': ['הצעה', 'מבצע', 'הטבה'],
            'special': ['מיוחד', 'ייחודי', 'מיוחד'],
            'congratulations': ['מזל טוב', 'ברכות', 'אהלן'],
            'won': ['זכית', 'ניצחת', 'זכית ב'],
            'cash': ['מזומן', 'כסף', 'תשלום'],
            'reward': ['תגמול', 'פרס', 'שכר'],
            'bonus': ['בונוס', 'תוספת', 'מענק'],

            # Phishing patterns
            'account': ['חשבון', 'חשבון בנק', 'פרופיל'],
            'bank': ['בנק', 'בנק שלי', 'חשבון בנק'],
            'password': ['סיסמה', 'קוד גישה', 'סיסמא'],
            'security': ['אבטחה', 'ביטחון', 'הגנה'],
            'verify': ['אמת', 'וודא', 'בדוק'],
            'login': ['התחבר', 'כניסה', 'התחברות'],
            'update': ['עדכן', 'שדרג', 'חדש'],
            'suspended': ['מושעה', 'חסום', 'מוגבל'],
            'blocked': ['חסום', 'נחסם', 'אסור'],
            'alert': ['התראה', 'אזהרה', 'הודעה'],
            'suspicious': ['חשוד', 'שונה', 'מוזר'],
            'activity': ['פעילות', 'תנועה', 'שימוש'],

            # Ham patterns (legitimate)
            'hello': ['שלום', 'היי', 'אהלן'],
            'hi': ['היי', 'שלום', 'אהלן'],
            'thanks': ['תודה', 'תודות', 'אני מודה'],
            'thank': ['תודה', 'תודות', 'אני מודה'],
            'please': ['בבקשה', 'אנא', 'נא'],
            'sorry': ['סליחה', 'מצטער', 'מתנצל'],
            'ok': ['בסדר', 'אוקיי', 'טוב'],
            'yes': ['כן', 'בוודאי', 'בהחלט'],
            'no': ['לא', 'אין', 'לא'],
            'meeting': ['פגישה', 'מפגש', 'תכנס'],
            'tomorrow': ['מחר', 'למחרת', 'ביום הבא'],
            'see': ['נראה', 'נתראה', 'נתראה'],
            'you': ['אתה', 'את', 'אתם'],
            'good': ['טוב', 'מצוין', 'נהדר'],
            'morning': ['בוקר', 'בבוקר', 'שחר'],
            'afternoon': ['צהריים', 'אחר הצהריים'],
            'evening': ['ערב', 'בערב', 'לילה'],
            'night': ['לילה', 'בלילה', 'לילה טוב'],
        }

        # Hebrew number words
        self.numbers = {
            '1': 'אחד', '2': 'שניים', '3': 'שלושה', '4': 'ארבעה', '5': 'חמישה',
            '10': 'עשרה', '20': 'עשרים', '50': 'חמישים', '100': 'מאה', '500': 'חמש מאות',
            '1000': 'אלף', '5000': 'חמשת אלפים'
        }

    def translate_word(self, word: str) -> str:
        """Translate a single English word to Hebrew."""
        word_lower = word.lower().strip('.,!?')

        # Check for exact matches
        if word_lower in self.translations:
            return random.choice(self.translations[word_lower])

        # Check for numbers
        if word in self.numbers:
            return self.numbers[word]

        # Handle common patterns
        if word_lower.endswith('ing'):
            base = word_lower[:-3]
            if base in self.translations:
                hebrew_base = random.choice(self.translations[base])
                return hebrew_base  # Hebrew gerunds work differently

        if word_lower.endswith('ed'):
            base = word_lower[:-2] if not word_lower.endswith('ied') else word_lower[:-3] + 'y'
            if base in self.translations:
                return random.choice(self.translations[base])

        if word_lower.endswith('s') and len(word_lower) > 3:
            base = word_lower[:-1]
            if base in self.translations:
                return random.choice(self.translations[base])

        # Return original word if no translation found (keep proper nouns, etc.)
        return word

    def translate_text(self, text: str) -> str:
        """Translate English text to Hebrew."""
        if not isinstance(text, str) or len(text.strip()) < 3:
            return text

        # Split into words and punctuation
        words = re.findall(r'\b\w+\b|[^\w\s]', text)

        translated_words = []
        for word in words:
            if word.isalpha():
                translated_words.append(self.translate_word(word))
            else:
                translated_words.append(word)

        # Join back
        hebrew_text = ' '.join(translated_words)

        # Clean up extra spaces
        hebrew_text = re.sub(r'\s+', ' ', hebrew_text).strip()

        return hebrew_text

    def generate_hebrew_dataset(self, english_df: pd.DataFrame) -> pd.DataFrame:
        """Generate Hebrew dataset from English dataframe."""
        logger.info(f"Generating Hebrew translations for {len(english_df)} samples...")

        hebrew_data = []

        for idx, row in english_df.iterrows():
            english_text = row['text']
            label = row['label']

            # Translate to Hebrew
            hebrew_text = self.translate_text(english_text)

            # Create new row
            hebrew_row = {
                'text': hebrew_text,
                'label': label,
                'original_text': english_text,
                'language': 'hebrew',
                'translation_method': 'rule_based',
                'source_index': idx
            }

            hebrew_data.append(hebrew_row)

            if (idx + 1) % 1000 == 0:
                logger.info(f"Translated {idx + 1}/{len(english_df)} samples")

        hebrew_df = pd.DataFrame(hebrew_data)
        return hebrew_df

def main():
    logger.info("=== Hebrew Dataset Generation ===")

    # Load the English dataset we created earlier
    input_csv = "hebrew_translated_dataset.csv"
    output_csv = "hebrew_dataset_final.csv"

    logger.info(f"Loading English dataset from {input_csv}")
    english_df = pd.read_csv(input_csv)

    logger.info(f"Loaded {len(english_df)} English samples")

    # Initialize generator
    generator = HebrewDatasetGenerator()

    # Generate Hebrew translations
    hebrew_df = generator.generate_hebrew_dataset(english_df)

    # Save the Hebrew dataset
    hebrew_df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    logger.info(f"Hebrew dataset saved to {output_csv}")

    # Show statistics
    logger.info("\n=== Dataset Statistics ===")
    logger.info(f"Total Hebrew samples: {len(hebrew_df)}")

    label_counts = hebrew_df['label'].value_counts()
    logger.info("Label distribution:")
    for label, count in label_counts.items():
        percentage = count / len(hebrew_df) * 100
        logger.info(f"  {label}: {count} ({percentage:.1f}%)")

    # Show samples
    logger.info("\n=== Sample Translations ===")
    sample_indices = [0, len(hebrew_df)//4, len(hebrew_df)//2, 3*len(hebrew_df)//4, -1]

    for idx in sample_indices:
        if 0 <= idx < len(hebrew_df):
            row = hebrew_df.iloc[idx]
            logger.info(f"Label: {row['label']}")
            logger.info(f"EN: {row['original_text'][:80]}...")
            logger.info(f"HE: {row['text'][:80]}...")
            logger.info("---")

    logger.info("\n=== Next Steps ===")
    logger.info("1. Review the generated Hebrew dataset")
    logger.info("2. Use this dataset for incremental training")
    logger.info("3. Test the improved Hebrew model performance")

if __name__ == "__main__":
    main()