#!/usr/bin/env python3
"""
Translate English dataset to Hebrew using Google Translate API.
"""

import pandas as pd
import time
from pathlib import Path
from typing import List
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from googletrans import Translator
    GOOGLETRANS_AVAILABLE = True
    logger.info("googletrans library loaded successfully")
except ImportError as e:
    GOOGLETRANS_AVAILABLE = False
    logger.warning(f"googletrans not available: {e}. Install with: pip install googletrans==4.0.0rc1")

class HebrewTranslator:
    def __init__(self):
        if not GOOGLETRANS_AVAILABLE:
            raise ImportError("googletrans library is required for translation")

        self.translator = Translator()
        self.max_retries = 3
        self.retry_delay = 2  # seconds

    def translate_text(self, text: str) -> str:
        """Translate a single text to Hebrew with retry logic."""
        for attempt in range(self.max_retries):
            try:
                # Clean the text a bit
                clean_text = str(text).strip()
                if len(clean_text) < 3:
                    return clean_text

                # Translate to Hebrew
                translation = self.translator.translate(clean_text, src='en', dest='he')
                translated_text = translation.text

                # Add small delay to avoid rate limiting
                time.sleep(0.5)

                return translated_text

            except Exception as e:
                logger.warning(f"Translation attempt {attempt + 1} failed for text: {text[:50]}... Error: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))  # Exponential backoff
                else:
                    logger.error(f"Failed to translate after {self.max_retries} attempts: {text[:50]}...")
                    return f"[TRANSLATION_FAILED] {text}"

    def translate_batch(self, texts: List[str], batch_size: int = 50) -> List[str]:
        """Translate a batch of texts to Hebrew."""
        translated_texts = []

        logger.info(f"Starting translation of {len(texts)} texts in batches of {batch_size}")

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            batch_translations = []

            logger.info(f"Translating batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")

            for text in batch:
                translated = self.translate_text(text)
                batch_translations.append(translated)

            translated_texts.extend(batch_translations)

            # Longer delay between batches to avoid rate limiting
            if i + batch_size < len(texts):
                time.sleep(5)
                logger.info(f"Completed {min(i+batch_size, len(texts))}/{len(texts)} translations")

        return translated_texts

def main():
    # Configuration
    input_csv = "hebrew_translated_dataset.csv"  # The placeholder dataset we created
    output_csv = "hebrew_translated_real.csv"    # Final dataset with real translations

    logger.info("=== Hebrew Translation Script ===")

    if not GOOGLETRANS_AVAILABLE:
        logger.error("googletrans library not installed. Please install with:")
        logger.error("pip install googletrans==4.0.0rc1")
        return

    # Load the dataset
    logger.info(f"Loading dataset from {input_csv}")
    df = pd.read_csv(input_csv)

    logger.info(f"Dataset loaded: {len(df)} samples")

    # Initialize translator
    translator = HebrewTranslator()

    # Get texts to translate (the original English texts)
    english_texts = df['original_text'].tolist()

    # Translate to Hebrew
    logger.info("Starting Hebrew translation...")
    hebrew_texts = translator.translate_batch(english_texts)

    # Update the dataframe
    df['text'] = hebrew_texts
    df['translation_status'] = 'completed'
    df['translated_at'] = pd.Timestamp.now().isoformat()

    # Save the translated dataset
    df.to_csv(output_csv, index=False)
    logger.info(f"Translated dataset saved to {output_csv}")

    # Show statistics
    failed_translations = sum(1 for text in hebrew_texts if text.startswith('[TRANSLATION_FAILED]'))
    logger.info(f"Translation completed: {len(hebrew_texts) - failed_translations}/{len(hebrew_texts)} successful")

    # Show sample translations
    logger.info("\n=== Sample Translations ===")
    for i in range(min(5, len(df))):
        original = df.iloc[i]['original_text']
        translated = df.iloc[i]['text']
        label = df.iloc[i]['label']
        logger.info(f"Label: {label}")
        logger.info(f"EN: {original[:80]}...")
        logger.info(f"HE: {translated[:80]}...")
        logger.info("---")

    logger.info("\n=== Next Steps ===")
    logger.info("1. Review the translated dataset for quality")
    logger.info("2. Use the Hebrew dataset for incremental training")
    logger.info("3. Test the improved Hebrew model performance")

if __name__ == "__main__":
    main()