#!/usr/bin/env python3
"""
Create Hebrew dataset by translating English messages from the combined dataset.
"""

import pandas as pd
import re
from typing import List, Tuple
import time
from pathlib import Path

def is_english_text(text) -> bool:
    """
    Check if text is primarily English.
    Uses heuristics: English words, Latin characters, common English patterns.
    """
    if not isinstance(text, str) or not text or len(text.strip()) < 3:
        return False

    # Skip if it contains Hebrew characters
    if re.search(r'[\u0590-\u05FF]', text):
        return False

    # Skip if it contains Arabic characters
    if re.search(r'[\u0600-\u06FF]', text):
        return False

    # Skip if it contains Cyrillic characters
    if re.search(r'[\u0400-\u04FF]', text):
        return False

    # Skip if it contains other non-Latin scripts (Devanagari, etc.)
    if re.search(r'[\u0980-\u09FF\u0A80-\u0AFF\u0B80-\u0BFF]', text):
        return False

    # Remove URLs and numbers for language detection
    clean_text = re.sub(r'http[s]?://\S+', '', text)
    clean_text = re.sub(r'\d+', '', clean_text)
    clean_text = re.sub(r'[^\w\s]', '', clean_text)  # Remove punctuation

    if len(clean_text.strip()) < 3:
        return False

    # Count English vs non-English characters
    english_chars = sum(1 for c in clean_text if c.isascii() and c.isalpha())
    total_alpha = sum(1 for c in clean_text if c.isalpha())

    if total_alpha == 0:
        return False

    # Must be at least 90% English characters
    english_ratio = english_chars / total_alpha

    # Must contain common English words (expanded list)
    has_english_words = bool(re.search(r'\b(the|and|or|but|in|on|at|to|for|of|with|by|an|a|is|are|was|were|be|been|being|have|has|had|do|does|did|will|would|can|could|should|may|might|must|shall|this|that|these|those|here|there|where|when|why|how|what|which|who|all|some|any|every|most|many|much|few|little|first|last|next|new|old|good|bad|big|small|long|short|right|wrong|true|false|yes|no|okay|thanks|please|sorry|hello|hi|bye|goodbye|morning|afternoon|evening|night|today|tomorrow|yesterday|now|then|soon|later|before|after|up|down|left|right|in|out|on|off|open|close|win|prize|free|call|text|urgent|congratulations|won|cash|reward|bonus|account|bank|password|security|verify|login|update|suspended|blocked|alert|suspicious|activity|message|sms|mobile|phone|number|code|link|click|visit|website|online|internet|email|mail|send|receive|payment|money|dollar|pound|euro|credit|card|paypal|amazon|facebook|whatsapp|instagram|twitter|linkedin|youtube|google|apple|microsoft|samsung|android|ios|windows|linux|computer|laptop|phone|tablet|wifi|internet|web|site|page|browser|download|upload|file|photo|video|music|game|app|application|software|program|system|network|server|cloud|data|information|user|customer|client|service|support|help|contact|address|location|time|date|day|week|month|year|hour|minute|second)\b', clean_text.lower()))

    return english_ratio > 0.9 and has_english_words

def filter_english_messages(csv_path: str, max_samples: int = 15000) -> pd.DataFrame:
    """
    Filter English messages from the combined dataset.
    """
    print(f"Loading dataset from {csv_path}...")
    df = pd.read_csv(csv_path)

    print(f"Original dataset size: {len(df)} samples")

    # Filter English messages
    english_mask = df['text'].apply(is_english_text)
    english_df = df[english_mask].copy()

    print(f"English messages found: {len(english_df)} samples")

    # Limit to max_samples
    if len(english_df) > max_samples:
        english_df = english_df.head(max_samples)
        print(f"Limited to {max_samples} samples")

    return english_df

def translate_to_hebrew_batch(texts: List[str], batch_size: int = 10) -> List[str]:
    """
    Translate English texts to Hebrew using Google Translate API.
    Note: This is a placeholder - in practice you'd use googletrans or similar.
    """
    hebrew_texts = []

    print(f"Translating {len(texts)} texts to Hebrew in batches of {batch_size}...")

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        batch_translations = []

        for text in batch:
            # Placeholder for translation - in real implementation use:
            # from googletrans import Translator
            # translator = Translator()
            # translated = translator.translate(text, src='en', dest='he').text

            # For now, just mark as placeholder
            translated = f"[HEBREW_TRANSLATION_PLACEHOLDER] {text}"
            batch_translations.append(translated)

            # Add small delay to avoid rate limiting
            time.sleep(0.1)

        hebrew_texts.extend(batch_translations)
        print(f"Translated {min(i+batch_size, len(texts))}/{len(texts)} texts...")

    return hebrew_texts

def create_hebrew_dataset(english_df: pd.DataFrame, output_path: str) -> None:
    """
    Create Hebrew dataset by translating English messages.
    """
    print("Creating Hebrew translations...")

    # Get English texts
    english_texts = english_df['text'].tolist()

    # Translate to Hebrew (placeholder implementation)
    print("⚠️  Note: This script contains placeholder translation logic.")
    print("For actual translation, you would need to implement Google Translate API calls.")
    print("Creating sample Hebrew dataset with placeholder translations...")

    # Create Hebrew dataset with placeholders for now
    hebrew_df = english_df.copy()
    hebrew_df['original_text'] = hebrew_df['text']
    hebrew_df['language'] = 'hebrew'
    hebrew_df['translation_method'] = 'google_translate_placeholder'

    # Add Hebrew placeholder translations
    hebrew_translations = []
    for text in english_texts:
        # Create a simple Hebrew-like placeholder
        hebrew_placeholder = f"עברית: {text[:50]}..."
        hebrew_translations.append(hebrew_placeholder)

    hebrew_df['text'] = hebrew_translations

    # Save to CSV
    hebrew_df.to_csv(output_path, index=False)
    print(f"Hebrew dataset saved to {output_path}")
    print(f"Dataset size: {len(hebrew_df)} samples")

    return hebrew_df

def main():
    # Configuration
    input_csv = "src/mBERT/training/model-training/dataset/sms_spam_phishing_dataset_v2.4_combined.csv"
    output_csv = "hebrew_translated_dataset.csv"
    max_english_samples = 12000  # Target for >10k as requested

    print("=== Creating Hebrew Dataset from English Translations ===")

    # Step 1: Filter English messages
    english_df = filter_english_messages(input_csv, max_english_samples)

    # Step 2: Create Hebrew translations
    hebrew_df = create_hebrew_dataset(english_df, output_csv)

    # Step 3: Show statistics
    print("\n=== Dataset Statistics ===")
    print(f"Total Hebrew samples: {len(hebrew_df)}")
    label_counts = hebrew_df['label'].value_counts()
    print("Label distribution:")
    for label, count in label_counts.items():
        print(f"  {label}: {count} ({count/len(hebrew_df)*100:.1f}%)")

    print("\n=== Next Steps ===")
    print("1. Implement actual Google Translate API calls in translate_to_hebrew_batch()")
    print("2. Run the translation to get real Hebrew text")
    print("3. Use the Hebrew dataset for incremental training")
    print("4. Test the improved Hebrew model performance")

if __name__ == "__main__":
    main()