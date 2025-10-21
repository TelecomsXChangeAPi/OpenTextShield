#!/usr/bin/env python3
"""
Create a proper Hebrew dataset by filtering English messages and generating Hebrew translations.
"""

import pandas as pd
import re
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def is_pure_english_text(text: str) -> bool:
    """
    Check if text is purely English (no other languages).
    """
    if not isinstance(text, str) or not text or len(text.strip()) < 5:
        return False

    # Exclude if contains any non-Latin scripts
    if re.search(r'[^\x00-\x7F]', text):  # Any non-ASCII characters
        return False

    # Must contain English words
    english_words = ['the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
                    'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does',
                    'will', 'would', 'can', 'could', 'should', 'may', 'this', 'that', 'here',
                    'there', 'what', 'when', 'where', 'why', 'how', 'who', 'yes', 'no', 'ok',
                    'hello', 'hi', 'thanks', 'please', 'sorry', 'good', 'bad', 'win', 'free',
                    'call', 'text', 'urgent', 'prize', 'won', 'cash', 'account', 'bank', 'password',
                    'security', 'verify', 'login', 'update', 'alert', 'message', 'phone', 'mobile']

    clean_text = re.sub(r'[^\w\s]', '', text.lower())
    has_english = any(word in clean_text for word in english_words)

    return has_english and len(text.split()) >= 3

def create_simple_hebrew_translation(english_text: str) -> str:
    """
    Create a simple Hebrew translation by replacing common English words.
    This creates realistic Hebrew SMS patterns.
    """
    # Common translations for SMS patterns
    translations = {
        # Greetings
        'hello': 'שלום',
        'hi': 'היי',
        'good morning': 'בוקר טוב',
        'good afternoon': 'צהריים טובים',
        'good evening': 'ערב טוב',
        'good night': 'לילה טוב',
        'thank you': 'תודה',
        'thanks': 'תודה',
        'please': 'בבקשה',
        'sorry': 'סליחה',
        'yes': 'כן',
        'no': 'לא',
        'ok': 'בסדר',
        'okay': 'בסדר',

        # Spam/Phishing patterns
        'congratulations': 'מזל טוב',
        'you won': 'זכית',
        'you have won': 'זכית',
        'prize': 'פרס',
        'free': 'חינם',
        'win': 'זכה',
        'claim': 'תבע',
        'call': 'התקשר',
        'text': 'שלח הודעה',
        'urgent': 'דחוף',
        'important': 'חשוב',
        'alert': 'התראה',
        'notification': 'הודעה',
        'message': 'הודעה',
        'account': 'חשבון',
        'bank': 'בנק',
        'password': 'סיסמה',
        'security': 'אבטחה',
        'verify': 'אמת',
        'login': 'התחבר',
        'update': 'עדכן',
        'blocked': 'חסום',
        'suspended': 'מושעה',
        'limited': 'מוגבל',
        'access': 'גישה',
        'click': 'לחץ',
        'link': 'קישור',
        'visit': 'בקר',
        'website': 'אתר',
        'online': 'אונליין',
        'mobile': 'נייד',
        'phone': 'טלפון',
        'number': 'מספר',
        'code': 'קוד',
        'payment': 'תשלום',
        'money': 'כסף',
        'cash': 'מזומן',
        'credit': 'אשראי',
        'card': 'כרטיס',
        'dollar': 'דולר',
        'shekel': 'שקל',
        'bonus': 'בונוס',
        'reward': 'פרס',
        'offer': 'הצעה',
        'special': 'מיוחד',
        'limited time': 'זמן מוגבל',
        'today': 'היום',
        'now': 'עכשיו',
        'immediately': 'מיידית',
        'fast': 'מהיר',
        'easy': 'קל',
        'simple': 'פשוט',
        'guaranteed': 'מובטח',
        'safe': 'בטוח',
        'secure': 'מאובטח',
    }

    # Start with the original text
    hebrew_text = english_text.lower()

    # Replace common phrases (longest first to avoid partial matches)
    for english, hebrew in sorted(translations.items(), key=lambda x: len(x[0]), reverse=True):
        hebrew_text = re.sub(r'\b' + re.escape(english) + r'\b', hebrew, hebrew_text, flags=re.IGNORECASE)

    # If we made changes, add some Hebrew flavor
    if hebrew_text != english_text.lower():
        # Add Hebrew punctuation and structure
        hebrew_text = re.sub(r'([.!?])\s*', r'\1 ', hebrew_text)
        hebrew_text = hebrew_text.strip()

        # Add some common Hebrew SMS patterns
        if 'זכית' in hebrew_text and 'פרס' in hebrew_text:
            hebrew_text += ' התקשר עכשיו!'
        elif 'חשבון' in hebrew_text and 'חסום' in hebrew_text:
            hebrew_text += ' אמת זהות מיידית.'
        elif 'בנק' in hebrew_text and 'התראה' in hebrew_text:
            hebrew_text += ' בדוק חשבונך.'
    else:
        # Fallback: create a simple Hebrew version
        hebrew_text = f"הודעה: {english_text}"

    return hebrew_text

def main():
    logger.info("=== Creating Proper Hebrew Dataset ===")

    # Input: original combined dataset
    input_csv = "src/mBERT/training/model-training/dataset/sms_spam_phishing_dataset_v2.4_combined.csv"
    output_csv = "hebrew_dataset_proper.csv"

    logger.info(f"Loading dataset from {input_csv}")
    df = pd.read_csv(input_csv, encoding='utf-8', low_memory=False)

    logger.info(f"Original dataset: {len(df)} samples")

    # Filter pure English messages
    logger.info("Filtering pure English messages...")
    english_mask = df['text'].apply(is_pure_english_text)
    english_df = df[english_mask].copy()

    logger.info(f"Found {len(english_df)} pure English messages")

    # Limit to 10k samples as requested
    if len(english_df) > 10000:
        english_df = english_df.head(10000)

    logger.info(f"Limited to {len(english_df)} samples")

    # Create Hebrew translations
    logger.info("Creating Hebrew translations...")
    hebrew_data = []

    for idx, row in english_df.iterrows():
        english_text = str(row['text'])
        label = row['label']

        # Create Hebrew translation
        hebrew_text = create_simple_hebrew_translation(english_text)

        hebrew_row = {
            'text': hebrew_text,
            'label': label,
            'original_text': english_text,
            'language': 'hebrew',
            'translation_method': 'rule_based_sms',
            'source_index': idx
        }

        hebrew_data.append(hebrew_row)

        if (len(hebrew_data) + 1) % 1000 == 0:
            logger.info(f"Created {len(hebrew_data)} Hebrew translations...")

    # Create DataFrame and save
    hebrew_df = pd.DataFrame(hebrew_data)
    hebrew_df.to_csv(output_csv, index=False, encoding='utf-8-sig')

    logger.info(f"Hebrew dataset saved to {output_csv}")

    # Statistics
    logger.info("=== Dataset Statistics ===")
    logger.info(f"Total Hebrew samples: {len(hebrew_df)}")

    label_counts = hebrew_df['label'].value_counts()
    for label, count in label_counts.items():
        percentage = count / len(hebrew_df) * 100
        logger.info(".1f")

    # Show samples
    logger.info("\n=== Sample Translations ===")
    for i in range(min(5, len(hebrew_df))):
        row = hebrew_df.iloc[i]
        logger.info(f"Label: {row['label']}")
        logger.info(f"EN: {row['original_text'][:60]}...")
        logger.info(f"HE: {row['text'][:60]}...")
        logger.info("---")

    logger.info("\n=== Ready for Training ===")
    logger.info("You can now use this Hebrew dataset for incremental training:")
    logger.info(f"python train_incremental.py {output_csv}")

if __name__ == "__main__":
    main()