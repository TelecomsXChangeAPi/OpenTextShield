"""
Adversarial Data Generation for OpenTextShield mBERT

Generates synthetic adversarial examples to improve model robustness against:
- Emoji and special characters
- Typos and obfuscation
- Encoding tricks (homoglyphs)
- Contextual ambiguity
"""

import pandas as pd
import random
import re
import unicodedata
from typing import List, Dict, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdversarialDataGenerator:
    """Generates adversarial examples for SMS classification robustness."""

    def __init__(self):
        # Homoglyph mapping for encoding tricks
        self.homoglyphs = {
            'a': ['а', 'а', 'α', 'а'],  # Cyrillic, Greek
            'e': ['е', 'е', 'ε'],
            'i': ['і', 'і', 'ι'],
            'o': ['о', 'ο', 'ο'],
            'u': ['υ'],
            'c': ['с'],
            'p': ['р'],
            'x': ['х'],
            'y': ['у'],
            '1': ['l', 'i', 'ı'],
            '0': ['o', 'ο'],
            '3': ['е'],
            '5': ['s'],
            '7': ['t'],
            '8': ['b']
        }

        # Common typos and substitutions
        self.typos = {
            'a': ['4', '@'],
            'e': ['3'],
            'i': ['1', '!'],
            'o': ['0'],
            's': ['5', '$'],
            't': ['7'],
            'l': ['1', '|'],
            'b': ['8'],
            'g': ['9'],
            'z': ['2']
        }

        # Emojis for injection
        self.emojis = ['🚨', '⚠️', '💰', '🎉', '🔒', '🛡️', '📱', '💳', '🏦', '🔗', '📧', '🎁', '💸', '⚡', '🔥']

        # Special characters
        self.special_chars = ['!', '@', '#', '$', '%', '^', '&', '*', '(', ')', '_', '+', '=', '[', ']', '{', '}', '|', '\\', ';', ':', '"', "'", '<', '>', ',', '.', '?', '/', '~', '`']

    def load_base_dataset(self, filepath: str) -> pd.DataFrame:
        """Load the base dataset for augmentation."""
        logger.info(f"Loading base dataset from {filepath}")
        df = pd.read_csv(filepath)
        logger.info(f"Loaded {len(df)} samples")
        return df

    def generate_homoglyph_variations(self, text: str, num_variations: int = 3) -> List[str]:
        """Generate homoglyph variations of text."""
        variations = []

        for _ in range(num_variations):
            variation = ""
            for char in text.lower():
                if char in self.homoglyphs and random.random() < 0.3:  # 30% chance to replace
                    variation += random.choice(self.homoglyphs[char])
                else:
                    variation += char
            variations.append(variation)

        return variations

    def generate_typo_variations(self, text: str, num_variations: int = 3) -> List[str]:
        """Generate typo variations."""
        variations = []

        for _ in range(num_variations):
            chars = list(text.lower())
            # Random substitutions
            for i in range(len(chars)):
                if chars[i] in self.typos and random.random() < 0.2:  # 20% chance
                    chars[i] = random.choice(self.typos[chars[i]])

            # Random insertions/deletions
            if random.random() < 0.1 and len(chars) > 5:  # 10% chance to delete
                del chars[random.randint(0, len(chars)-1)]
            elif random.random() < 0.1:  # 10% chance to insert
                insert_pos = random.randint(0, len(chars))
                chars.insert(insert_pos, random.choice(self.special_chars))

            variations.append(''.join(chars))

        return variations

    def generate_emoji_injections(self, text: str, num_variations: int = 3) -> List[str]:
        """Inject emojis and special characters."""
        variations = []

        for _ in range(num_variations):
            words = text.split()
            variation = []

            for word in words:
                variation.append(word)
                # Randomly insert emoji/special char after word
                if random.random() < 0.4:  # 40% chance
                    if random.random() < 0.5:
                        variation.append(random.choice(self.emojis))
                    else:
                        variation.append(random.choice(self.special_chars))

            variations.append(' '.join(variation))

        return variations

    def generate_contextual_ambiguity(self, text: str, label: str, num_variations: int = 3) -> List[Tuple[str, str]]:
        """Generate contextually ambiguous variations."""
        variations = []

        # Templates for ambiguity based on label
        if label == 'phishing':
            templates = [
                "Important: {text}",
                "ALERT: {text}",
                "Security: {text}",
                "Update: {text}",
                "Notice: {text}"
            ]
        elif label == 'spam':
            templates = [
                "WIN: {text}",
                "FREE: {text}",
                "PROMO: {text}",
                "DEAL: {text}",
                "OFFER: {text}"
            ]
        else:  # ham
            templates = [
                "Hey, {text}",
                "Hi, {text}",
                "Just wanted to say {text}",
                "FYI: {text}",
                "Reminder: {text}"
            ]

        for _ in range(num_variations):
            template = random.choice(templates)
            variation = template.format(text=text)
            variations.append((variation, label))

        return variations

    def generate_url_obfuscation(self, text: str, num_variations: int = 3) -> List[str]:
        """Generate URL obfuscation variations."""
        variations = []

        # Find URLs in text
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        urls = re.findall(url_pattern, text)

        if not urls:
            return [text] * num_variations  # No URLs to obfuscate

        for _ in range(num_variations):
            variation = text
            for url in urls:
                # Replace with obfuscated version
                if 'bit.ly' in url or 'tinyurl' in url or 'goo.gl' in url:
                    # Already obfuscated, keep as is
                    continue
                elif random.random() < 0.5:
                    # Replace with bit.ly style
                    variation = variation.replace(url, f'bit.ly/{random.randint(1000,9999)}')
                else:
                    # Replace with tinyurl style
                    variation = variation.replace(url, f'tinyurl.com/{random.randint(100,999)}')

            variations.append(variation)

        return variations

    def augment_dataset(self, df: pd.DataFrame, augmentation_factor: int = 5) -> pd.DataFrame:
        """Augment the dataset with adversarial examples."""
        logger.info(f"Starting dataset augmentation with factor {augmentation_factor}")

        augmented_data = []
        total_samples = len(df)

        for idx, row in df.iterrows():
            if idx % 100 == 0:
                logger.info(f"Processing sample {idx}/{total_samples}")

            text = str(row['text'])
            label = str(row['label'])

            # Keep original
            augmented_data.append({'text': text, 'label': label, 'augmentation_type': 'original'})

            # Generate adversarial variations
            for _ in range(augmentation_factor):
                augmentation_type = random.choice([
                    'homoglyph', 'typo', 'emoji', 'contextual', 'url_obfuscation'
                ])

                try:
                    if augmentation_type == 'homoglyph':
                        variations = self.generate_homoglyph_variations(text, 1)
                        for var in variations:
                            augmented_data.append({
                                'text': var,
                                'label': label,
                                'augmentation_type': 'homoglyph'
                            })

                    elif augmentation_type == 'typo':
                        variations = self.generate_typo_variations(text, 1)
                        for var in variations:
                            augmented_data.append({
                                'text': var,
                                'label': label,
                                'augmentation_type': 'typo'
                            })

                    elif augmentation_type == 'emoji':
                        variations = self.generate_emoji_injections(text, 1)
                        for var in variations:
                            augmented_data.append({
                                'text': var,
                                'label': label,
                                'augmentation_type': 'emoji'
                            })

                    elif augmentation_type == 'contextual':
                        variations = self.generate_contextual_ambiguity(text, label, 1)
                        for var, lbl in variations:
                            augmented_data.append({
                                'text': var,
                                'label': lbl,
                                'augmentation_type': 'contextual'
                            })

                    elif augmentation_type == 'url_obfuscation':
                        variations = self.generate_url_obfuscation(text, 1)
                        for var in variations:
                            augmented_data.append({
                                'text': var,
                                'label': label,
                                'augmentation_type': 'url_obfuscation'
                            })

                except Exception as e:
                    logger.warning(f"Error generating {augmentation_type} for text: {text[:50]}... Error: {e}")
                    continue

        augmented_df = pd.DataFrame(augmented_data)
        logger.info(f"Augmentation complete. Original: {total_samples}, Augmented: {len(augmented_df)}")
        return augmented_df

    def save_augmented_dataset(self, df: pd.DataFrame, output_path: str):
        """Save the augmented dataset."""
        df.to_csv(output_path, index=False)
        logger.info(f"Augmented dataset saved to {output_path}")

        # Print statistics
        print("\nAugmentation Statistics:")
        print(f"Total samples: {len(df)}")
        print("\nBy augmentation type:")
        print(df['augmentation_type'].value_counts())
        print("\nBy label:")
        print(df['label'].value_counts())

def main():
    generator = AdversarialDataGenerator()

    # Load base dataset
    base_df = generator.load_base_dataset('dataset/sms_spam_phishing_dataset_v2.4_balanced.csv')

    # Augment dataset
    augmented_df = generator.augment_dataset(base_df, augmentation_factor=3)

    # Save augmented dataset
    generator.save_augmented_dataset(augmented_df, 'src/mBERT/training/model-training/dataset/sms_spam_phishing_dataset_v2.4_adversarial_augmented.csv')

if __name__ == "__main__":
    main()