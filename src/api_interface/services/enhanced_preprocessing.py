"""
Enhanced Text Preprocessing for OpenTextShield

Implements robust preprocessing to handle adversarial inputs:
- Unicode normalization
- Homoglyph detection and normalization
- Emoji and special character handling
- URL parsing and feature extraction
"""

import re
import unicodedata
from typing import Dict, List, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedPreprocessor:
    """Enhanced text preprocessing for adversarial robustness."""

    # Zero-width / invisible formatting characters used in obfuscation attacks.
    # Inserting these between letters ("V​e​r​i​f​y")
    # breaks keywords without changing how the text looks to a human.
    INVISIBLE_CHARS = frozenset({
        "​",  # zero-width space
        "‌",  # zero-width non-joiner
        "‍",  # zero-width joiner
        "‎",  # left-to-right mark
        "‏",  # right-to-left mark
        "⁠",  # word joiner
        "⁡", "⁢", "⁣", "⁤",  # invisible math operators
        "﻿",  # zero-width no-break space / BOM
        "­",  # soft hyphen
        "᠎",  # Mongolian vowel separator
        "͏",  # combining grapheme joiner
        "؜",  # Arabic letter mark
    })

    def __init__(self):
        # Homoglyph normalization mappings
        self.homoglyph_map = {
            # Cyrillic to Latin
            'а': 'a', 'е': 'e', 'і': 'i', 'о': 'o', 'р': 'p', 'с': 'c', 'у': 'y', 'х': 'x',
            # Greek to Latin
            'α': 'a', 'ε': 'e', 'ι': 'i', 'ο': 'o', 'υ': 'u',
            # Other common homoglyphs
            'і': 'i', 'а': 'a', 'е': 'e', 'о': 'o', 'р': 'p', 'с': 'c', 'у': 'u', 'х': 'x',
            'а': 'a', 'е': 'e', 'і': 'i', 'ο': 'o', 'р': 'p', 'с': 'c', 'υ': 'u', 'х': 'x'
        }

        # Suspicious patterns
        self.suspicious_patterns = [
            r'\b(?:bit\.ly|tinyurl\.com|goo\.gl|t\.co|ow\.ly)\b',  # URL shorteners
            r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b',  # IP addresses
            r'\b(?:paypal|bank|secure|login|verify|account|update)\b',  # Suspicious keywords
            r'[^\x00-\x7F]{3,}',  # Multiple non-ASCII characters
            r'\b\d{4,}\b',  # Long numbers (potentially fake account numbers)
        ]

        # Emoji patterns
        self.emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F1E0-\U0001F1FF"  # flags (iOS)
            "\U00002700-\U000027BF"  # dingbats
            "\U0001f926-\U0001f937"  # gestures
            "\U00010000-\U0010ffff"  # other unicode
            "\u2640-\u2642"  # gender symbols
            "\u2600-\u2B55"  # misc symbols
            "\u200d"  # zero width joiner
            "\u23cf"  # eject symbol
            "\u23e9"  # fast forward
            "\u231a"  # watch
            "\ufe0f"  # variation selector
            "\u3030"  # wavy dash
            "]+",
            flags=re.UNICODE
        )

    def normalize_unicode(self, text: str) -> str:
        """Normalize Unicode characters and handle homoglyphs.

        Also strips zero-width / invisible formatting characters that attackers
        insert between letters to break up keywords (e.g. "V​e​r​i​f​y" with
        U+200B). These are removed before homoglyph mapping so an obfuscated
        word collapses back to its real tokens for the model.
        """
        # NFC normalization
        text = unicodedata.normalize('NFC', text)

        # Strip zero-width and invisible formatting characters. NFC does not
        # remove these, so a zero-width-joined string survives tokenisation as
        # a wall of [UNK]s otherwise.
        text = ''.join(ch for ch in text if ch not in self.INVISIBLE_CHARS)

        # Replace homoglyphs
        normalized = []
        for char in text:
            normalized_char = self.homoglyph_map.get(char.lower(), char)
            # Preserve case if original was uppercase
            if char.isupper() and normalized_char != char:
                normalized_char = normalized_char.upper()
            normalized.append(normalized_char)

        return ''.join(normalized)

    def extract_url_features(self, text: str) -> Dict[str, int]:
        """Extract URL-related features."""
        features = {
            'url_count': 0,
            'shortener_count': 0,
            'suspicious_domain_count': 0,
            'ip_address_count': 0
        }

        # Find URLs
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        urls = re.findall(url_pattern, text)
        features['url_count'] = len(urls)

        # Check for URL shorteners
        shorteners = ['bit.ly', 'tinyurl.com', 'goo.gl', 't.co', 'ow.ly', 'is.gd', 'buff.ly']
        for url in urls:
            for shortener in shorteners:
                if shortener in url.lower():
                    features['shortener_count'] += 1
                    break

        # Check for suspicious domains
        suspicious_domains = ['paypal-secure', 'bank-login', 'secure-bank', 'account-verify', 'login-secure']
        for url in urls:
            for domain in suspicious_domains:
                if domain in url.lower():
                    features['suspicious_domain_count'] += 1
                    break

        # Check for IP addresses
        ip_pattern = r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b'
        features['ip_address_count'] = len(re.findall(ip_pattern, text))

        return features

    def extract_text_features(self, text: str) -> Dict[str, int]:
        """Extract text-based features."""
        features = {
            'emoji_count': len(self.emoji_pattern.findall(text)),
            'special_char_count': sum(1 for c in text if not c.isalnum() and not c.isspace()),
            'uppercase_ratio': sum(1 for c in text if c.isupper()) / max(len(text), 1),
            'digit_ratio': sum(1 for c in text if c.isdigit()) / max(len(text), 1),
            'suspicious_keyword_count': 0
        }

        # Count suspicious keywords
        suspicious_words = ['urgent', 'alert', 'warning', 'suspended', 'locked', 'verify', 'confirm', 'login', 'password', 'account', 'security', 'bank', 'paypal', 'credit', 'card']
        text_lower = text.lower()
        for word in suspicious_words:
            features['suspicious_keyword_count'] += text_lower.count(word)

        return features

    def preprocess_text(self, text: str) -> Tuple[str, Dict[str, int]]:
        """Main preprocessing function."""
        # Unicode normalization
        normalized_text = self.normalize_unicode(text)

        # Extract features
        url_features = self.extract_url_features(normalized_text)
        text_features = self.extract_text_features(normalized_text)

        # Combine features
        features = {**url_features, **text_features}

        # Clean text (optional - remove for now to preserve original for model)
        # cleaned_text = self.clean_text(normalized_text)

        return normalized_text, features

    def clean_text(self, text: str) -> str:
        """Clean text for model input."""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        # Handle emojis (replace with [EMOJI] token)
        text = self.emoji_pattern.sub('[EMOJI]', text)

        # Normalize case (optional - keep original case for now)
        # text = text.lower()

        return text

    def get_adversarial_score(self, text: str, features: Dict[str, int]) -> float:
        """Calculate adversarial risk score."""
        score = 0.0

        # URL-based risk
        if features['url_count'] > 0:
            score += 0.3
            if features['shortener_count'] > 0:
                score += 0.4
            if features['suspicious_domain_count'] > 0:
                score += 0.5
            if features['ip_address_count'] > 0:
                score += 0.6

        # Text-based risk
        if features['emoji_count'] > 2:
            score += 0.2
        if features['special_char_count'] / max(len(text), 1) > 0.1:
            score += 0.1
        if features['uppercase_ratio'] > 0.3:
            score += 0.1
        if features['suspicious_keyword_count'] > 3:
            score += 0.2

        # Unicode risk
        non_ascii_ratio = sum(1 for c in text if ord(c) > 127) / max(len(text), 1)
        if non_ascii_ratio > 0.1:
            score += 0.1

        return min(score, 1.0)  # Cap at 1.0

def test_preprocessor():
    """Test the enhanced preprocessor."""
    preprocessor = EnhancedPreprocessor()

    test_cases = [
        "Normal message",
        "🚨 ALERT: Account locked! Login now: 🔒 secure-bank.com",
        "URGENT: Your PayPal account is suspended. Verify now: paypal-secure-login.com",
        "Free money! Click: http://аррӏе.com",  # Homoglyph
        "Gewinne 1000€! Klicke hier: http://spam.de",
        "مرحبا، كيف حالك؟",  # Arabic
    ]

    for text in test_cases:
        processed_text, features = preprocessor.preprocess_text(text)
        risk_score = preprocessor.get_adversarial_score(text, features)

        print(f"Original: {text}")
        print(f"Processed: {processed_text}")
        print(f"Features: {features}")
        print(f"Risk Score: {risk_score:.3f}")
        print("-" * 50)

if __name__ == "__main__":
    test_preprocessor()