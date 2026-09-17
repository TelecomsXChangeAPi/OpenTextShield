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


def _script(ch: str) -> str:
    """First word of the Unicode name ("LATIN", "CYRILLIC", "GREEK", ...)."""
    try:
        return unicodedata.name(ch).split(" ", 1)[0]
    except ValueError:
        return ""

class EnhancedPreprocessor:
    """Enhanced text preprocessing for adversarial robustness."""

    # Zero-width / invisible formatting characters used in obfuscation attacks.
    # Inserting these between letters ("V​e​r​i​f​y")
    # breaks keywords without changing how the text looks to a human.
    INVISIBLE_CHARS = frozenset({
        "\u200B",  # U+200B zero-width space
        "\u200C",  # U+200C zero-width non-joiner
        "\u200D",  # U+200D zero-width joiner
        "\u200E",  # U+200E left-to-right mark
        "\u200F",  # U+200F right-to-left mark
        "\u2060",  # U+2060 word joiner
        "\u2061",  # U+2061 function application (invisible math)
        "\u2062",  # U+2062 invisible times
        "\u2063",  # U+2063 invisible separator
        "\u2064",  # U+2064 invisible plus
        "\uFEFF",  # U+FEFF zero-width no-break space / BOM
        "\u00AD",  # U+00AD soft hyphen
        "\u180E",  # U+180E Mongolian vowel separator
        "\u034F",  # U+034F combining grapheme joiner
        "\u061C",  # U+061C Arabic letter mark
    })

    # Cyrillic and Greek letters that render like Latin ones. Folded only where
    # they are plausibly a spoof (see normalize_unicode), never in real
    # Russian, Ukrainian, Bulgarian or Greek text.
    CONFUSABLES = {
        # Cyrillic lowercase
        "а": "a", "с": "c", "ԁ": "d", "е": "e", "һ": "h", "і": "i", "ј": "j",
        "ӏ": "l", "о": "o", "р": "p", "ԛ": "q", "ѕ": "s", "ԝ": "w", "х": "x",
        "у": "y",
        # Cyrillic uppercase
        "А": "A", "В": "B", "С": "C", "Е": "E", "Н": "H", "І": "I", "Ӏ": "I",
        "Ј": "J", "К": "K", "М": "M", "О": "O", "Р": "P", "Ԛ": "Q", "Ѕ": "S",
        "Т": "T", "Ԝ": "W", "Х": "X", "Ү": "Y",
        # Greek lowercase
        "α": "a", "ι": "i", "κ": "k", "ν": "v", "ο": "o", "ρ": "p", "τ": "t",
        "υ": "u", "χ": "x",
        # Greek uppercase
        "Α": "A", "Β": "B", "Ε": "E", "Ζ": "Z", "Η": "H", "Ι": "I", "Κ": "K",
        "Μ": "M", "Ν": "N", "Ο": "O", "Ρ": "P", "Τ": "T", "Υ": "Y", "Χ": "X",
    }

    def __init__(self):
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
        """Undo text obfuscation without damaging real non-Latin text.

        1. NFC, then strip zero-width / invisible characters that attackers
           insert between letters ("V​e​r​i​f​y" with U+200B).
        2. Fold compatibility forms of letters and digits to ASCII, so
           fullwidth "ＰａｙＰａｌ" and math-bold "𝐏𝐚𝐲𝐏𝐚𝐥" read as "PayPal".
           Fullwidth punctuation is folded only in mostly-Latin messages:
           there it disguises links ("ｐａｙｐａｌ．ｃｏｍ"), while CJK text
           uses fullwidth commas and colons natively.
        3. Fold Cyrillic/Greek look-alikes only inside a spoof: a word that
           mixes them with Latin letters ("РayРаl"), or a word made entirely of
           look-alikes in a mostly-Latin message ("Log in to раураӏ"). Real
           Russian or Greek sentences are returned unchanged.
        """
        text = unicodedata.normalize("NFC", text)
        text = "".join(ch for ch in text if ch not in self.INVISIBLE_CHARS)
        text = "".join(self._fold_compat_alnum(ch) for ch in text)

        letters = [ch for ch in text if ch.isalpha()]
        latin = sum(1 for ch in letters if _script(ch) == "LATIN")
        mostly_latin = bool(letters) and latin * 2 > len(letters)
        if mostly_latin:
            text = "".join(
                unicodedata.normalize("NFKC", ch) if "！" <= ch <= "～" or ch == "　" else ch
                for ch in text
            )
        return re.sub(r"\S+", lambda m: self._fold_spoofed_word(m.group(), mostly_latin), text)

    @staticmethod
    def _fold_compat_alnum(ch: str) -> str:
        if ch.isascii() or unicodedata.category(ch)[0] not in "LN":
            return ch
        folded = unicodedata.normalize("NFKC", ch)
        return folded if len(folded) == 1 and folded.isascii() and folded.isalnum() else ch

    def _fold_spoofed_word(self, word: str, mostly_latin: bool) -> str:
        foreign = [ch for ch in word if ch.isalpha() and _script(ch) in ("CYRILLIC", "GREEK")]
        if not foreign or any(ch not in self.CONFUSABLES for ch in foreign):
            return word
        has_latin = any(ch.isalpha() and _script(ch) == "LATIN" for ch in word)
        if has_latin or mostly_latin:
            return "".join(self.CONFUSABLES.get(ch, ch) for ch in word)
        return word

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