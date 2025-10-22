#!/usr/bin/env python3
"""
Indonesian SMS Dataset Expansion Tool

This script expands the Indonesian SMS dataset from ~1,000 samples to 50,000+ samples
by generating synthetic messages based on patterns from existing data.
"""

import csv
import random
import re
from collections import defaultdict, Counter
import argparse

def load_indonesian_dataset(file_path):
    """Load the Indonesian dataset with proper parsing."""
    data = []
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if 'Kategori' in row and 'Pesan' in row:
                label = row['Kategori'].strip().lower()
                text = row['Pesan'].strip()

                # Map Indonesian labels to English
                if label == 'spam':
                    label = 'spam'
                elif label in ['ham', 'normal', 'legitimate']:
                    label = 'ham'
                elif label in ['phishing', 'fraud']:
                    label = 'phishing'
                else:
                    continue  # Skip unknown labels

                if text:  # Only add if text is not empty
                    data.append({'text': text, 'label': label})

    return data

def analyze_patterns(data):
    """Analyze patterns in the existing Indonesian SMS data."""
    patterns = {
        'greetings': [],
        'questions': [],
        'statements': [],
        'commands': [],
        'spam_templates': [],
        'ham_templates': []
    }

    # Indonesian common words and patterns
    indonesian_words = {
        'greetings': ['halo', 'hai', 'selamat', 'assalamualaikum', 'hi', 'hey'],
        'questions': ['apa', 'bagaimana', 'kenapa', 'kapan', 'dimana', 'siapa', 'berapa'],
        'responses': ['iya', 'ya', 'tidak', 'baik', 'bagus', 'sedih', 'senang'],
        'time_words': ['hari', 'minggu', 'bulan', 'tahun', 'kemarin', 'besok', 'sekarang'],
        'places': ['jakarta', 'bandung', 'surabaya', 'medan', 'makassar', 'jogja'],
        'money_terms': ['rupiah', 'rb', 'jt', 'juta', 'ribu', 'ratus'],
        'phone_terms': ['sms', 'telepon', 'nelp', 'hubungi', 'kontak']
    }

    spam_keywords = [
        'bonus', 'hadiah', 'menang', 'cek', 'klik', 'pin', 'kode', 'verifikasi',
        'pulsa', 'isi ulang', 'transfer', 'rekening', 'bank', 'promo', 'diskon'
    ]

    for item in data:
        text = item['text'].lower()
        label = item['label']

        # Categorize by structure
        if any(text.startswith(greet) for greet in indonesian_words['greetings']):
            patterns['greetings'].append(item)
        elif any(q in text for q in indonesian_words['questions']):
            patterns['questions'].append(item)
        elif text.endswith('!') or text.endswith('.'):
            patterns['statements'].append(item)
        elif any(cmd in text for cmd in ['silahkan', 'ayo', 'mari', 'bisa']):
            patterns['commands'].append(item)

        # Categorize by label
        if label == 'spam':
            patterns['spam_templates'].append(item)
        elif label == 'ham':
            patterns['ham_templates'].append(item)

    return patterns, indonesian_words, spam_keywords

def generate_ham_messages(patterns, indonesian_words, count):
    """Generate synthetic ham messages."""
    messages = []

    # Templates for ham messages
    ham_templates = [
        "Halo, {greeting} hari ini",
        "Bagaimana kabarmu? Semoga {response}",
        "Saya sedang di {place}, cuacanya {weather}",
        "Apakah kamu {activity} besok?",
        "Terima kasih atas {noun} yang diberikan",
        "Sampai jumpa di {place} nanti",
        "Maaf saya tidak bisa {activity} hari ini",
        "Selamat {occasion} untukmu",
        "Kapan kita bisa {activity} bersama?",
        "Saya sedang {activity} sekarang"
    ]

    # Fillers for templates
    fillers = {
        'greeting': ['selamat pagi', 'selamat siang', 'selamat malam', 'apa kabar'],
        'response': ['baik-baik saja', 'dalam keadaan sehat', 'selalu bahagia'],
        'place': ['rumah', 'kantor', 'sekolah', 'mall', 'restoran', 'bandara'],
        'weather': ['cerah', 'hujan', 'mendung', 'panas', 'dingin'],
        'activity': ['makan', 'belajar', 'bekerja', 'bermain', 'berjalan', 'berbicara'],
        'noun': ['bantuan', 'hadiah', 'peringatan', 'informasi', 'surat'],
        'occasion': ['ulang tahun', 'hari raya', 'natal', 'tahun baru', 'lebaran'],
    }

    for _ in range(count):
        template = random.choice(ham_templates)
        message = template

        # Fill in placeholders
        for key, values in fillers.items():
            placeholder = f"{{{key}}}"
            if placeholder in message:
                message = message.replace(placeholder, random.choice(values), 1)

        messages.append({'text': message, 'label': 'ham'})

    return messages

def generate_spam_messages(patterns, indonesian_words, spam_keywords, count):
    """Generate synthetic spam messages."""
    messages = []

    # Spam templates
    spam_templates = [
        "SELAMAT! Nomor Anda mendapat {prize} Rp{amount} dari {company}",
        "Bonus pulsa {amount} untuk pengisian minimal Rp{min_amount}",
        "Kode verifikasi: {code} untuk {purpose}",
        "Transfer ke rekening {bank} a/n {name} No. Rek {account}",
        "Klik link berikut untuk klaim hadiah: {url}",
        "PIN Anda: {pin} untuk cek hadiah {company}",
        "Promo spesial: Diskon {discount}% untuk {product}",
        "Hubungi {number} untuk info lebih lanjut",
        "Pesan resmi dari {company}: Anda menang {prize}"
    ]

    # Fillers for spam templates
    spam_fillers = {
        'prize': ['hadiah', 'bonus', 'reward', 'prize', 'doorprize'],
        'amount': ['50000', '100000', '250000', '500000', '1000000'],
        'company': ['Telkomsel', 'Indosat', 'XL', 'Tri', 'Smartfren', 'Bank BRI', 'Bank BCA'],
        'code': ['123456', 'ABCDEF', '789XYZ', '456QWE', '987RTY'],
        'purpose': ['aktivasi akun', 'verifikasi identitas', 'konfirmasi transfer'],
        'bank': ['BCA', 'BNI', 'Mandiri', 'BRI', 'CIMB'],
        'name': ['Ahmad', 'Siti', 'Budi', 'Rina', 'Joko', 'Sari'],
        'account': ['1234567890', '9876543210', '5555666677'],
        'url': ['bit.ly/abc123', 'tinyurl.com/def456', 'short.link/ghi789'],
        'pin': ['A1B2C3', 'X9Y8Z7', 'P4Q5R6'],
        'discount': ['10', '20', '30', '50', '70'],
        'product': ['pulsa', 'paket data', 'smartphone', 'laptop'],
        'number': ['021-1234567', '08123456789', '0800123456']
    }

    for _ in range(count):
        template = random.choice(spam_templates)
        message = template

        # Fill in placeholders
        for key, values in spam_fillers.items():
            placeholder = f"{{{key}}}"
            if placeholder in message:
                message = message.replace(placeholder, random.choice(values), 1)

        messages.append({'text': message, 'label': 'spam'})

    return messages

def generate_phishing_messages(patterns, indonesian_words, count):
    """Generate synthetic phishing messages."""
    messages = []

    # Phishing templates (more deceptive than spam)
    phishing_templates = [
        "PENTING: Akun {service} Anda akan diblokir dalam 24 jam. Verifikasi di {url}",
        "PERINGATAN KEAMANAN: Aktivitas mencurigakan terdeteksi. Konfirmasi identitas Anda",
        "Bank {bank} - Verifikasi transfer Rp{amount}. Kode: {code}",
        "Microsoft: Akun email Anda perlu diverifikasi. Masuk ke {url}",
        "WhatsApp: Kode verifikasi Anda adalah {code}. Jangan bagikan ke siapa pun",
        "URGENT: Paket Anda tertahan. Bayar biaya pengiriman Rp{fee} ke rekening {account}",
        "PAYPAL: Konfirmasi pembayaran sebesar ${usd_amount}. Klik untuk verifikasi",
        "DHL: Tracking nomor {tracking} - Masalah pengiriman. Hubungi {number}"
    ]

    # Fillers for phishing templates
    phishing_fillers = {
        'service': ['Gmail', 'Facebook', 'Instagram', 'Twitter', 'WhatsApp', 'Telegram'],
        'url': ['secure-login.com/verify', 'account-security.net', 'login-secure.org'],
        'bank': ['BCA', 'Mandiri', 'BRI', 'BNI', 'CIMB Niaga'],
        'amount': ['500000', '1000000', '2500000', '5000000'],
        'code': ['123-456', '789-012', '345-678', '901-234'],
        'fee': ['50000', '75000', '100000', '150000'],
        'account': ['1234567890', '0987654321'],
        'usd_amount': ['50', '100', '250', '500'],
        'tracking': ['TH123456789', 'ID987654321', 'SG555666777'],
        'number': ['021-88990011', '0800-123456', '14045']
    }

    for _ in range(count):
        template = random.choice(phishing_templates)
        message = template

        # Fill in placeholders
        for key, values in phishing_fillers.items():
            placeholder = f"{{{key}}}"
            if placeholder in message:
                message = message.replace(placeholder, random.choice(values), 1)

        messages.append({'text': message, 'label': 'phishing'})

    return messages

def create_variations(messages, count_per_message=3):
    """Create variations of existing messages."""
    variations = []

    for item in messages:
        base_text = item['text']
        label = item['label']

        # Create variations by replacing words with synonyms
        synonyms = {
            'halo': ['hai', 'hi', 'hey'],
            'selamat': ['salam', 'greetings'],
            'terima kasih': ['thanks', 'makasih', 'terimakasih'],
            'maaf': ['sorry', 'maafkan'],
            'bagaimana': ['gimana', 'how'],
            'kapan': ['when', 'kapankah'],
            'dimana': ['where', 'dimanakah'],
            'siapa': ['who', 'siapakah'],
            'apa': ['what', 'apakah'],
            'iya': ['ya', 'yes'],
            'tidak': ['no', 'nggak', 'enggak'],
            'baik': ['good', 'bagus'],
            'senang': ['happy', 'bahagia'],
            'sedih': ['sad', 'sedih']
        }

        for _ in range(count_per_message):
            text = base_text
            for word, syns in synonyms.items():
                if word in text.lower():
                    replacement = random.choice(syns)
                    # Simple replacement (case-insensitive)
                    text = re.sub(r'\b' + re.escape(word) + r'\b', replacement, text, flags=re.IGNORECASE)

            variations.append({'text': text, 'label': label})

    return variations

def main():
    parser = argparse.ArgumentParser(description='Expand Indonesian SMS dataset')
    parser.add_argument('--input', default='dataset/sms_spam_indo-xsmall.csv',
                       help='Input Indonesian dataset file')
    parser.add_argument('--output', default='dataset/sms_spam_indo_expanded.csv',
                       help='Output expanded dataset file')
    parser.add_argument('--target-size', type=int, default=50000,
                       help='Target dataset size')

    args = parser.parse_args()

    # Load original dataset
    print(f"Loading Indonesian dataset from {args.input}...")
    original_data = load_indonesian_dataset(args.input)
    print(f"Original dataset size: {len(original_data)}")

    # Analyze patterns
    patterns, indonesian_words, spam_keywords = analyze_patterns(original_data)

    # Calculate how many new messages to generate
    current_size = len(original_data)
    target_size = args.target_size
    new_messages_needed = target_size - current_size

    if new_messages_needed <= 0:
        print("Dataset already meets target size")
        return

    # Distribute new messages across classes (maintain similar proportions)
    label_counts = Counter(item['label'] for item in original_data)
    total_original = sum(label_counts.values())

    if total_original == 0:
        print("No valid data found in original dataset")
        return

    proportions = {}
    for label, count in label_counts.items():
        proportions[label] = count / total_original

    new_messages_per_class = {}
    for label in ['ham', 'spam', 'phishing']:  # Ensure all labels are considered
        if label in proportions:
            new_messages_per_class[label] = int(new_messages_needed * proportions[label])
        else:
            new_messages_per_class[label] = 0

    # Adjust to meet exact target
    total_new = sum(new_messages_per_class.values())
    if total_new < new_messages_needed:
        # Add remaining to ham class (or most common class)
        most_common_label = label_counts.most_common(1)[0][0] if label_counts else 'ham'
        new_messages_per_class[most_common_label] += (new_messages_needed - total_new)

    print(f"Generating {sum(new_messages_per_class.values())} new messages:")
    for label, count in new_messages_per_class.items():
        print(f"  {label}: {count}")

    # Generate new messages
    new_messages = []

    # Generate ham messages
    if 'ham' in new_messages_per_class:
        ham_count = new_messages_per_class['ham']
        new_ham = generate_ham_messages(patterns, indonesian_words, ham_count)
        new_messages.extend(new_ham)

    # Generate spam messages
    if 'spam' in new_messages_per_class:
        spam_count = new_messages_per_class['spam']
        new_spam = generate_spam_messages(patterns, indonesian_words, spam_keywords, spam_count)
        new_messages.extend(new_spam)

    # Generate phishing messages
    if 'phishing' in new_messages_per_class:
        phishing_count = new_messages_per_class['phishing']
        new_phishing = generate_phishing_messages(patterns, indonesian_words, phishing_count)
        new_messages.extend(new_phishing)

    # Create variations of original messages
    variations = create_variations(original_data, count_per_message=2)
    new_messages.extend(variations)

    # Combine all data
    expanded_data = original_data + new_messages

    # Shuffle
    random.seed(42)
    random.shuffle(expanded_data)

    print(f"\nExpanded dataset size: {len(expanded_data)}")

    # Save expanded dataset
    with open(args.output, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['text', 'label'])
        writer.writeheader()
        writer.writerows(expanded_data)

    print(f"Expanded dataset saved to {args.output}")

    # Print statistics
    final_label_counts = Counter(item['label'] for item in expanded_data)
    print("\nFinal label distribution:")
    for label, count in final_label_counts.items():
        percentage = (count / len(expanded_data)) * 100
        print(f"  {label}: {count} ({percentage:.1f}%)")

if __name__ == '__main__':
    main()