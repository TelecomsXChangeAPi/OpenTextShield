"""
Dataset augmentation script for OpenTextShield.

Generates additional training samples to address weak points identified in testing.
"""

import pandas as pd
import random
from typing import List, Dict, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatasetAugmenter:
    """Augments SMS dataset for better model performance."""

    def __init__(self):
        # Base templates for generation
        self.ham_templates = [
            "Hi {name}, how are you?",
            "Thanks for your message. Talk soon.",
            "Meeting at {time} tomorrow.",
            "Your order #{number} has been shipped.",
            "Reminder: Doctor appointment on {date}.",
            "Happy birthday! Hope you have a great day.",
            "Can we reschedule to {time}?",
            "Thanks for the help with {task}.",
            "Weather looks good for {activity} today.",
            "Just checking in, hope everything is well.",
            # Ambiguous ham - might look suspicious
            "Your verification code is {number}",
            "Login to your account at {url}",
            "Your payment of ${amount} was processed",
            "Call {number} for customer support",
            "Your appointment is confirmed for {time}",
            "Track your package at {url}",
            "Your account balance is ${amount}",
            "New message from {name}: Check your email",
            "Security update: Password changed successfully",
            "Your subscription will renew on {date}",
            "Download the latest update at {url}",
            "Your friend request from {name} is pending",
            "Meeting invite: {time} tomorrow, join at {url}",
            "Your test results are ready. View at {url}",
            "Confirmation code: {number}. Valid for 10 minutes"
        ]

        self.spam_templates = [
            "WIN {amount} NOW! Click {url} to claim!",
            "Congratulations! You've won a {prize}!",
            "Limited time offer: {discount}% off everything!",
            "Your account has been selected for a special promotion!",
            "Earn ${amount} per week working from home!",
            "FREE {item}! Limited stock available!",
            "URGENT: Your prize is waiting! Call {number}",
            "You have been chosen for our exclusive giveaway!",
            "Make money fast with our proven system!",
            "Don't miss this amazing deal: {product} for only ${price}!",
            # Tricky spam - more convincing
            "🎉 CONGRATS! You won ${amount} Amazon gift card! Redeem at {url} 🎉",
            "FLASH SALE: {discount}% OFF all items! Shop now: {url}",
            "Make $5000/month from home! No experience needed. Apply: {url}",
            "Your entry won! Claim ${amount} cash prize at {url}",
            "FREE iPhone 15! Enter code {number} at {url}",
            "Adult content alert! View hot singles near you: {url}",
            "Weight loss secret: Lose 20lbs in 2 weeks! Get it: {url}",
            "Investment opportunity: 1000% returns guaranteed! Invest at {url}",
            "Dating site: Meet local singles for FREE! Sign up: {url}",
            "Cheap flights to Europe! Book now: {url}",
            "Work from home: Earn ${amount}/hour! Start today: {url}",
            "Your dream job awaits! Apply now: {url}",
            "Exclusive deal: {product} at 90% off! Limited time: {url}",
            "Win a car! Enter daily for free: {url}",
            "Debt relief program: Get out of debt fast! Call {number}"
        ]

        self.phishing_templates = [
            "Your account has been suspended. Verify at {url}",
            "Security alert: Unusual activity detected. Login at {url}",
            "Your payment failed. Update card details at {url}",
            "Bank alert: Confirm your identity at {url}",
            "Your package is on hold. Pay delivery fee at {url}",
            "Microsoft: Your account needs verification. Visit {url}",
            "Netflix: Payment method expired. Update at {url}",
            "Amazon: Suspicious login. Secure account at {url}",
            "PayPal: Unusual transaction. Confirm at {url}",
            "IRS: Tax refund pending. Claim at {url}",
            # Tricky phishing - more sophisticated
            "Hi {name}, your Amazon order is delayed. Confirm shipping at {url}",
            "Bank of America: Suspicious charge of ${amount}. Review at {url}",
            "Apple ID: Your account was accessed from unknown device. Secure it now: {url}",
            "FedEx: Package delivery failed. Reschedule pickup at {url}",
            "LinkedIn: {name} viewed your profile. Connect now: {url}",
            "Facebook: Your account will be deactivated in 24h unless verified at {url}",
            "Google: Unusual sign-in detected. Verify it's you: {url}",
            "Uber: Payment failed for ride. Update card at {url}",
            "Instagram: Someone tried to hack your account. Change password at {url}",
            "WhatsApp: New security code: {number}. Don't share it.",
            "Your friend {name} sent you money via PayPal. Claim at {url}",
            "COVID-19 test results available. View at {url}",
            "Your tax refund of ${amount} is ready. Claim at {url}",
            "Prize winner! Claim your ${amount} gift card at {url}",
            "Job offer: Earn ${amount}/month. Apply at {url}",
            "Your subscription expires soon. Renew at {url} to avoid interruption"
        ]

        # Multilingual templates - expanded
        self.multilingual_templates = {
            'Spanish': {
                'ham': [
                    "Hola {name}, ¿cómo estás?",
                    "Gracias por tu mensaje. Hablamos pronto.",
                    "Reunión a las {time} mañana.",
                    "Tu pedido #{number} ha sido enviado.",
                    "El clima está bonito hoy.",
                    "Confirmo la cita para mañana."
                ],
                'spam': [
                    "¡GANA {amount} AHORA! Haz clic en {url} para reclamar!",
                    "¡Felicidades! ¡Has ganado un {prize}!",
                    "Oferta limitada: {discount}% de descuento en todo!",
                    "¡Trabaja desde casa y gana ${amount} al mes!"
                ],
                'phishing': [
                    "Tu cuenta ha sido suspendida. Verifica en {url}",
                    "Alerta de seguridad: Actividad inusual detectada. Inicia sesión en {url}",
                    "Tu pago falló. Actualiza los datos de la tarjeta en {url}",
                    "Banco: Actividad sospechosa en tu cuenta. Confirma identidad en {url}"
                ]
            },
            'French': {
                'ham': [
                    "Salut {name}, comment ça va?",
                    "Merci pour ton message. On se parle bientôt.",
                    "Réunion à {time} demain.",
                    "Votre commande #{number} a été expédiée.",
                    "Le temps est beau aujourd'hui.",
                    "Je confirme le rendez-vous pour demain."
                ],
                'spam': [
                    "GAGNEZ {amount} MAINTENANT! Cliquez sur {url} pour réclamer!",
                    "Félicitations! Vous avez gagné un {prize}!",
                    "Offre limitée: {discount}% de réduction sur tout!",
                    "Travaillez à domicile et gagnez {amount}€ par mois!"
                ],
                'phishing': [
                    "Votre compte a été suspendu. Vérifiez sur {url}",
                    "Alerte sécurité: Activité inhabituelle détectée. Connectez-vous sur {url}",
                    "Votre paiement a échoué. Mettez à jour les détails de la carte sur {url}",
                    "Banque: Activité suspecte sur votre compte. Confirmez votre identité sur {url}"
                ]
            },
            'German': {
                'ham': [
                    "Hallo {name}, wie geht's?",
                    "Danke für deine Nachricht. Wir reden bald.",
                    "Treffen um {time} morgen.",
                    "Ihre Bestellung #{number} wurde versendet.",
                    "Das Wetter ist schön heute.",
                    "Ich bestätige den Termin für morgen."
                ],
                'spam': [
                    "GEWINNE {amount} JETZT! Klicke auf {url} um zu beanspruchen!",
                    "Herzlichen Glückwunsch! Sie haben ein {prize} gewonnen!",
                    "Begrenztes Angebot: {discount}% Rabatt auf alles!",
                    "Arbeite von zu Hause und verdiene {amount}€ pro Monat!"
                ],
                'phishing': [
                    "Ihr Konto wurde gesperrt. Überprüfen Sie auf {url}",
                    "Sicherheitswarnung: Ungewöhnliche Aktivität erkannt. Melden Sie sich bei {url} an",
                    "Ihre Zahlung ist fehlgeschlagen. Aktualisieren Sie die Kartendaten auf {url}",
                    "Bank: Verdächtige Aktivität auf Ihrem Konto. Bestätigen Sie Ihre Identität auf {url}"
                ]
            },
            'Arabic': {
                'ham': [
                    "مرحبا {name}، كيف حالك؟",
                    "شكرا لرسالتك. نتكلم قريبا.",
                    "اجتماع في {time} غدا.",
                    "تم شحن طلبك رقم {number}.",
                    "الطقس جميل اليوم.",
                    "أؤكد الموعد للغد."
                ],
                'spam': [
                    "اربح {amount} الآن! اضغط {url} للمطالبة!",
                    "مبروك! لقد فزت بـ {prize}!",
                    "عرض محدود: خصم {discount}% على كل شيء!",
                    "اعمل من المنزل واكسب {amount} ريال شهريا!"
                ],
                'phishing': [
                    "تم تعليق حسابك. تحقق في {url}",
                    "تنبيه أمني: تم اكتشاف نشاط غير عادي. سجل الدخول في {url}",
                    "فشل الدفع. حدث بيانات البطاقة في {url}",
                    "البنك: نشاط مشبوه في حسابك. أكد هويتك في {url}"
                ]
            },
            'Chinese': {
                'ham': [
                    "你好{name}，你怎么样？",
                    "谢谢你的消息。我们很快再聊。",
                    "明天{time}开会。",
                    "您的订单#{number}已发货。",
                    "今天天气很好。",
                    "我确认明天的约会。"
                ],
                'spam': [
                    "立即赢得{amount}！点击{url}领取！",
                    "恭喜！你赢得了{prize}！",
                    "限时优惠：全场{discount}%折扣！",
                    "在家工作每月赚{amount}元！"
                ],
                'phishing': [
                    "您的账户已被暂停。请在{url}验证",
                    "安全警报：检测到异常活动。请登录{url}",
                    "您的付款失败。请在{url}更新卡详情",
                    "银行：您的账户有可疑活动。请在{url}确认身份"
                ]
            },
            'Hindi': {
                'ham': [
                    "नमस्ते {name}, आप कैसे हैं?",
                    "आपके संदेश के लिए धन्यवाद। जल्दी बात करते हैं।",
                    "कल {time} बजे मीटिंग।",
                    "आपका ऑर्डर #{number} भेज दिया गया है।",
                    "आज मौसम अच्छा है।",
                    "मैं कल की अपॉइंटमेंट की पुष्टि करता हूं।"
                ],
                'spam': [
                    "अभी {amount} जीतें! दावा करने के लिए {url} पर क्लिक करें!",
                    "बधाई! आपने {prize} जीता है!",
                    "सीमित समय की पेशकश: सब कुछ पर {discount}% छूट!",
                    "घर से काम करें और महीने में {amount} रुपये कमाएं!"
                ],
                'phishing': [
                    "आपका खाता निलंबित कर दिया गया है। {url} पर सत्यापित करें",
                    "सुरक्षा अलर्ट: असामान्य गतिविधि का पता चला। {url} पर लॉग इन करें",
                    "आपका भुगतान विफल हुआ। {url} पर कार्ड विवरण अपडेट करें",
                    "बैंक: आपके खाते में संदिग्ध गतिविधि। {url} पर पहचान की पुष्टि करें"
                ]
            },
            'Portuguese': {
                'ham': [
                    "Olá {name}, como você está?",
                    "Obrigado pela sua mensagem. Falamos em breve.",
                    "Reunião às {time} amanhã.",
                    "Seu pedido #{number} foi enviado.",
                    "O tempo está bom hoje.",
                    "Confirmo o compromisso para amanhã."
                ],
                'spam': [
                    "GANHE {amount} AGORA! Clique em {url} para reivindicar!",
                    "Parabéns! Você ganhou um {prize}!",
                    "Oferta limitada: {discount}% de desconto em tudo!",
                    "Trabalhe em casa e ganhe R$ {amount} por mês!"
                ],
                'phishing': [
                    "Sua conta foi suspensa. Verifique em {url}",
                    "Alerta de segurança: Atividade incomum detectada. Faça login em {url}",
                    "Seu pagamento falhou. Atualize os dados do cartão em {url}",
                    "Banco: Atividade suspeita na sua conta. Confirme identidade em {url}"
                ]
            }
        }

        # Edge case templates - expanded with tricky cases
        self.edge_cases = {
            'short': [
                ("Hi", "ham"),
                ("OK", "ham"),
                ("Yes", "ham"),
                ("No", "ham"),
                ("Win!", "spam"),
                ("Free!", "spam"),
                ("Call!", "spam"),
                ("Now!", "spam"),
                ("Alert!", "phishing"),
                ("Urgent!", "phishing"),
                ("Help!", "ham"),
                ("Stop", "ham"),
                ("Go", "ham"),
                ("Buy!", "spam"),
                ("Sale!", "spam"),
                ("Code", "ham"),
                ("Link", "spam"),
                ("Click", "spam"),
                ("Verify", "phishing"),
                ("Secure", "phishing")
            ],
            'long': [
                ("This is a very long message that contains multiple sentences and a lot of information that should test how well the model can handle lengthy SMS messages that might be truncated or processed differently by the classification system. It includes various topics and should help improve the model's ability to classify longer texts accurately." * 2, "ham"),
                ("Congratulations on your recent purchase! Your order has been processed and will be shipped within 2-3 business days. You will receive a tracking number via email once it ships. If you have any questions about your order, please don't hesitate to contact our customer service team at support@company.com or call us at 1-800-123-4567. Thank you for shopping with us!" * 2, "ham"),
                ("ATTENTION: Your account has been flagged for suspicious activity. To prevent unauthorized access, we need you to verify your identity immediately. Click the link below to complete the verification process. Failure to do so within 24 hours may result in temporary account suspension. Verify now: http://secure-bank-verification.com/verify?user=12345&token=abc123" * 2, "phishing")
            ],
            'special_chars': [
                ("🎉 WIN $1000! Click here: http://bit.ly/123 🎉", "spam"),
                ("Hello!!! How are you??? 😊", "ham"),
                ("URGENT: Your account #12345 is locked!!!", "phishing"),
                ("Call me ASAP!!! 📞", "ham"),
                ("FREE iPhone!!! Don't miss out!!! 💰", "spam"),
                ("🚨 ALERT: Your PayPal account is limited! 🚨", "phishing"),
                ("💰 MAKE MONEY FAST! 💰 No investment required!", "spam"),
                ("❤️ Dating: Meet your soulmate today! ❤️", "spam"),
                ("🔒 Secure your account now! 🔒", "phishing"),
                ("📱 New iPhone for FREE! 📱 Just pay shipping!", "spam"),
                ("⚠️ Warning: Unusual login detected ⚠️", "phishing"),
                ("🎁 Gift card giveaway! 🎁 Enter to win!", "spam"),
                ("📧 Email verification required 📧", "phishing"),
                ("💸 Earn $1000/week from home 💸", "spam"),
                ("🔐 Password reset needed 🔐", "phishing")
            ],
            'numbers': [
                ("123456", "ham"),
                ("Your code is 123456", "ham"),
                ("Account: 123456789", "ham"),
                ("Win code: 123456", "spam"),
                ("Verification: 123456", "phishing"),
                ("OTP: 123456", "ham"),
                ("PIN: 1234", "ham"),
                ("Prize code: 987654", "spam"),
                ("Security code: 111111", "phishing"),
                ("Order #: 555-1234", "ham"),
                ("Ticket #: 789012", "ham"),
                ("Lottery #: 456789", "spam"),
                ("Case #: 321098", "phishing")
            ],
            'urls': [
                ("Check this: https://google.com", "ham"),
                ("Visit: http://facebook.com", "ham"),
                ("Win at: http://fake-prize.com", "spam"),
                ("Secure login: https://bank-secure.com", "phishing"),
                ("Download: http://malware-site.ru", "phishing"),
                ("Track order: https://amazon.com/track", "ham"),
                ("Verify email: https://gmail.com/verify", "ham"),
                ("Claim prize: http://win-big.com", "spam"),
                ("Update payment: https://paypal-secure.com", "phishing"),
                ("Join meeting: https://zoom.us/join", "ham"),
                ("Reset password: https://netflix.com/reset", "ham"),
                ("Free download: http://free-stuff.net", "spam"),
                ("Account recovery: https://microsoft-recovery.com", "phishing")
            ],
            'mixed_languages': [
                ("Hello, ¿cómo estás? Guten Tag!", "ham"),
                ("Win 1000€! Cliquez ici: http://spam.fr", "spam"),
                ("Tu cuenta está suspendida. Verifica aquí: http://banco-falso.es", "phishing"),
                ("Привет, how are you? Ça va?", "ham"),
                ("Gewinn 1000€! Klicke hier: http://spam.de", "spam"),
                ("您的账户被暂停。请验证：http://fake-bank.cn", "phishing")
            ],
            'typos_grammar': [
                ("Urgent! Your acount has been hacked. Click hear to fix: http://bank-fix.com", "phishing"),
                ("Congrats! You wone a free ipone! Claim now: http://apple-gift.com", "spam"),
                ("Hi, this is john from bank. Your account need verification. Call 123-456-7890", "phishing"),
                ("Free money! No strings attached. Send email to freemoney@spam.com", "spam"),
                ("Your package is stuck in customs. Pay $50 to release: http://customs-fee.com", "phishing"),
                ("Work from home and earn $5000/month! No degree required. Apply today!", "spam"),
                ("Security alert: Someone tried to access your account from Russia. Confirm identity: http://secure-login.net", "phishing")
            ]
        }

    def generate_sample(self, template: str, label: str) -> str:
        """Generate a sample from template."""
        # Fill in placeholders
        replacements = {
            '{name}': random.choice(['John', 'Sarah', 'Mike', 'Anna', 'David']),
            '{time}': random.choice(['2pm', '3:30', '10am', '5pm']),
            '{date}': random.choice(['Monday', 'Tuesday', 'Friday']),
            '{task}': random.choice(['project', 'report', 'presentation']),
            '{activity}': random.choice(['hiking', 'swimming', 'shopping']),
            '{number}': str(random.randint(1000, 9999)),
            '{amount}': str(random.randint(500, 5000)),
            '{prize}': random.choice(['iPhone', 'car', 'vacation', 'cash prize']),
            '{discount}': str(random.randint(20, 80)),
            '{item}': random.choice(['gift card', 'phone', 'laptop']),
            '{price}': str(random.randint(10, 100)),
            '{url}': random.choice(['bit.ly/123', 'tinyurl.com/abc', 'goo.gl/xyz']),
            '{product}': random.choice(['shoes', 'watch', 'bag', 'jacket'])
        }

        for placeholder, value in replacements.items():
            template = template.replace(placeholder, value)

        return template

    def augment_dataset(self, original_csv: str, num_new_samples: int = 1000, balance_classes: bool = True) -> pd.DataFrame:
        """Augment the original dataset with new samples."""
        logger.info(f"Loading original dataset from {original_csv}")
        original_df = pd.read_csv(original_csv)

        new_samples = []

        if balance_classes:
            # Calculate target samples per class to balance
            min_class_size = original_df['label'].value_counts().min()
            target_per_class = max(min_class_size, 50000)  # At least 50k per class
            samples_per_class = num_new_samples // 3
        else:
            samples_per_class = num_new_samples // 3
            target_per_class = samples_per_class

        classes = ['ham', 'spam', 'phishing']

        for label in classes:
            logger.info(f"Generating {target_per_class} new {label} samples")

            if label == 'ham':
                templates = self.ham_templates
            elif label == 'spam':
                templates = self.spam_templates
            else:
                templates = self.phishing_templates

            for _ in range(target_per_class):
                template = random.choice(templates)
                text = self.generate_sample(template, label)
                new_samples.append({'text': text, 'label': label})

        # Add multilingual samples
        logger.info("Adding multilingual samples")
        for lang, lang_templates in self.multilingual_templates.items():
            for label, templates in lang_templates.items():
                for template in templates[:10]:  # More per language
                    text = self.generate_sample(template, label)
                    new_samples.append({'text': text, 'label': label})

        # Add edge cases
        logger.info("Adding edge cases")
        for category, samples in self.edge_cases.items():
            for text, label in samples:
                new_samples.append({'text': text, 'label': label})

        # Add back-translated samples (simple simulation)
        logger.info("Adding back-translated variations")
        back_translate_templates = [
            ("Hello, your package is ready for pickup at the post office.", "ham"),
            ("WINNER! You have won $1,000,000! Call now to claim!", "spam"),
            ("Your bank account has been compromised. Change password immediately.", "phishing")
        ]
        for template, label in back_translate_templates:
            # Simulate back-translation variations
            variations = [
                template,
                template.replace("Hello", "Hi").replace("package", "parcel"),
                template.replace("WINNER", "CONGRATS").replace("$1,000,000", "1 million dollars"),
                template.replace("bank account", "banking account").replace("compromised", "hacked")
            ]
            for var in variations:
                new_samples.append({'text': var, 'label': label})

        # Create augmented dataset
        new_df = pd.DataFrame(new_samples)
        augmented_df = pd.concat([original_df, new_df], ignore_index=True)

        # Remove duplicates
        augmented_df = augmented_df.drop_duplicates(subset=['text'])

        # Shuffle
        augmented_df = augmented_df.sample(frac=1, random_state=42).reset_index(drop=True)

        logger.info(f"Original dataset: {len(original_df)} samples")
        logger.info(f"New samples: {len(new_df)} samples")
        logger.info(f"Augmented dataset (after deduplication): {len(augmented_df)} samples")

        return augmented_df

    def save_augmented_dataset(self, df: pd.DataFrame, output_path: str):
        """Save augmented dataset."""
        df.to_csv(output_path, index=False)
        logger.info(f"Augmented dataset saved to {output_path}")

def main():
    augmenter = DatasetAugmenter()

    # Augment the dataset with balancing
    augmented_df = augmenter.augment_dataset(
        "dataset/sms_spam_phishing_dataset_v2.1.csv",
        num_new_samples=50000,  # More samples for better balance
        balance_classes=True
    )

    # Save augmented dataset
    augmenter.save_augmented_dataset(
        augmented_df,
        "dataset/sms_spam_phishing_dataset_v2.3_augmented.csv"
    )

    # Print statistics
    print("\nAugmented Dataset Statistics:")
    print(augmented_df['label'].value_counts())
    print(f"\nTotal samples: {len(augmented_df)}")

    # Check balance
    counts = augmented_df['label'].value_counts()
    print(f"\nClass distribution:")
    for label, count in counts.items():
        pct = count / len(augmented_df) * 100
        print(f"{label}: {count} ({pct:.1f}%)")

if __name__ == "__main__":
    main()