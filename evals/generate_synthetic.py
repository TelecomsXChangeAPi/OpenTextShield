#!/usr/bin/env python3
"""
Targeted synthetic SMS generator for OpenTextShield model improvement.

The v2.5 model evaluation (see evals/REPORT.md) showed near-perfect accuracy on
classic 2010-era spam (UCI 99.3%) but a collapse on modern smishing: ~25% of
real-world phishing was labelled `ham` and silently passed. This generator
manufactures training data for exactly the categories that failed, with three
design goals:

  1. Cover modern smishing archetypes the model has never seen well:
     toll/delivery fee scams, "hi mum" family impersonation, vishing callback
     numbers, BEC gift-card requests, crypto-wallet scares, OTP-forwarding theft,
     and text obfuscation (homoglyph / zero-width / leet / spaced).

  2. Multilingual parity — every phishing/spam archetype is emitted across the
     languages where block-rate was weakest (es, fr, de, pt, it, nl, ar, he, ru,
     id, tr, ja, zh, hi-latn) plus English.

  3. HARD NEGATIVES. The biggest risk when teaching a model to block more is that
     it starts blocking legitimate OTP codes, bank fraud alerts, and delivery
     notices. For every attack archetype there is a benign twin (real OTP vs
     OTP-theft, real fraud alert vs fake "verify now", real delivery vs fee scam)
     so the decision boundary is taught explicitly.

Output: CSV with `text,label` columns (label in {ham,spam,phishing}), the same
schema the training pipeline consumes.

Usage:
  python evals/generate_synthetic.py --n-per-template 6 \
      --out src/mBERT/training/model-training/dataset/synthetic_fable5_v1.csv
"""

import argparse
import csv
import random
from pathlib import Path

LANGS = ["en", "es", "fr", "de", "pt", "it", "nl", "ar", "he", "ru",
         "id", "tr", "ja", "zh", "hi"]

# ---------------------------------------------------------------- value pools

BANKS = {
    "en": ["Chase", "Wells Fargo", "Bank of America", "Citibank", "HSBC", "Barclays", "Lloyds", "NatWest"],
    "es": ["BBVA", "Santander", "CaixaBank", "Banorte", "Bancomer", "Banco Sabadell"],
    "fr": ["Société Générale", "BNP Paribas", "Crédit Agricole", "La Banque Postale", "LCL"],
    "de": ["Sparkasse", "Deutsche Bank", "Commerzbank", "Volksbank", "ING"],
    "pt": ["Banco do Brasil", "Itaú", "Bradesco", "Caixa", "Nubank", "Santander"],
    "it": ["Intesa Sanpaolo", "UniCredit", "BPER", "Poste Italiane"],
    "nl": ["ING", "Rabobank", "ABN AMRO", "SNS Bank"],
    "ar": ["البنك الأهلي", "بنك الرياض", "مصرف الراجحي", "بنك مصر"],
    "he": ["בנק הפועלים", "בנק לאומי", "בנק דיסקונט", "מזרחי טפחות"],
    "ru": ["Сбербанк", "Тинькофф", "Альфа-Банк", "ВТБ"],
    "id": ["BRI", "BCA", "Mandiri", "BNI"],
    "tr": ["Ziraat Bankası", "Garanti BBVA", "İş Bankası", "Akbank"],
    "ja": ["三井住友銀行", "三菱UFJ銀行", "みずほ銀行", "ゆうちょ銀行"],
    "zh": ["工商银行", "建设银行", "招商银行", "中国银行"],
    "hi": ["SBI", "HDFC Bank", "ICICI", "Axis Bank", "Paytm"],
}
COURIERS = {
    "en": ["USPS", "FedEx", "UPS", "DHL", "Royal Mail", "Amazon Logistics"],
    "es": ["Correos", "SEUR", "MRW", "DHL"],
    "fr": ["Chronopost", "Colissimo", "Mondial Relay", "DPD"],
    "de": ["DHL", "Hermes", "DPD", "GLS"],
    "pt": ["Correios", "Jadlog", "Sequoia", "DHL"],
    "it": ["Poste Italiane", "BRT", "GLS", "SDA"],
    "nl": ["PostNL", "DHL", "DPD", "GLS"],
    "ar": ["أرامكس", "سمسا", "البريد السعودي", "DHL"],
    "he": ["דואר ישראל", "UPS", "DHL", "FedEx"],
    "ru": ["Почта России", "СДЭК", "Boxberry", "DPD"],
    "id": ["JNE", "J&T", "SiCepat", "Pos Indonesia"],
    "tr": ["PTT Kargo", "Yurtiçi Kargo", "Aras Kargo", "MNG Kargo"],
    "ja": ["日本郵便", "ヤマト運輸", "佐川急便"],
    "zh": ["顺丰速运", "中国邮政", "圆通速递", "京东物流"],
    "hi": ["India Post", "Blue Dart", "DTDC", "Delhivery"],
}
SHORTENERS = ["bit.ly", "tinyurl.com", "cutt.ly", "t.ly", "rb.gy", "is.gd", "shorturl.at"]
BAD_TLDS = ["-secure.com", "-verify.net", ".top", ".vip", ".live", ".info", "-login.app",
            ".xyz", "-pay.online", ".cn-bank.com", "-billing.net"]


def rng_phone():
    return random.choice([
        f"+1 (8{random.randint(0,8)}{random.randint(0,9)}) {random.randint(200,999)}-{random.randint(1000,9999)}",
        f"+44 7{random.randint(100,999)} {random.randint(100000,999999)}",
        f"0{random.randint(6,7)} {random.randint(10,99)} {random.randint(10,99)} {random.randint(10,99)} {random.randint(10,99)}",
        f"1-8{random.randint(0,8)}{random.randint(0,9)}-{random.randint(200,999)}-{random.randint(1000,9999)}",
    ])


def rng_code(n=6):
    return "".join(str(random.randint(0, 9)) for _ in range(n))


_AMOUNTS = {
    "en": ["$1.99", "$3.95", "$499.99", "$1,482.00", "£1.10", "£278.44"],
    "es": ["1,99 €", "184,50 €", "8,450 MXN", "499,99 €"],
    "fr": ["1,99 €", "184,50 €", "499,99 €", "12,90 €"],
    "de": ["1,99 €", "184,50 €", "499,99 €", "9,90 €"],
    "pt": ["R$ 9,90", "R$ 1.482,00", "R$ 49,90"],
    "it": ["1,99 €", "184,50 €", "499,99 €"],
    "nl": ["€ 1,99", "€ 184,50", "€ 12,90"],
    "ar": ["10 ريال", "1.99 دولار", "500 ريال"],
    "he": ["9.90 ₪", "278 ₪", "1,490 ₪"],
    "ru": ["1 990 ₽", "14 500 ₽", "499 ₽"],
    "id": ["Rp 9.900", "Rp 1.482.000", "Rp 49.900"],
    "tr": ["19,90 TL", "1.482 TL", "499 TL"],
    "ja": ["¥1,980", "¥14,800", "¥4,990"],
    "zh": ["¥9.90", "¥1,482", "¥49.90"],
    "hi": ["₹99", "₹1,482", "₹499"],
}


def rng_amount(lang="en"):
    return random.choice(_AMOUNTS.get(lang, _AMOUNTS["en"]))


def rng_badurl(brandhint="secure"):
    base = random.choice([brandhint, "account", "verify", "portal", "id-check", "billing", "update"])
    return f"{base}{random.choice(['-', '.'])}{random.choice(['login','client','pay','service'])}{random.choice(BAD_TLDS)}"


def rng_shorturl():
    return f"https://{random.choice(SHORTENERS)}/{random.choice('abcdefghjkmnpqrstuvwxyz')}{rng_code(random.randint(4,6))}"


# ---------------------------------------------------------------- homoglyph / obfuscation

CYR = {"a": "а", "e": "е", "o": "о", "p": "р", "c": "с", "y": "у", "x": "х", "i": "і"}


def homoglyph(text):
    out = []
    for ch in text:
        low = ch.lower()
        if low in CYR and random.random() < 0.5:
            sub = CYR[low]
            out.append(sub.upper() if ch.isupper() else sub)
        else:
            out.append(ch)
    return "".join(out)


def zero_width(text):
    return "​".join(text)


def leet(text):
    table = str.maketrans({"o": "0", "i": "1", "e": "3", "a": "@", "s": "$"})
    return text.translate(table)


def spaced(text):
    return " ".join(text.replace(" ", ""))


# ---------------------------------------------------------------- templates
# Each builder returns a finished message string. Builders take the language
# code and pull from the localized pools above. Where we lack a faithful
# translation we keep the structure and localize the brand/number, which is
# realistic (many regional scams reuse English scaffolding).

def b(lang, **pools):
    """Helper: pick a localized brand/courier for the language with en fallback."""
    return pools


PHISHING_BUILDERS = []
SPAM_BUILDERS = []
HAM_BUILDERS = []


def phishing(fn):
    PHISHING_BUILDERS.append(fn); return fn


def spam(fn):
    SPAM_BUILDERS.append(fn); return fn


def ham(fn):
    HAM_BUILDERS.append(fn); return fn


def bank(lang):
    return random.choice(BANKS.get(lang, BANKS["en"]))


def courier(lang):
    return random.choice(COURIERS.get(lang, COURIERS["en"]))


# ---- PHISHING archetypes (the categories the model missed) ----

@phishing
def toll_scam(lang):
    msg = {
        "en": f"FINAL NOTICE: Unpaid toll of {rng_amount(lang)}. Pay now to avoid a late fee and license suspension: {rng_badurl('toll')}",
        "es": f"AVISO: Peaje pendiente de {rng_amount(lang)}. Pague ahora para evitar recargos: {rng_badurl('peaje')}",
        "fr": f"DERNIER AVIS: Péage impayé de {rng_amount(lang)}. Réglez maintenant pour éviter une amende: {rng_badurl('peage')}",
        "de": f"LETZTE MAHNUNG: Offene Mautgebühr {rng_amount(lang)}. Zahlen Sie jetzt: {rng_badurl('maut')}",
        "pt": f"AVISO FINAL: Pedágio não pago de {rng_amount(lang)}. Pague agora: {rng_badurl('pedagio')}",
        "it": f"ULTIMO AVVISO: Pedaggio non pagato di {rng_amount(lang)}. Paga ora: {rng_badurl('pedaggio')}",
        "nl": f"LAATSTE WAARSCHUWING: Onbetaalde tol {rng_amount(lang)}. Betaal nu: {rng_badurl('tol')}",
    }
    return msg.get(lang, msg["en"])


@phishing
def delivery_fee_scam(lang):
    c = courier(lang)
    msg = {
        "en": f"{c}: Your parcel is held pending a {rng_amount(lang)} customs fee. Pay within 24h or it returns to sender: {rng_badurl('redelivery')}",
        "es": f"{c}: Su paquete está retenido en aduana. Pague {rng_amount(lang)} para liberarlo: {rng_badurl('aduana')}",
        "fr": f"{c}: Votre colis est bloqué. Frais de {rng_amount(lang)} requis sous 48h: {rng_badurl('colis')}",
        "de": f"{c}: Ihr Paket konnte nicht zugestellt werden. Zollgebühr {rng_amount(lang)}: {rng_badurl('paket')}",
        "pt": f"{c}: Sua encomenda está retida na alfândega. Pague {rng_amount(lang)}: {rng_badurl('encomenda')}",
        "it": f"{c}: Il tuo pacco è in giacenza. Paga {rng_amount(lang)} per lo sdoganamento: {rng_badurl('pacco')}",
        "nl": f"{c}: Uw pakket wacht op een douanekosten van {rng_amount(lang)}: {rng_badurl('pakket')}",
        "ar": f"{c}: شحنتك معلقة في الجمارك. ادفع {rng_amount(lang)} للإفراج: {rng_badurl('delivery')}",
        "he": f"{c}: החבילה שלך מעוכבת במכס. שלם {rng_amount(lang)} לשחרור: {rng_badurl('fee')}",
        "ru": f"{c}: Ваша посылка задержана на таможне. Оплатите {rng_amount(lang)}: {rng_badurl('posylka')}",
        "id": f"{c}: Paket Anda tertahan di bea cukai. Bayar {rng_amount(lang)}: {rng_badurl('paket')}",
        "tr": f"{c}: Paketiniz gümrükte bekliyor. {rng_amount(lang)} ödeyin: {rng_badurl('kargo')}",
        "ja": f"【{c}】お荷物のお届けに{rng_amount(lang)}の関税が必要です: {rng_badurl('haitatsu')}",
        "zh": f"【{c}】您的包裹因关税被扣留，请支付{rng_amount(lang)}：{rng_badurl('express')}",
        "hi": f"{c}: Aapka parcel customs mein ruka hai. {rng_amount(lang)} pay karein: {rng_badurl('delivery')}",
    }
    return msg.get(lang, msg["en"])


@phishing
def bank_lock_scam(lang):
    bk = bank(lang)
    msg = {
        "en": f"{bk} Alert: Your account has been LOCKED due to suspicious activity. Verify your identity now: {rng_badurl('secure')}",
        "es": f"{bk}: Su cuenta ha sido bloqueada por seguridad. Verifique su identidad: {rng_badurl('seguridad')}",
        "fr": f"{bk}: Votre compte a été bloqué. Vérifiez votre identité immédiatement: {rng_badurl('securite')}",
        "de": f"{bk}: Ihr Konto wurde gesperrt. Bestätigen Sie Ihre Identität: {rng_badurl('konto')}",
        "pt": f"{bk}: Sua conta foi bloqueada. Atualize seus dados agora: {rng_badurl('cliente')}",
        "it": f"{bk}: Il tuo conto è stato bloccato. Verifica la tua identità: {rng_badurl('sicurezza')}",
        "nl": f"{bk}: Uw rekening is geblokkeerd. Verifieer uw identiteit: {rng_badurl('beveiligd')}",
        "ar": f"{bk}: تم إيقاف حسابك مؤقتاً. تحقق من هويتك الآن: {rng_badurl('verify')}",
        "he": f"{bk}: חשבונך הוגבל עקב פעילות חריגה. אמת את פרטיך: {rng_badurl('verify')}",
        "ru": f"{bk}: Ваша карта заблокирована. Подтвердите личность: {rng_badurl('verify')}",
        "id": f"{bk}: Rekening Anda akan diblokir. Verifikasi data Anda: {rng_badurl('verifikasi')}",
        "tr": f"{bk}: Hesabınız askıya alındı. Kimliğinizi doğrulayın: {rng_badurl('dogrulama')}",
        "ja": f"【{bk}】お取引を一時制限しました。本人確認はこちら: {rng_badurl('kakunin')}",
        "zh": f"【{bk}】您的银行卡已被冻结，请立即验证身份：{rng_badurl('verify')}",
        "hi": f"{bk} Alert: Aapka account block ho jayega. Turant KYC update karein: {rng_badurl('kyc')}",
    }
    return msg.get(lang, msg["en"])


@phishing
def family_impersonation(lang):
    msg = {
        "en": random.choice([
            "Hi mum, my phone broke so this is my new number. Can you message me on WhatsApp? Something urgent came up x",
            "Dad it's me, I dropped my phone in the toilet, this is my temp number. I need to pay a bill today, can you help?",
            "Hey it's me, new number, old phone died. Need a quick favour, can you transfer me some money? I'll explain later",
        ]),
        "es": random.choice([
            "Hola mamá, se me rompió el celular, este es mi número nuevo. ¿Me puedes hacer una transferencia urgente? Luego te explico",
            "Papá soy yo, perdí mi teléfono, este es mi número temporal. Necesito que me ayudes con un pago hoy",
        ]),
        "fr": "Maman c'est moi, j'ai cassé mon téléphone. Écris-moi sur WhatsApp à ce nouveau numéro, c'est urgent. J'ai besoin d'aide pour un paiement",
        "de": "Hallo Mama, mein Handy ist kaputt. Das ist meine neue Nummer. Kannst du mir schnell bei einer Überweisung helfen?",
        "pt": "Mãe, quebrei meu celular, esse é meu número novo. Preciso pagar uma conta hoje, me ajuda? Te explico depois",
        "it": "Ciao mamma, ho rotto il telefono, questo è il mio nuovo numero. Mi serve aiuto per un pagamento urgente",
        "nl": "Hoi mam, mijn telefoon is kapot, dit is mijn nieuwe nummer. Kun je me helpen met een spoedbetaling?",
        "ar": "أمي، تلفوني تعطل وهذا رقمي الجديد. أحتاج مساعدتك في تحويل مبلغ بشكل عاجل، سأشرح لاحقاً",
        "he": "אמא זה אני, הטלפון נשבר וזה המספר החדש שלי. אני צריך עזרה דחופה עם תשלום, אסביר אחר כך",
        "ru": "Мам, это я, телефон сломался, вот мой новый номер. Срочно нужна помощь с переводом, потом объясню",
        "id": "Halo ma, hp ku rusak, ini nomor baru. Bisa bantu transfer dulu? Nanti aku jelasin",
        "tr": "Anne benim, telefonum bozuldu, bu yeni numaram. Acil bir ödeme için yardımına ihtiyacım var",
        "ja": "お母さん、携帯が壊れて新しい番号だよ。急ぎで振込をお願いできる？後で説明する",
        "zh": "妈，我手机坏了这是我新号码。有急事需要你帮我转点钱，回头解释",
        "hi": "Mummy mera phone toot gaya, ye mera naya number hai. Aaj ek payment karni hai, help karo please",
    }
    return msg.get(lang, msg["en"])


@phishing
def vishing_callback(lang):
    ph = rng_phone()
    msg = {
        "en": random.choice([
            f"PayPal: You sent {rng_amount(lang)} to GeekTech Store. If you did not authorize this, call {ph} within 24 hrs.",
            f"Your bank card has been deactivated. Call {ph} now to reactivate. Do NOT visit a branch.",
            f"Geek Squad: Your {rng_amount(lang)} annual plan auto-renews today. To cancel, call {ph} immediately.",
        ]),
        "es": f"Se ha realizado un cargo de {rng_amount(lang)} en su cuenta. Si no lo reconoce, llame al {ph} de inmediato.",
        "fr": f"Un prélèvement de {rng_amount(lang)} a été effectué. Si vous n'êtes pas à l'origine, appelez le {ph} sous 24h.",
        "de": f"Eine Abbuchung von {rng_amount(lang)} wurde vorgenommen. Falls nicht autorisiert, rufen Sie {ph} an.",
        "pt": f"Uma cobrança de {rng_amount(lang)} foi feita. Se não reconhece, ligue para {ph} imediatamente.",
        "it": f"Addebito di {rng_amount(lang)} effettuato. Se non riconosci, chiama il {ph} entro 24h.",
        "nl": f"Er is {rng_amount(lang)} afgeschreven. Niet herkend? Bel direct {ph}.",
        "ru": f"С вашей карты списано {rng_amount(lang)}. Если это не вы, срочно позвоните {ph}.",
        "tr": f"Hesabınızdan {rng_amount(lang)} çekildi. Tanımıyorsanız hemen {ph} numarasını arayın.",
        "hi": f"Aapke account se {rng_amount(lang)} cut hua hai. Agar aapne nahi kiya to turant {ph} par call karein.",
    }
    return msg.get(lang, msg["en"])


@phishing
def bec_giftcard(lang):
    msg = {
        "en": random.choice([
            "Boss here. I'm in a meeting, can't talk. Buy 5 Apple gift cards $100 each for client gifts. Send me the codes, will reimburse today.",
            "Hi, this is HR. We're updating payroll. Please confirm your bank account details before Friday's run: " + rng_badurl("payroll"),
            "Are you at your desk? I need you to process an urgent vendor payment, keep this confidential for now. Reply when ready.",
        ]),
        "es": "Soy tu jefe, estoy en una reunión. Necesito que compres 5 tarjetas de regalo de Apple de $100 para clientes. Mándame los códigos, te reembolso hoy.",
        "fr": "C'est le directeur. Je suis en réunion. Achète 5 cartes cadeaux Apple de 100€ pour des clients et envoie-moi les codes, je te rembourse.",
        "de": "Hier ist der Chef. Bin im Meeting. Kauf bitte 5 Apple-Geschenkkarten zu je 100€ für Kunden und schick mir die Codes.",
        "pt": "Aqui é o chefe, estou em reunião. Compre 5 cartões-presente Apple de $100 para clientes e me envie os códigos, reembolso hoje.",
    }
    return msg.get(lang, msg["en"])


@phishing
def otp_theft(lang):
    msg = {
        "en": "Your WhatsApp account will be deactivated. To keep it active, forward the 6-digit code we just sent to your number.",
        "es": "Tu cuenta de WhatsApp será desactivada. Para mantenerla, reenvía el código de 6 dígitos que acabamos de enviarte.",
        "fr": "Votre compte WhatsApp va être désactivé. Pour le garder, transférez le code à 6 chiffres que nous venons d'envoyer.",
        "de": "Ihr WhatsApp-Konto wird deaktiviert. Leiten Sie den soeben gesendeten 6-stelligen Code weiter, um es zu behalten.",
        "pt": "Sua conta do WhatsApp será desativada. Para mantê-la, encaminhe o código de 6 dígitos que acabamos de enviar.",
        "ru": "Ваш аккаунт WhatsApp будет деактивирован. Перешлите 6-значный код, который мы только что отправили.",
        "hi": "Aapka WhatsApp account band ho jayega. Active rakhne ke liye abhi bheja 6-digit code forward karein.",
    }
    return msg.get(lang, msg["en"])


@phishing
def crypto_scare(lang):
    msg = {
        "en": random.choice([
            f"Coinbase: Withdrawal of 0.45 BTC initiated. If this wasn't you, cancel immediately: {rng_badurl('securityhold')}",
            f"Your MetaMask wallet will be suspended. Validate your seed phrase to keep access: {rng_badurl('restore')}",
        ]),
        "es": f"Coinbase: Se inició un retiro de 0.45 BTC. Si no fue usted, cancele aquí: {rng_badurl('seguridad')}",
        "de": f"Coinbase: Auszahlung von 0,45 BTC eingeleitet. Falls nicht von Ihnen, hier stornieren: {rng_badurl('stornieren')}",
        "ru": f"Coinbase: Инициирован вывод 0.45 BTC. Если это не вы, отмените: {rng_badurl('otmena')}",
    }
    return msg.get(lang, msg["en"])


@phishing
def gov_refund_scam(lang):
    msg = {
        "en": f"IRS: You have a pending tax refund of {rng_amount(lang)}. Claim before the deadline: {rng_badurl('refund')}",
        "es": f"Agencia Tributaria: Tiene un reembolso pendiente de {rng_amount(lang)}. Reclámelo: {rng_badurl('reembolso')}",
        "fr": f"Impots.gouv: Vous avez droit à un remboursement de {rng_amount(lang)}. Confirmez vos coordonnées: {rng_badurl('remboursement')}",
        "de": f"Finanzamt: Ihnen steht eine Erstattung von {rng_amount(lang)} zu. Bestätigen Sie Ihre Daten: {rng_badurl('erstattung')}",
        "pt": f"Receita Federal: Você tem uma restituição de {rng_amount(lang)}. Confirme seus dados: {rng_badurl('restituicao')}",
        "it": f"Agenzia Entrate: Hai diritto a un rimborso di {rng_amount(lang)}. Conferma i tuoi dati: {rng_badurl('rimborso')}",
        "nl": f"Belastingdienst: U krijgt {rng_amount(lang)} terug. Bevestig uw gegevens: {rng_badurl('teruggave')}",
    }
    return msg.get(lang, msg["en"])


@phishing
def extortion(lang):
    msg = {
        "en": "I know what you visited online. I recorded you through your camera. Pay $950 in Bitcoin within 48 hours or everyone in your contacts sees the video.",
        "es": "Sé lo que viste en internet. Te grabé con tu cámara. Paga $950 en Bitcoin en 48 horas o tus contactos verán el video.",
        "de": "Ich weiß, welche Seiten Sie besucht haben. Ich habe Sie über die Kamera aufgenommen. Zahlen Sie 950$ in Bitcoin, sonst sehen es alle Kontakte.",
        "fr": "Je sais ce que vous avez regardé. Je vous ai filmé avec votre caméra. Payez 950$ en Bitcoin sous 48h sinon tous vos contacts verront la vidéo.",
    }
    return msg.get(lang, msg["en"])


@phishing
def obfuscated(lang):
    """English-only obfuscation variants (homoglyph/zero-width/leet/spaced)."""
    base = random.choice([
        f"{bank('en')} Alert: Your account has been locked. Verify now at {rng_badurl('secure')}",
        f"Your package needs a fee. Pay at {rng_shorturl()}",
        f"PayPal: your account is limited. Restore access: {rng_badurl('restore')}",
        "Your card is suspended. Call 888-402-1176 to re-activate.",
    ])
    return random.choice([homoglyph, zero_width, leet, spaced])(base)


# ---- SPAM archetypes ----

@spam
def promo_spam(lang):
    pct = random.choice([50, 70, 80, 90])
    msg = {
        "en": f"FLASH SALE {pct}% OFF everything today only! Shop now: {rng_shorturl()} Reply STOP to opt out",
        "es": f"¡OFERTA! {pct}% de descuento solo hoy. Compra ya: {rng_shorturl()}",
        "fr": f"PROMO! -{pct}% sur tout aujourd'hui seulement. Achetez: {rng_shorturl()}",
        "de": f"FLASH SALE {pct}% Rabatt nur heute! Jetzt kaufen: {rng_shorturl()}",
        "pt": f"PROMOÇÃO! {pct}% de desconto só hoje. Compre já: {rng_shorturl()}",
        "it": f"SALDI {pct}% di sconto solo oggi! Acquista: {rng_shorturl()}",
        "nl": f"UITVERKOOP {pct}% korting alleen vandaag! Shop nu: {rng_shorturl()}",
        "ar": f"تخفيضات {pct}% على كل المنتجات اليوم فقط! تسوق الآن: {rng_shorturl()}",
        "he": f"מבצע! {pct}% הנחה היום בלבד! לרכישה: {rng_shorturl()}",
        "ru": f"РАСПРОДАЖА! Скидка {pct}% только сегодня! Купить: {rng_shorturl()}",
        "id": f"PROMO {pct}% diskon hari ini saja! Beli sekarang: {rng_shorturl()}",
        "tr": f"İNDİRİM! Bugüne özel %{pct} indirim! Hemen al: {rng_shorturl()}",
        "ja": f"本日限定{pct}%OFFセール！今すぐ購入: {rng_shorturl()}",
        "zh": f"限时{pct}折！仅限今天，立即抢购：{rng_shorturl()}",
        "hi": f"SALE! Aaj sirf {pct}% off. Abhi khareedein: {rng_shorturl()}",
    }
    return msg.get(lang, msg["en"])


@spam
def loan_spam(lang):
    msg = {
        "en": f"Need cash fast? Get up to $5,000 deposited TODAY. Bad credit OK! Apply: {rng_shorturl()}",
        "es": f"¿Necesitas dinero? Hasta 5.000€ hoy mismo. Sin aval. Solicita: {rng_shorturl()}",
        "fr": f"Crédit rapide jusqu'à 10 000€ en 24h sans justificatif. Demandez: {rng_shorturl()}",
        "de": f"Schnellkredit bis 5.000€ heute! Auch bei negativer Schufa. Beantragen: {rng_shorturl()}",
        "pt": f"Empréstimo rápido até R$5.000 hoje! Nome sujo aprova. Solicite: {rng_shorturl()}",
        "id": f"Pinjaman cepat cair hari ini hingga 10 juta tanpa jaminan! Daftar: {rng_shorturl()}",
        "hi": f"Turant loan chahiye? Aaj hi 50,000 tak. Bina document. Apply: {rng_shorturl()}",
    }
    return msg.get(lang, msg["en"])


@spam
def gambling_spam(lang):
    bonus = rng_amount(lang)
    msg = {
        "en": f"WIN BIG! Mega Jackpot Casino - 200 FREE spins, no deposit! Play: {rng_shorturl()} 18+",
        "es": f"¡GANA! Casino VULKAN regala {bonus} a nuevos jugadores: {rng_shorturl()} +18",
        "ru": f"Раздача! {bonus} каждому новому игроку казино: {rng_shorturl()} 18+",
        "tr": f"Bahis sitemize üye ol, {bonus} deneme bonusu kap: {rng_shorturl()}",
        "de": f"200 Freispiele ohne Einzahlung im Jackpot Casino! Jetzt spielen: {rng_shorturl()}",
        "pt": f"Cassino online: {bonus} de bônus grátis para novos jogadores! {rng_shorturl()}",
    }
    return msg.get(lang, msg["en"])


@spam
def health_spam(lang):
    msg = {
        "en": "Lose 15 lbs in 2 weeks with this 1 weird trick doctors HATE! Free trial: " + rng_shorturl(),
        "es": "¡Pierde 10 kilos en 2 semanas! Producto natural, prueba gratis: " + rng_shorturl(),
        "fr": "Perdez 10 kg en 2 semaines avec cette astuce! Essai gratuit: " + rng_shorturl(),
        "hi": "Bina diet weight ghatao! Ayurvedic capsule sirf 999. Order: " + rng_shorturl(),
        "ar": "خصم 70% على منتجات التخسيس! اطلب الآن: " + rng_shorturl(),
    }
    return msg.get(lang, msg["en"])


@spam
def job_spam(lang):
    msg = {
        "en": f"Earn $3000/week working from home! No experience needed. Limited spots: {rng_shorturl()}",
        "es": f"¡Gana 800€/semana desde casa! Sin experiencia. Únete: {rng_shorturl()}",
        "fr": f"Gagnez 700€/semaine depuis chez vous! Aucune expérience. Rejoignez: {rng_shorturl()}",
        "id": f"Kerja dari rumah hingga 5 juta/minggu tanpa pengalaman! Gabung: {rng_shorturl()}",
        "zh": f"网赚兼职日入500元，无需经验，加微信：zq{rng_code(5)}",
        "ja": f"在宅ワークで月50万円！スマホだけでOK。登録: {rng_shorturl()}",
    }
    return msg.get(lang, msg["en"])


# ---- HAM archetypes (incl. HARD NEGATIVES) ----

@ham
def legit_otp(lang):
    code = rng_code(random.choice([4, 6]))
    brand = random.choice(["WhatsApp", "Google", "Microsoft", "Uber", "Amazon", "Netflix", bank(lang)])
    msg = {
        "en": random.choice([
            f"Your {brand} verification code is {code}. Never share this code. We will never call you to ask for it.",
            f"{code} is your {brand} verification code.",
            f"G-{code} is your Google verification code.",
        ]),
        "es": f"Tu código de verificación de {brand} es {code}. No lo compartas con nadie.",
        "fr": f"Votre code de vérification {brand} est {code}. Ne le partagez avec personne.",
        "de": f"Ihr {brand} Bestätigungscode ist {code}. Geben Sie ihn niemandem weiter.",
        "pt": f"Seu código de verificação {brand}: {code}. Não compartilhe com ninguém.",
        "it": f"Il tuo codice di verifica {brand} è {code}. Non condividerlo con nessuno.",
        "nl": f"Je {brand} verificatiecode is {code}. Deel deze met niemand.",
        "ar": f"رمز التحقق الخاص بـ {brand} هو {code}. لا تشاركه مع أي شخص.",
        "he": f"קוד האימות שלך ל-{brand} הוא {code}. אל תשתף אותו עם אף אחד.",
        "ru": f"Ваш код подтверждения {brand}: {code}. Никому не сообщайте его.",
        "id": f"Kode verifikasi {brand} Anda adalah {code}. Jangan berikan kepada siapa pun.",
        "tr": f"{brand} doğrulama kodunuz {code}. Kimseyle paylaşmayın.",
        "ja": f"{brand}の確認コードは{code}です。誰にも教えないでください。",
        "zh": f"您的{brand}验证码是{code}，请勿告诉他人。",
        "hi": f"Aapka {brand} verification code hai {code}. Kisi ke saath share na karein.",
    }
    return msg.get(lang, msg["en"])


@ham
def legit_fraud_alert(lang):
    bk = bank(lang)
    amt = rng_amount(lang)
    msg = {
        "en": random.choice([
            f"{bk}: Your card was used for {amt} at WALMART #2891. Reply YES if this was you, NO if not.",
            f"{bk}: A new device signed in to your account. If this was you, no action needed.",
        ]),
        "es": f"{bk}: Se usó su tarjeta por {amt} en un comercio. Responda SÍ si fue usted, NO si no.",
        "fr": f"{bk}: Votre carte a été utilisée pour {amt}. Répondez OUI si c'est vous, NON sinon.",
        "de": f"{bk}: Ihre Karte wurde für {amt} verwendet. Antworten Sie JA wenn das Sie waren, sonst NEIN.",
        "pt": f"{bk}: Seu cartão foi usado em {amt}. Responda SIM se foi você, NÃO caso contrário.",
        "hi": f"{bk}: Aapke card se {amt} ka transaction hua. Reply YES agar aapne kiya, NO agar nahi.",
    }
    return msg.get(lang, msg["en"])


@ham
def legit_delivery(lang):
    c = courier(lang)
    tn = rng_code(random.choice([10, 12]))
    msg = {
        "en": random.choice([
            f"{c}: Your package {tn} was delivered to your mailbox at 2:14 PM.",
            f"{c}: Your order has shipped and arrives Thursday by 8 PM. Track at {c.lower().replace(' ','')}.com",
            f"{c}: Out for delivery today. Your driver is Marcus in a grey van.",
        ]),
        "es": f"{c}: Su paquete {tn} fue entregado hoy a las 14:14.",
        "fr": f"{c}: Votre colis {tn} sera livré demain entre 9h et 12h.",
        "de": f"{c}: Ihr Paket {tn} wurde heute zugestellt.",
        "pt": f"{c}: Sua encomenda {tn} foi entregue hoje às 14h.",
        "it": f"{c}: Il tuo pacco {tn} è stato consegnato oggi.",
        "nl": f"{c}: Je pakket {tn} is vandaag bezorgd.",
        "ru": f"{c}: Ваша посылка {tn} доставлена сегодня.",
        "id": f"{c}: Paket Anda {tn} telah dikirim, tiba besok.",
        "tr": f"{c}: Kargonuz {tn} bugün teslim edildi.",
        "ja": f"【{c}】お荷物{tn}を本日お届けしました。",
        "zh": f"【{c}】您的包裹{tn}已于今日送达。",
        "hi": f"{c}: Aapka parcel {tn} aaj deliver ho gaya.",
    }
    return msg.get(lang, msg["en"])


@ham
def legit_reminder(lang):
    msg = {
        "en": random.choice([
            "Reminder: Your dental appointment with Dr. Patel is tomorrow at 9:30 AM. Reply C to confirm or R to reschedule.",
            "Your prescription is ready for pickup at the pharmacy on Oak St. Questions? Call the number on your bottle.",
            "Your table for 4 is confirmed for 7:30 PM tonight. Reply CANCEL to release the reservation.",
        ]),
        "es": "Recordatorio: Su cita con el Dr. Pérez es mañana a las 9:30. Responda C para confirmar.",
        "fr": "Rappel: Votre rendez-vous chez le médecin est demain à 9h30. Répondez C pour confirmer.",
        "de": "Erinnerung: Ihr Termin beim Bürgeramt ist morgen um 10:15 Uhr. Bitte Ausweis mitbringen.",
        "pt": "Lembrete: Sua consulta é amanhã às 9h30. Responda C para confirmar.",
        "hi": "Reminder: Kal aapka doctor appointment 9:30 baje hai. Confirm karne ke liye C reply karein.",
    }
    return msg.get(lang, msg["en"])


@ham
def personal(lang):
    msg = {
        "en": random.choice([
            "Hey, are we still meeting for coffee at 3pm tomorrow?",
            "Running 10 mins late, order me a flat white if you get there first",
            "Can you send me the $40 for the concert tickets when you get a chance?",
            "Dinner at ours Saturday? Bring that wine you mentioned",
            "Did you catch the game last night? That last-minute goal was insane",
        ]),
        "es": random.choice([
            "Oye, ¿seguimos para el café mañana a las 3?",
            "Mamá ya llegué a casa, te llamo después de cenar, besos",
            "¿Me pasas los 40€ del concierto cuando puedas?",
        ]),
        "fr": "Salut, on se voit toujours pour le café demain à 15h? Tu peux me passer les 20€ quand tu peux",
        "de": "Hey, treffen wir uns morgen um 15 Uhr auf einen Kaffee? Bring bitte das Buch mit",
        "pt": "Oi amor, cheguei bem, te ligo mais tarde. Não esquece de me mandar os 40 do show",
        "it": "Ciao, ci vediamo domani alle 15 per un caffè? Portami quel libro se puoi",
        "nl": "Hoi, zien we elkaar morgen om 15 uur voor koffie? Neem dat boek mee",
        "ar": "أمي وصلت بالسلامة، سأتصل بك مساءً. لا تنسي تجيبي الكتاب",
        "he": "אמא, אני מאחר בעשר דקות, תתחילו לאכול בלעדיי",
        "ru": "Привет! Встретимся завтра в 18:00 у метро, как договаривались?",
        "id": "Halo, besok jadi ketemu jam 3 buat kopi? Bawain bukunya ya",
        "tr": "Selam, yarın saat 3'te kahve için buluşuyor muyuz? Kitabı getir lütfen",
        "ja": "お母さん、今駅に着いたよ。夕飯までには帰る。",
        "zh": "妈，我到学校了，晚上视频聊。记得帮我带下书",
        "hi": "Beta khana kha lena, main der se aaungi aaj office se. Doodh laana yaad rakhna",
    }
    return msg.get(lang, msg["en"])


@ham
def legit_billing(lang):
    amt = rng_amount(lang)
    msg = {
        "en": f"Your electricity bill of {amt} is due Jun 21. Pay in the app or at your provider's website to avoid late fees.",
        "es": f"Su factura de luz de {amt} vence el 21 de junio. Pague en la app para evitar recargos.",
        "fr": f"Votre facture d'électricité de {amt} est due le 21 juin. Payez dans l'application.",
        "de": f"Ihre Stromrechnung über {amt} ist am 21. Juni fällig. Zahlen Sie in der App.",
        "pt": f"Sua conta de luz de {amt} vence em 21/06. Pague no aplicativo.",
    }
    return msg.get(lang, msg["en"])


def build(n_per_template, langs, seed=1234):
    random.seed(seed)
    rows = []
    groups = [("phishing", PHISHING_BUILDERS), ("spam", SPAM_BUILDERS), ("ham", HAM_BUILDERS)]
    for label, builders in groups:
        for fn in builders:
            for lang in langs:
                for _ in range(n_per_template):
                    try:
                        text = fn(lang)
                    except Exception:
                        continue
                    if text and text.strip():
                        rows.append((text.strip(), label))
    # de-dup exact repeats (templates with no random content collapse)
    seen = set()
    uniq = []
    for text, label in rows:
        key = (text, label)
        if key not in seen:
            seen.add(key)
            uniq.append((text, label))
    random.shuffle(uniq)
    return uniq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-template", type=int, default=6)
    ap.add_argument("--langs", default=",".join(LANGS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    langs = args.langs.split(",")
    rows = build(args.n_per_template, langs, seed=args.seed)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["text", "label"])
        w.writerows(rows)

    from collections import Counter
    dist = Counter(label for _, label in rows)
    print(f"Wrote {len(rows)} rows to {out}")
    print(f"Distribution: {dict(dist)}")


if __name__ == "__main__":
    main()
