#!/usr/bin/env python3
"""Conversational and transactional archetypes model 2.7 gets wrong, with benign twins.

The bench in evals/datasets/bench100_v1.csv showed three gaps: social-engineering
phishing with no link and no brand (new-number family scams, code-relay requests,
wrong-number openers, fake order confirmations with a callback number), prize and
job lures that the corpus labels spam, and real transactional notices that get
blocked. evals/generate_synthetic.py already has these archetypes, but each one
is a single template per language, so the model learned the surface form rather
than the meaning. This generator builds every message from independent pools
(opener, reason, ask, closer, names, brands, amounts) so no two rows share a
template, and emits the benign twin of every attack.

Rows are intended labels. Nothing here trains directly: evals/distill_labels.py
asks TypeSafe about every row and keeps only those where the teacher agrees or is
very confident.

    python evals/generate_archetypes.py --n 300 --seed 11 \
        --out src/mBERT/training/model-training/dataset/curated/synthetic_archetypes_v1.csv
"""
import argparse
import csv
import random
from pathlib import Path

R = random.Random()

# ----------------------------------------------------------------- value pools
FIRST = ["Sarah", "Mike", "Emma", "Tom", "Liam", "Olivia", "Noah", "Ava", "Jake", "Lucy", "Ben", "Mia", "Chris", "Zoe",
         "Carlos", "Lucía", "Mateo", "Sofía", "Pierre", "Camille", "Louis", "Léa", "Jonas", "Lena", "Max", "Anna",
         "João", "Ana", "Pedro", "Beatriz", "Ahmed", "Fatima", "Omar", "Layla", "Rahul", "Priya", "Arjun", "Neha",
         "Yuki", "Kenji", "Wei", "Mei", "Ivan", "Olga", "Dmitri", "Natasha"]
FAMILY = {"en": ["Mum", "Mom", "Dad", "Mama", "Papa", "Nan", "Grandma", "Grandpa", "Auntie", "Uncle", "Sis", "Bro"],
          "es": ["Mamá", "Papá", "Abuela", "Abuelo", "Tía", "Tío"], "fr": ["Maman", "Papa", "Mamie", "Papi", "Tata", "Tonton"],
          "de": ["Mama", "Papa", "Oma", "Opa"], "pt": ["Mãe", "Pai", "Vó", "Vô", "Tia", "Tio"],
          "ar": ["ماما", "بابا", "خالتي", "عمي", "جدتي"], "hi": ["Mummy", "Papa", "Didi", "Bhaiya", "Nani", "Dadi"]}
BRANDS_SHOP = ["Amazon", "Walmart", "Target", "Best Buy", "eBay", "Apple", "Costco", "Home Depot", "Argos", "Currys",
               "MediaMarkt", "Fnac", "El Corte Inglés", "Mercado Livre", "Flipkart", "Noon", "Jumia"]
BRANDS_SUB = ["Netflix", "Spotify", "Disney+", "Hulu", "Amazon Prime", "Apple Music", "YouTube Premium", "HBO Max",
              "Xbox Game Pass", "PlayStation Plus", "Canva", "Dropbox", "iCloud", "Google One", "Adobe"]
BRANDS_JOB = ["Amazon", "TikTok", "Shein", "Walmart", "Temu", "Netflix", "Uber", "DHL", "Costco", "Google", "Marriott",
              "Deloitte", "Tesla", "Zara", "IKEA"]
GIFTCARD = ["Walmart", "Amazon", "Target", "Costco", "Apple", "Tesco", "Sainsbury's", "Carrefour", "Lidl", "Aldi", "Starbucks",
            "Shell", "Best Buy", "Home Depot"]
BANKS = ["Chase", "Wells Fargo", "Bank of America", "Citi", "Capital One", "TD Bank", "US Bank", "PNC", "Truist",
         "Barclays", "HSBC", "Lloyds", "NatWest", "Santander", "Monzo", "Revolut", "BBVA", "CaixaBank", "BNP Paribas",
         "Crédit Agricole", "Sparkasse", "Commerzbank", "ING", "Itaú", "Nubank", "Bradesco", "SBI", "HDFC Bank", "ICICI",
         "Al Rajhi", "Emirates NBD", "QNB", "Rakuten Bank", "DBS", "Maybank", "BCA", "Scotiabank", "RBC", "Commonwealth Bank"]
TELCOS = ["Verizon", "AT&T", "T-Mobile", "Vodafone", "EE", "O2", "Three", "Orange", "SFR", "Telekom", "Movistar", "Telcel",
          "Claro", "Vivo", "TIM", "Airtel", "Jio", "STC", "Etisalat", "Zain", "Telstra", "Rogers", "Bell"]
UTIL = ["ConEd", "PG&E", "Duke Energy", "British Gas", "EDF", "Enel", "Iberdrola", "E.ON", "Xcel Energy", "National Grid",
        "Thames Water", "Vattenfall", "Engie", "Naturgy"]
COURIERS = ["USPS", "FedEx", "UPS", "DHL", "Royal Mail", "Evri", "DPD", "Hermes", "Correos", "La Poste", "Chronopost",
            "Correios", "Aramex", "Canada Post", "Australia Post", "PostNL", "Bpost", "Yodel", "GLS", "Amazon Logistics"]
GOV = ["IRS", "HMRC", "Social Security Administration", "Medicare", "DMV", "DVLA", "CRA", "ATO", "the Tax Office",
       "Department of Motor Vehicles", "Centrelink", "CAF", "URSSAF", "Finanzamt", "Agencia Tributaria", "Receita Federal"]
CITIES = ["Chicago", "Dallas", "Miami", "Seattle", "London", "Manchester", "Madrid", "Paris", "Berlin", "Lisbon", "Dubai",
          "Mumbai", "Toronto", "Sydney", "Lagos", "Nairobi", "Moscow", "Kyiv", "Jakarta", "Manila"]
REAL_DOMAINS = {"Amazon": "amazon.com", "Walmart": "walmart.com", "Target": "target.com", "Best Buy": "bestbuy.com",
                "eBay": "ebay.com", "Apple": "apple.com", "Netflix": "netflix.com", "Spotify": "spotify.com",
                "Verizon": "myvzw.com", "AT&T": "att.com", "T-Mobile": "t-mobile.com", "Vodafone": "vodafone.co.uk",
                "Chase": "chase.com", "Wells Fargo": "wellsfargo.com", "PayPal": "paypal.com", "USPS": "usps.com",
                "FedEx": "fedex.com", "UPS": "ups.com", "DHL": "dhl.com", "Royal Mail": "royalmail.com"}
BAD_TLDS = [".top", ".xyz", ".info", ".online", ".site", ".club", ".icu", ".buzz", ".cc", ".biz", ".click", ".link", ".vip", ".win"]
SHORT = ["bit.ly", "tinyurl.com", "t.co", "cutt.ly", "rb.gy", "is.gd", "shorturl.at"]
ITEMS = ["MacBook Pro", "iPhone 16 Pro", "PlayStation 5", "Samsung 65\" TV", "Dyson V15", "AirPods Pro", "Nintendo Switch",
         "Bose headphones", "Ring doorbell", "Kindle Paperwhite", "GoPro Hero 13", "Apple Watch", "Xbox Series X",
         "Bluetooth speaker", "running shoes", "a winter coat", "printer ink", "dog food", "a phone case", "a desk lamp"]
DEPTS = ["HR", "Recruitment", "Talent Acquisition", "Hiring Team", "People Ops", "Staffing"]
JOB_TITLES = ["data entry", "product reviewer", "remote assistant", "order processor", "customer support", "app tester",
              "social media assistant", "typist", "survey taker", "package handler", "virtual assistant", "content rater"]
TASKS = ["like videos", "rate products", "follow accounts", "review hotels", "watch ads", "boost listings", "complete surveys",
         "write short reviews", "test apps", "share posts"]
REASONS_NEWPHONE = ["dropped my phone in the sink", "my phone got stolen", "smashed my screen", "lost my phone on the train",
                    "my phone died for good", "phone fell in the pool", "my old sim got blocked", "washed my phone with the laundry",
                    "phone got run over", "my number got cut off"]
URGENT_NEEDS = ["pay a bill that's overdue today", "sort out my rent before 5pm", "pay for the phone repair", "cover a payment "
                "that bounced", "settle a fine before it doubles", "pay my car insurance", "clear an overdraft before it charges me",
                "buy a new phone today", "pay a deposit for the flat", "cover my card that got blocked"]
GREET = ["Hi", "Hey", "Hello", "Hiya", "Yo", "Hi there", "Heyy", "Hello?", "Hey!!", "Hi,"]
CLOSERS = ["x", "xx", "thanks", "thank you!", "pls", "please", "love you", "ok?", "asap", "ttyl", "", "", "", "🙏", "❤️", "😊"]
APOLOGY = ["Sorry if wrong number!", "Apologies if I have the wrong number.", "Sorry, is this the right number?",
           "Hope this is still your number.", "Not sure if this is still you.", "Sorry to bother you."]
CODE_EXCUSE = ["I put your number in by mistake", "I typed your number instead of mine", "my phone is acting up",
               "it sent the code to your phone somehow", "I'm locked out and it went to you", "I used your number for my account"]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday", "tomorrow", "today", "tonight", "this weekend"]
TIMES = ["8am", "9:30am", "10am", "noon", "1pm", "2:15pm", "3pm", "4:45pm", "6pm", "7pm", "8:30pm"]


def money():
    return R.choice(["$", "$", "$", "£", "€", "₹", "R$", "AED "]) + R.choice(
        [str(R.randint(9, 99)), f"{R.randint(100, 999)}", f"{R.randint(1000, 4999):,}", f"{R.randint(10, 999)}.{R.randint(0, 99):02d}"])


def small_money():
    return R.choice(["$", "£", "€"]) + R.choice(["0.99", "1.49", "1.99", "2.50", "2.99", "3.20", "4.99", "5.00"])


def big_money():
    return R.choice(["$", "£", "€"]) + R.choice(["500", "750", "1,000", "1,500", "2,000", "5,000", "10,000", "50,000", "250,000", "850,000"])


def phone():
    return R.choice([f"+1 {R.randint(201, 989)} {R.randint(200, 999)} {R.randint(1000, 9999)}",
                     f"1-8{R.choice('0578')}{R.randint(0, 9)}-{R.randint(200, 999)}-{R.randint(1000, 9999)}",
                     f"({R.randint(201, 989)}) {R.randint(200, 999)}-{R.randint(1000, 9999)}",
                     f"+44 7{R.randint(100, 999)} {R.randint(100000, 999999)}",
                     f"+34 6{R.randint(10, 99)} {R.randint(100, 999)} {R.randint(100, 999)}",
                     f"+91 {R.randint(70000, 99999)} {R.randint(10000, 99999)}",
                     f"+971 5{R.randint(0, 8)} {R.randint(100, 999)} {R.randint(1000, 9999)}",
                     f"0{R.randint(6, 7)} {R.randint(10, 99)} {R.randint(10, 99)} {R.randint(10, 99)} {R.randint(10, 99)}"])


def order_no():
    return R.choice([f"#{R.randint(100, 999)}-{R.randint(1000000, 9999999)}-{R.randint(1000000, 9999999)}",
                     f"#{R.randint(10000000, 99999999)}", f"{R.choice('ABCDEFGH')}{R.randint(100000, 999999)}",
                     f"#{R.randint(1000, 9999)}-{R.randint(10000, 99999)}"])


def code():
    return R.choice([f"{R.randint(100000, 999999)}", f"{R.randint(100, 999)} {R.randint(100, 999)}", f"{R.randint(1000, 9999)}",
                     f"{R.randint(100, 999)}-{R.randint(100, 999)}"])


def card_last4():
    return f"{R.randint(1000, 9999)}"


def bad_url(*words):
    w = R.choice(words) if words else "secure"
    host = R.choice([f"{w}-{R.choice(['verify', 'secure', 'update', 'claim', 'pay', 'help', 'support', 'alert', 'notice'])}",
                     f"{R.choice(['my', 'get', 'go', 'e', 'app'])}-{w}", f"{w}{R.randint(1, 99)}", f"{w}-{R.choice(CITIES).lower()}"])
    url = host.replace(" ", "").replace("'", "").lower() + R.choice(BAD_TLDS)
    if R.random() < 0.35:
        url = f"https://{R.choice(SHORT)}/{R.choice('abcdefghjkmnpqrstuvwxyz')}{R.randint(1000, 99999)}"
    elif R.random() < 0.4:
        url = "http://" + url
    if R.random() < 0.5:
        url += "/" + R.choice(["claim", "verify", "pay", "update", "login", "confirm", "id", "secure", "refund"])
    return url


def real_url(brand):
    d = REAL_DOMAINS.get(brand, brand.lower().replace(" ", "").replace("'", "").replace("&", "") + ".com")
    return R.choice([d, f"{d}/orders", f"{d}/account", f"{d}/track", f"the {brand} app", f"the {brand} app or {d}"])


def wa():
    return R.choice([f"WhatsApp {phone()}", f"WhatsApp me on {phone()}", f"message {phone()} on WhatsApp", f"wa.me/{R.randint(10000000000, 99999999999)}",
                     f"Telegram @{R.choice(['hr', 'jobs', 'recruit', 'team', 'talent'])}{R.randint(10, 999)}", f"reply YES", f"text {phone()}"])


def name():
    return R.choice(FIRST)


def fam(lang="en"):
    return R.choice(FAMILY.get(lang, FAMILY["en"]))


def maybe(s, p=0.5):
    return s if R.random() < p else ""


def join(*parts):
    return " ".join(p for p in parts if p).replace("  ", " ").strip()


# ----------------------------------------------------------------- builders
PH, SP, HM = [], [], []


def ph(f): PH.append(f); return f
def sp(f): SP.append(f); return f
def hm(f): HM.append(f); return f


# --- phishing: no link, no brand, pure social engineering
@ph
def new_number_family():
    f = fam()
    opener = R.choice([f"{R.choice(GREET)} {f}, it's me.", f"{R.choice(GREET)} {f} it's me", f"{f} it's me,", f"Hi {f}!!",
                       f"{f}, this is my new number.", f"{f} I {R.choice(REASONS_NEWPHONE)}", f"{R.choice(GREET)} {f}, I {R.choice(REASONS_NEWPHONE)} so this is my new number now",
                       f"{f} it's your {R.choice(['son', 'daughter', 'eldest', 'youngest', 'boy', 'girl'])}, new number"])
    mid = R.choice([f"Save this number and delete the old one.", f"Can you message me on WhatsApp on this number?", f"Text me back on here.",
                    f"This is my temporary number until I get the new phone.", f"Please reply here so I know you got this.", ""])
    ask = R.choice([f"I need to {R.choice(URGENT_NEEDS)} but my banking app is locked on the old phone, can you help?",
                    f"Can you {R.choice(['transfer', 'send', 'lend'])} me {money()} today? I'll pay you back {R.choice(DAYS)}.",
                    f"I can't get into my bank, can you pay something for me quickly?", f"I need a favour, it's a bit urgent.",
                    f"Can you send me your card details so I can order a replacement phone?", "I'm in a bit of a mess, can you help me out?",
                    f"Something's come up, are you free to help with a payment?", ""])
    return join(opener, mid, ask, R.choice(CLOSERS))


@ph
def new_number_family_i18n():
    return R.choice([
        f"Hola {fam('es')}, se me rompió el móvil y este es mi nuevo número. {R.choice(['Guárdalo y escríbeme por WhatsApp.', 'Necesito que me ayudes con un pago urgente.', '¿Puedes pasarme ' + money() + ' hoy? Mañana te lo devuelvo.'])}",
        f"Coucou {fam('fr')}, c'est moi, j'ai {R.choice(['cassé mon téléphone', 'perdu mon portable', 'changé de numéro'])}. {R.choice(['Écris-moi sur WhatsApp à ce numéro.', 'Tu peux me faire un virement de ' + money() + ' ? Je te rembourse vite.', 'J ai besoin d aide pour payer une facture urgente.'])}",
        f"Hallo {fam('de')}, ich bin's, mein Handy ist {R.choice(['kaputt', 'weg', 'ins Wasser gefallen'])}. Das ist meine neue Nummer. {R.choice(['Schreib mir bitte auf WhatsApp.', 'Kannst du mir schnell ' + money() + ' überweisen? Ich kann nicht in mein Banking.', 'Ich muss dringend eine Rechnung bezahlen, hilfst du mir?'])}",
        f"Oi {fam('pt')}, sou eu, {R.choice(['quebrei o celular', 'perdi o telefone', 'roubaram meu celular'])}, esse é meu número novo. {R.choice(['Salva aí e me chama no WhatsApp.', 'Consegue me mandar um Pix de ' + money() + ' hoje? Amanhã devolvo.', 'Preciso pagar um boleto urgente e não consigo entrar no banco.'])}",
        f"{fam('ar')} هذا رقمي الجديد، جوالي {R.choice(['انكسر', 'ضاع', 'تعطل'])}. {R.choice(['احفظي الرقم وكلميني واتساب.', 'ممكن تحولين لي ' + money() + ' اليوم وأرجعها لك بكرة؟', 'محتاج مساعدة عاجلة في دفع فاتورة.'])}",
        f"{fam('hi')} ye mera naya number hai, purana phone {R.choice(['toot gaya', 'kho gaya', 'chori ho gaya'])}. {R.choice(['Isko save kar lo aur WhatsApp pe msg karo.', 'Kya aap aaj ' + money() + ' bhej sakte ho? Kal wapas kar dunga.', 'Urgent bill bharna hai, bank app nahi khul raha.'])}",
    ])


@ph
def code_relay():
    return join(R.choice(GREET), R.choice([
        f"did you just get a {R.choice(['6 digit', 'six digit', '4 digit', 'verification', 'login', 'Google', 'WhatsApp', 'Facebook', 'bank', 'Venmo', 'PayPal'])} code? {R.choice(CODE_EXCUSE)}, can you send it to me?",
        f"can you send me the code you just received? {R.choice(CODE_EXCUSE)}.",
        f"a code should have come to your phone, {R.choice(CODE_EXCUSE)}. What is it?",
        f"quick one, forward me the text with the code that just came through, {R.choice(CODE_EXCUSE)}",
        f"I sent a verification code to your number by accident, can you read it to me? {R.choice(['Need it to log in.', 'It expires in 5 min.', 'Sorry!!'])}",
        f"what's the code in the message you just got? {R.choice(CODE_EXCUSE)}",
        f"hey it's {name()}, my account sent the code to your number, {R.choice(['pls send it', 'can you screenshot it', 'read it out to me'])}?"]),
        R.choice(CLOSERS))


@ph
def code_relay_i18n():
    return R.choice([
        f"Hola, ¿te llegó un código de {R.choice(['6 dígitos', 'verificación', 'WhatsApp', 'Bizum'])}? Puse tu número por error, ¿me lo pasas?",
        f"Salut, tu as reçu un code {R.choice(['à 6 chiffres', 'de vérification', 'WhatsApp'])} ? Je me suis trompé de numéro, tu peux me l'envoyer ?",
        f"Hallo, hast du gerade einen {R.choice(['6-stelligen', 'Bestätigungs', 'WhatsApp'])} Code bekommen? Ich habe aus Versehen deine Nummer eingegeben, schick ihn mir bitte.",
        f"Oi, chegou um código de {R.choice(['6 dígitos', 'verificação', 'WhatsApp', 'Pix'])} aí? Coloquei seu número sem querer, me manda?",
        f"مرحبا، وصلك كود {R.choice(['تحقق', 'واتساب', 'من 6 أرقام'])}؟ كتبت رقمك بالغلط، ممكن ترسله لي؟",
        f"Hi, ek {R.choice(['OTP', '6 digit code', 'WhatsApp code'])} aaya kya? Galti se tumhara number daal diya, please bhej do.",
    ])


@ph
def wrong_number_opener():
    n = name()
    return R.choice([
        f"{R.choice(GREET)} {n}, are you still coming to the {R.choice(['dinner', 'golf', 'party', 'yoga class', 'wedding'])} on {R.choice(DAYS)}?",
        f"Is this {n}? It's {name()}, we met at {R.choice(['the conference', 'Lisa\'s party', 'the gym', 'the wine tasting', 'the airport lounge'])} {R.choice(['last week', 'last month', 'in ' + R.choice(CITIES)])}.",
        f"{R.choice(GREET)}, long time no talk! How have you been? {R.choice(APOLOGY)}",
        f"Hi Dr. {n}, this is {name()}. Can we move my appointment to {R.choice(DAYS)}? {R.choice(APOLOGY)}",
        f"{n} it's {name()}, I got your number from {name()}. Are you free to chat {R.choice(DAYS)}?",
        f"Hey, is this still {n}'s number? Haven't heard from you since {R.choice(CITIES)}.",
        f"{R.choice(GREET)} {n}, did you get the {R.choice(['package', 'documents', 'photos', 'invoice'])} I sent? {R.choice(APOLOGY)}",
        f"Hi, I think we spoke about the {R.choice(['apartment', 'boat', 'investment', 'shipment'])} last week? This is {name()}.",
        f"{R.choice(GREET)} it's {name()}, we haven't spoken in ages, are you still in {R.choice(CITIES)}? {R.choice(APOLOGY)}",
    ])


@ph
def fake_order_callback():
    brand = R.choice(BRANDS_SHOP)
    item = R.choice(ITEMS)
    return R.choice([
        f"{brand}: Your order {order_no()} for {item} ({money()}) has {R.choice(['shipped', 'been confirmed', 'been placed', 'been charged to your card'])}. If you did not {R.choice(['place', 'authorize', 'make'])} this order, call {phone()} {R.choice(['immediately', 'within 24 hours', 'now to cancel', 'to dispute'])}.",
        f"Thank you for your {brand} purchase of {item}, total {money()}. Card ending {card_last4()} charged. Not you? Call our fraud line {phone()}.",
        f"{brand} Order Confirmation: {item} - {money()}. Delivery {R.choice(DAYS)}. To cancel or report unauthorized use call {phone()}.",
        f"Your {brand} account was used to buy {item} for {money()}. If this wasn't you contact support at {phone()} {R.choice(['within 2 hours', 'right away', 'before shipment'])}.",
        f"Receipt: {item} {money()} via {brand}. Ref {order_no()}. Questions or cancellation: {phone()} (24/7).",
        f"{brand}: a {item} was ordered from a new device in {R.choice(CITIES)}. Charged {money()}. Cancel: {phone()}",
    ])


@ph
def fake_subscription_charge():
    brand = R.choice(BRANDS_SUB)
    return R.choice([
        f"{brand}: Your {R.choice(['annual', 'monthly', 'premium', 'family'])} plan was charged {money()} to card ending {card_last4()}. Not you? Dispute here: {bad_url(brand.lower().split()[0])}",
        f"Your {brand} subscription renewed for {money()}. To cancel and get a refund call {phone()} within 24h.",
        f"{brand}: payment of {money()} {R.choice(['failed', 'was declined', 'could not be processed'])}. Update your billing within {R.choice(['24h', '48 hours', '2 days'])} or lose access: {bad_url('billing', 'payment', brand.lower().split()[0])}",
        f"{brand} receipt: {money()} charged {R.choice(DAYS)}. If you did not authorize this, cancel at {bad_url('refund', 'cancel')}",
    ])


@ph
def prize_win():
    g = R.choice(GIFTCARD)
    return R.choice([
        f"Congratulations! You've been {R.choice(['selected', 'chosen', 'picked'])} to receive a {big_money()} {g} gift card. Claim {R.choice(['here', 'now', 'within 24h'])}: {bad_url(g.lower().split()[0], 'rewards', 'winner')}",
        f"{g} {R.choice(['Rewards', 'Loyalty', 'Customer'])}: you have won {big_money()}! {R.choice(['Confirm your details', 'Claim your prize', 'Verify your address'])} at {bad_url('claim', 'prize', 'reward')}",
        f"You are our {R.choice(['1,000th', 'lucky', '100,000th', 'monthly'])} {R.choice(['visitor', 'customer', 'winner'])}! A {R.choice(ITEMS)} is waiting for you. Pay only shipping {small_money()}: {bad_url('win', 'prize')}",
        f"WINNER: {R.choice(['Your number', 'Your mobile', 'You'])} won {big_money()} in the {R.choice(['Mega', 'National', 'Mobile', 'Lucky', 'Anniversary'])} Draw. Send your name, address and bank details to claim{R.choice(['.', ' before ' + R.choice(DAYS) + '.'])} Ref {order_no()}",
        f"Your {R.choice(['loyalty points', 'reward points', 'cashback'])} of {money()} expire {R.choice(DAYS)}! Redeem now: {bad_url('points', 'rewards')}",
        f"Unclaimed prize! You were selected for a {R.choice(ITEMS)} from {g}. Reply YES with your address to receive it.",
        f"You've been chosen for a {money()} {R.choice(['refund', 'compensation', 'rebate', 'cash bonus'])} from {R.choice(GIFTCARD + BANKS)}. Enter your card to receive it: {bad_url('refund')}",
    ])


@ph
def prize_win_i18n():
    return R.choice([
        f"¡Felicidades! Has sido seleccionado para recibir una tarjeta regalo de {big_money()} de {R.choice(GIFTCARD)}. Reclama aquí: {bad_url('premio', 'regalo')}",
        f"Félicitations ! Vous avez gagné {big_money()} au tirage {R.choice(['mobile', 'national', 'anniversaire'])}. Confirmez vos coordonnées : {bad_url('gagnant', 'prix')}",
        f"Glückwunsch! Sie wurden für einen {big_money()} {R.choice(GIFTCARD)} Gutschein ausgewählt. Jetzt einlösen: {bad_url('gewinn', 'gutschein')}",
        f"Parabéns! Você foi selecionado para um vale de {big_money()} da {R.choice(GIFTCARD)}. Resgate agora: {bad_url('premio', 'vale')}",
        f"مبروك! تم اختيارك للفوز بقسيمة {big_money()} من {R.choice(GIFTCARD)}. اضغط للمطالبة: {bad_url('prize')}",
        f"Badhai ho! Aapko {big_money()} ka {R.choice(GIFTCARD)} gift card mila hai. Abhi claim karein: {bad_url('inaam', 'gift')}",
    ])


@ph
def job_scam():
    b = R.choice(BRANDS_JOB)
    return R.choice([
        f"Hi, this is {name()} from {b} {R.choice(DEPTS)}. {R.choice(['You have been selected', 'Your profile was shortlisted', 'We reviewed your resume and you qualify'])} for a {R.choice(['remote', 'part-time', 'work-from-home', 'flexible'])} {R.choice(JOB_TITLES)} role, {money()}/{R.choice(['day', 'hour', 'week'])}. {wa()} to start.",
        f"{b} is hiring! Earn {money()}-{money()} daily {R.choice(TASKS)}. No experience, {R.choice(['paid daily', 'same-day pay', 'instant withdrawal'])}. Ages 22+. {wa()}",
        f"Selected candidate: {b} needs {R.randint(5, 50)} people to {R.choice(TASKS)}, {money()}/day. Start today, {wa()}.",
        f"Dear applicant, congratulations, {b} has approved your {R.choice(JOB_TITLES)} application. Salary {money()} per {R.choice(['day', 'week'])}. Contact your manager {name()}: {wa()}",
        f"{name()} ({b} {R.choice(DEPTS)}): we saw your CV on {R.choice(['Indeed', 'LinkedIn', 'Glassdoor', 'a job board'])}. Online position, 1-2 hrs a day, {money()}/day. Interested? {wa()}",
        f"Part-time job offer from {b}: {R.choice(TASKS)} for {money()} per task, paid to your bank {R.choice(['instantly', 'daily', 'same day'])}. {wa()}",
    ])


@ph
def job_scam_i18n():
    b = R.choice(BRANDS_JOB)
    return R.choice([
        f"Hola, soy {name()} de RRHH de {b}. Has sido seleccionado para un trabajo remoto, {money()} al día. Escríbeme por WhatsApp {phone()}.",
        f"Bonjour, {name()} du service RH de {b}. Votre profil a été retenu pour un poste à domicile, {money()}/jour. Contactez-moi sur WhatsApp {phone()}.",
        f"Hallo, hier ist {name()} von {b} Recruiting. Sie wurden für eine Homeoffice-Stelle ausgewählt, {money()} pro Tag. Melden Sie sich per WhatsApp {phone()}.",
        f"Olá, sou {name()} do RH da {b}. Você foi selecionado para uma vaga home office, {money()} por dia. Me chama no WhatsApp {phone()}.",
        f"مرحبا، أنا {name()} من قسم التوظيف في {b}. تم اختيارك لوظيفة عن بعد براتب {money()} يومياً. تواصل واتساب {phone()}",
        f"Namaste, main {name()} {b} HR se. Aap part-time online job ke liye select hue hain, {money()}/day. WhatsApp karein {phone()}.",
    ])


@ph
def gov_callback():
    g = R.choice(GOV)
    return R.choice([
        f"{g}: Your {R.choice(['SSN', 'Social Security number', 'tax file', 'National Insurance number', 'Medicare number', 'driver license'])} has been {R.choice(['suspended', 'flagged', 'compromised', 'linked to fraud'])}. Call {phone()} {R.choice(['immediately', 'within 24 hours', 'to avoid arrest', 'to avoid legal action'])}.",
        f"Final notice from {g}: a warrant will be issued unless you settle {money()} today. Call {phone()} ref {order_no()}.",
        f"{g}: you are owed a {R.choice(['tax refund', 'rebate', 'benefit payment', 'stimulus payment'])} of {money()}. Submit your bank details at {bad_url('refund', 'gov', 'tax')} to receive it.",
        f"Your {R.choice(['Medicare', 'health card', 'benefits card', 'NHS'])} card must be replaced with a new {R.choice(['chip', 'digital', 'secure'])} card. Confirm your number here to keep coverage: {bad_url('medicare', 'card', 'benefits')}",
        f"{g} notice: {R.choice(['unpaid tax', 'an outstanding fine', 'a filing error'])} of {money()} found on your record. Pay by {R.choice(DAYS)} at {bad_url('pay', 'tax', 'fine')} or face prosecution.",
    ])


@ph
def account_problem_link():
    b = R.choice(BANKS + TELCOS + BRANDS_SUB + ["PayPal", "Venmo", "Cash App", "Coinbase", "Binance", "Instagram", "Facebook", "Microsoft", "Apple ID", "Google"])
    return R.choice([
        f"{b}: {R.choice(['unusual login', 'suspicious activity', 'a login from ' + R.choice(CITIES), 'a failed payment', 'a billing error'])} detected. {R.choice(['Verify', 'Confirm', 'Secure', 'Restore'])} your account {R.choice(['now', 'within 24h', 'immediately'])}: {bad_url(b.lower().split()[0], 'secure', 'verify')}",
        f"{b} Alert: your account will be {R.choice(['suspended', 'locked', 'closed', 'restricted'])} {R.choice(['today', 'in 24 hours', 'unless you act'])}. Update your details at {bad_url(b.lower().split()[0], 'update')}",
        f"{b}: your {R.choice(['password', 'card', 'KYC', 'ID verification', 'contact details'])} {R.choice(['expires today', 'has expired', 'needs re-verification'])}. Keep access: {bad_url(b.lower().split()[0], 'renew')}",
        f"{b}: a {R.choice(['wire transfer', 'withdrawal', 'payment'])} of {money()} is pending. Not you? Cancel it here: {bad_url('cancel', 'secure', b.lower().split()[0])}",
        f"{b} Security: enter your 2FA code at {bad_url(b.lower().split()[0], 'auth')} to confirm the pending {R.choice(['withdrawal', 'transfer', 'login'])} of {money()}.",
    ])


@ph
def delivery_fee():
    c = R.choice(COURIERS)
    return R.choice([
        f"{c}: your parcel {R.choice(['is held at customs', 'could not be delivered', 'is on hold', 'has an incomplete address'])}. Pay {small_money()} {R.choice(['redelivery', 'customs', 'handling'])} fee within {R.choice(['24h', '48h', '2 days'])}: {bad_url(c.lower().split()[0], 'parcel', 'track')}",
        f"{c}: we missed you {R.choice(DAYS)}. Reschedule delivery ({small_money()}) at {bad_url('redelivery', c.lower().split()[0])}",
        f"{c} notice: package {order_no()} returns to sender unless address is confirmed: {bad_url('confirm', c.lower().split()[0])}",
    ])


# --- spam: advertising that promises nothing already won and impersonates no one
@sp
def real_job_ad():
    b = R.choice(BRANDS_JOB + ["a local warehouse", "our client", "a national retailer", "Manpower", "Adecco", "Randstad"])
    return R.choice([
        f"Now hiring: {R.choice(['warehouse associates', 'drivers', 'cashiers', 'care assistants', 'call center agents', 'cleaners', 'forklift operators', 'night shift pickers'])} at {b}, {money()}/hr. Apply at {R.choice(['indeed.com', 'linkedin.com/jobs', 'glassdoor.com', 'our careers page', 'reed.co.uk', 'infojobs.net'])}. Reply STOP to opt out.",
        f"{b} is hiring {R.choice(['seasonal', 'part-time', 'full-time', 'weekend'])} staff. Open interviews {R.choice(DAYS)} {R.choice(TIMES)}-{R.choice(TIMES)}, bring ID. Text STOP to end.",
        f"Looking for a new job? Browse {R.randint(200, 9000)} {R.choice(['nursing', 'driving', 'retail', 'tech', 'hospitality'])} openings near you on {R.choice(['Indeed', 'Jooble', 'Monster', 'Totaljobs'])}. Unsubscribe: reply STOP",
        f"Job alert: {R.choice(JOB_TITLES)} roles from {money()}/hr. Set up alerts at {R.choice(['indeed.com', 'linkedin.com', 'ziprecruiter.com'])}. Opt out anytime.",
    ])


@sp
def giftcard_promo():
    g = R.choice(GIFTCARD)
    return R.choice([
        f"{g}: get a {money()} gift card when you spend {money()} {R.choice(DAYS)} only. In store and online. Terms apply. Reply STOP to opt out.",
        f"Buy {money()} of {R.choice(['groceries', 'fuel', 'electronics', 'toys', 'clothing'])} at {g} and get a {money()} bonus card. Ends {R.choice(DAYS)}.",
        f"{g} members: double points on everything this {R.choice(['weekend', 'week', 'Friday'])}. Show the app at checkout. STOP to opt out.",
        f"Enter our {R.choice(['spring', 'summer', 'holiday', 'back to school'])} giveaway for a chance to win a {big_money()} {g} gift card. Enter at {real_url(g)}/giveaway. No purchase necessary.",
        f"{g}: {R.randint(10, 60)}% off {R.choice(ITEMS)} this week. Shop at {real_url(g)}. Text STOP to unsubscribe.",
    ])


@sp
def generic_ads():
    return R.choice([
        f"{R.choice(['Personal loans', 'Car loans', 'Debt consolidation', 'Payday loans'])} from {R.choice(['2.9', '4.5', '5.9', '7.9'])}% APR, decision in minutes. Apply: {R.choice(['quickloans', 'cashnow', 'loanhub', 'creditfast'])}.{R.choice(['com', 'co.uk', 'net'])}",
        f"Learn {R.choice(['Python', 'data science', 'AI', 'digital marketing', 'nursing'])} in {R.randint(6, 16)} weeks and land a {big_money()} job. Enroll today at {R.choice(['codecamp', 'skillup', 'learnfast', 'bootcamp'])}.{R.choice(['edu', 'com', 'io'])}/apply",
        f"Extended car warranty: your coverage may be expiring. Protect your vehicle from repair bills, call {phone()} for a quote.",
        f"{R.choice(['Cheap Viagra', 'Generic Cialis', 'Weight-loss pills', 'CBD gummies'])} online, no prescription, discreet shipping. {R.choice(['pharma-direct', 'medsonline', 'rxcheap'])}.{R.choice(['ru', 'net', 'store'])}",
        f"Bitcoin is about to {R.choice(['explode', 'moon', 'double'])}! Join {R.randint(5, 90)}k members getting daily signals. Free guide: {R.choice(['btc-profits', 'cryptoedge', 'coinsignals'])}.{R.choice(['io', 'net', 'club'])}",
        f"Solar panels with $0 down, cut your bill by {R.randint(40, 80)}%. Free quote: {R.choice(['sunpower-deals', 'solarquote', 'gosolar'])}.com",
        f"Bet {money()} get {money()} in free bets at {R.choice(['BetKing', 'LuckySpin', 'WinZone', 'PlayMax'])}. 18+, T&Cs apply, new customers only.",
        f"Hot singles near {R.choice(CITIES)} want to chat! Free signup: {R.choice(['meetlocal', 'flirtnow', 'datehub'])}.{R.choice(['xyz', 'com', 'net'])}",
        f"Vote {name()} {R.choice(FIRST)} for {R.choice(['Council', 'Mayor', 'State Senate', 'School Board'])} on {R.choice(DAYS)}! {R.choice(['Lower taxes', 'Safer streets', 'Better schools'])}. Paid for by the campaign.",
        f"{R.choice(['Ramen Republic', 'Taco Loco', 'Burger Barn', 'Pizza Pronto', 'Sushi Go'])} is now open on {R.choice(['5th Ave', 'Main St', 'High Street', 'Market Square'])}! Show this text for a free {R.choice(['appetizer', 'drink', 'dessert'])} with any entree.",
        f"FLASH SALE {R.randint(30, 80)}% off at {R.choice(['FashionNova', 'Shein', 'Zara', 'H&M', 'ASOS', 'Boohoo'])} this {R.choice(['weekend', 'week', 'Friday'])} only. Shop now: {R.choice(['fnova.co', 'shein.com', 'asos.com', 'zara.com'])}/sale",
        f"Sell your house fast for cash, any condition, close in {R.randint(5, 14)} days. Call {phone()}",
        f"Unlock {R.randint(5, 50)}k followers overnight! Grow your Instagram fast: {R.choice(['instaboost', 'growfast', 'followerpro'])}.{R.choice(['pro', 'io', 'net'])}",
        f"Flights to {R.choice(CITIES)} from {money()}! Book by {R.choice(DAYS)} at {R.choice(['flyaway-travel', 'cheapflights', 'skydeals'])}.com",
        f"Refinance your mortgage at {R.choice(['2.9', '3.4', '4.1'])}% APR! Lock your rate today at {R.choice(['homeloanpro', 'ratelock', 'refi-now'])}.{R.choice(['us', 'com'])}",
    ])


@sp
def ads_i18n():
    return R.choice([
        f"¡Gana dinero desde casa! Hasta {money()} al mes sin experiencia. Regístrate: {R.choice(['trabajofacil', 'ingresosextra', 'ganaya'])}.es",
        f"Promo {R.choice(['Bouygues', 'Free', 'SFR', 'Orange'])} : forfait {R.choice(['50', '100', '150'])}Go à {R.choice(['7,99', '9,99', '12,99'])}€/mois pendant 12 mois ! Souscrivez sur {R.choice(['bouygues-promo', 'free-mobile-offre', 'sfr-promo'])}.fr",
        f"Jetzt {money()} Bonus sichern! Online Casino mit über {R.randint(500, 3000)} Spielen. {R.choice(['spielhalle24', 'casinoclub', 'jackpotzone'])}.de",
        f"عرض خاص! خصم {R.randint(20, 60)}% على جميع {R.choice(['العطور', 'الملابس', 'الأجهزة', 'الأحذية'])} في متجر {R.choice(['الياسمين', 'نون', 'الرياض مول'])} هذا الأسبوع فقط. تسوق الآن {R.choice(['yasmin-shop', 'noon', 'riyadhmall'])}.com",
        f"Ganhe até {money()} por dia com apostas online! Cadastre-se grátis: {R.choice(['apostaganha', 'betbrasil', 'jogafacil'])}.bet",
        f"{R.choice(['Flipkart', 'Myntra', 'Ajio', 'Meesho'])} Big Sale! Up to {R.randint(40, 80)}% off on {R.choice(['mobiles', 'fashion', 'electronics'])}. Shop now {R.choice(['flipkart.com', 'myntra.com', 'ajio.com'])}. Reply STOP to opt out.",
        f"Offre spéciale : {R.randint(30, 70)}% de réduction sur {R.choice(['les chaussures', 'la mode', 'les parfums'])} chez {R.choice(['Zalando', 'La Redoute', 'Sephora'])} jusqu'à {R.choice(['dimanche', 'vendredi', 'ce soir'])}. {R.choice(['zalando.fr', 'laredoute.fr', 'sephora.fr'])}",
    ])


# --- ham: the benign twins
@hm
def friend_favor():
    return join(R.choice(GREET), R.choice([
        f"can you send me the wifi password again, my phone forgot it",
        f"what's the code for the {R.choice(['garage', 'gate', 'front door', 'lockbox', 'gym locker'])}? I forgot it again",
        f"can you send me {name()}'s address? Need it for the {R.choice(['invite', 'card', 'delivery', 'taxi'])}",
        f"do you still have the {R.choice(['spare key', 'charger', 'blue jacket', 'book'])} I left at yours?",
        f"running {R.randint(5, 30)} min late, {R.choice(['traffic is terrible', 'train got delayed', 'meeting ran over', 'car wouldn\'t start'])}. Order me a {R.choice(['flat white', 'latte', 'beer', 'coke'])}?",
        f"are we still on for {R.choice(['dinner', 'lunch', 'the gym', 'football', 'drinks', 'the movie'])} {R.choice(DAYS)}? I can pick you up at {R.choice(TIMES)}",
        f"just finished the {R.choice(['report', 'deck', 'draft', 'invoice'])}, sending it over now. let me know if anything looks off",
        f"can you transfer me the {money()} for the {R.choice(['tickets', 'gift', 'dinner', 'airbnb'])} when you get a sec? sorted it with {name()} already",
        f"happy birthday!!! hope you have an amazing day, {R.choice(['drinks on me this weekend', 'see you saturday', 'call you later'])}",
        f"landed safely, will call you when I get to the hotel",
        f"{name()}'s {R.choice(['surgery', 'appointment', 'exam', 'interview'])} went well. Will update everyone tonight.",
        f"lol did you see what {name()} posted in the group chat",
        f"practice is cancelled {R.choice(DAYS)} because of the {R.choice(['rain', 'heat', 'storm'])}. Coach will send the new schedule tomorrow.",
        f"can you pick up {R.choice(['milk', 'bread', 'the kids', 'the dry cleaning', 'my parcel from the neighbour'])} on your way home?",
        f"what time does the {R.choice(['match', 'flight', 'show', 'ceremony'])} start {R.choice(DAYS)}?",
        f"mum said dinner is at {R.choice(TIMES)}, bring {name()} if they're free",
    ]), R.choice(CLOSERS))


@hm
def friend_new_number():
    n = name()
    return R.choice([
        f"{R.choice(GREET)} it's {n}, got a new number, save it! Old one stops working {R.choice(DAYS)}.",
        f"New number alert, this is {n} :) same WhatsApp, just a new sim. See you {R.choice(DAYS)}!",
        f"Hi all, {n} here. Switched carriers so this is my number from now on. No need to reply.",
        f"{R.choice(GREET)} {fam()}, it's {n}, this is my work phone number in case you can't reach the other one. Everything's fine!",
        f"hey it's {n} from {R.choice(['work', 'the office', 'yoga', 'the team', 'uni'])}, finally got a UK number, save me x",
        f"{n} here, new phone same number, lost all my contacts though. Who's this? 😅",
    ])


@hm
def friend_wrong_number():
    return R.choice([
        f"Hi, this is {name()} from {R.choice(['the dentist', 'Dr. ' + name() + '\'s office', 'the salon', 'the garage', 'the vet'])}. Is this still the best number for appointment reminders?",
        f"Sorry, wrong number!", f"Oops wrong number, sorry!", f"apologies, meant to send that to someone else",
        f"Hi, is this {name()}? It's {name()} from the {R.choice(['school run', 'book club', 'football team', 'PTA', 'landlord'])}, {name()} gave me your number about {R.choice(DAYS)}.",
        f"Hey {name()}, it's {name()}, we met at {name()}'s {R.choice(['wedding', 'birthday', 'leaving do'])} on {R.choice(DAYS)}. Great to meet you, let's do that coffee!",
        f"Hi {name()}, {name()} here from {R.choice(['the estate agent', 'the letting agency', 'the removal company'])}, confirming your viewing {R.choice(DAYS)} at {R.choice(TIMES)}.",
    ])


@hm
def real_order_notice():
    b = R.choice(BRANDS_SHOP)
    item = R.choice(ITEMS)
    return R.choice([
        f"{b}: your order {order_no()} ({item}) has shipped and will arrive {R.choice(DAYS)}. Track it at {real_url(b)}.",
        f"{b}: your package with {item} was delivered. It was {R.choice(['handed to a resident', 'left at the front door', 'left in the mailroom', 'left with the concierge'])}.",
        f"Thanks for your {b} order! {item}, {money()}. Est. delivery {R.choice(DAYS)}. Manage your order in {real_url(b)}.",
        f"{b}: {item} is out for delivery today. Someone should be available to sign.",
        f"Your {b} return for {item} was received. Refund of {money()} will appear on your card in 3-5 business days.",
        f"{b}: order {order_no()} is ready for {R.choice(['pickup', 'collection'])} at the {R.choice(CITIES)} store. Bring your order confirmation.",
    ])


@hm
def real_subscription_notice():
    b = R.choice(BRANDS_SUB)
    return R.choice([
        f"{b}: your payment of {money()} was processed. Next billing date {R.choice(DAYS)}. Manage your plan in Account settings.",
        f"{b}: a new device signed in to your account from {R.choice(CITIES)}. If this wasn't you, change your password in the {b} app.",
        f"Your {b} free trial ends {R.choice(DAYS)}. You'll be charged {money()}/month unless you cancel in the app.",
        f"{b}: we couldn't process your payment. Please update your payment method in the {b} app to keep your plan.",
        f"{b} receipt: {money()} for your {R.choice(['monthly', 'annual', 'family', 'student'])} plan. Thank you!",
    ])


@hm
def real_bank_notice():
    b = R.choice(BANKS)
    return R.choice([
        f"{b}: Did you make a {money()} purchase at {R.choice(['TARGET', 'BEST BUY', 'AMAZON', 'SHELL', 'WALMART', 'UBER', 'TESCO', 'IKEA'])} on {R.randint(1, 12)}/{R.randint(1, 28)}? Reply YES or NO.",
        f"{b}: your account ending {card_last4()} has a low balance of {money()}. This is the alert you set up.",
        f"{b}: a payment of {money()} to {name()} {R.choice(FIRST)} was sent from your account. Ref {order_no()}.",
        f"{b}: your statement is ready. View it in the {b} app or online banking.",
        f"{b}: {money()} was deposited into your account ending {card_last4()}. Available balance {money()}.",
        f"{b} will never ask for your PIN, password or full card number by text or phone. If in doubt, call the number on the back of your card.",
        f"{b}: your new card ending {card_last4()} has been sent and should arrive in 5-7 days. Activate it in the app when it arrives.",
        f"{b}: your card was used abroad in {R.choice(CITIES)} for {money()}. If you don't recognise this, call the number on your card.",
        f"{b}: your direct debit of {money()} to {R.choice(TELCOS + UTIL)} is due {R.choice(DAYS)}.",
    ])


@hm
def real_telco_util_notice():
    t = R.choice(TELCOS)
    u = R.choice(UTIL)
    return R.choice([
        f"{t}: you've used {R.choice(['80', '90', '100'])}% of your {R.choice(['10', '20', '50'])}GB monthly data. Your allowance resets on {R.randint(1, 28)} {R.choice(['Oct', 'Nov', 'Jan', 'Mar', 'Jun'])}.",
        f"{t}: your bill of {money()} is now available. View or pay at {real_url(t)}. Autopay will process on {R.randint(1, 28)}/{R.randint(1, 12)}.",
        f"{t}: your recharge of {money()} was successful. Validity {R.randint(28, 84)} days.",
        f"{t}: planned maintenance in your area {R.choice(DAYS)} {R.choice(TIMES)}-{R.choice(TIMES)}. You may lose signal briefly. Sorry for the inconvenience.",
        f"Your {u} bill of {money()} is due on {R.choice(DAYS)}. Pay in the {u} app or at {real_url(u)}. Thank you.",
        f"{u}: a power cut is affecting your area. Engineers are on site, estimated restoration {R.choice(TIMES)}.",
        f"{t}: thanks for your payment of {money()}. Your next bill is due {R.choice(DAYS)}.",
        f"{u}: your meter reading is due. Submit it in the app by {R.choice(DAYS)} to avoid an estimated bill.",
    ])


@hm
def real_delivery_notice():
    c = R.choice(COURIERS)
    return R.choice([
        f"{c}: your parcel {order_no()} will be delivered {R.choice(DAYS)} between {R.choice(TIMES)} and {R.choice(TIMES)}. No action needed.",
        f"{c}: we delivered your parcel to {R.choice(['your front porch', 'a neighbour at No. ' + str(R.randint(1, 99)), 'your safe place', 'the reception desk'])}.",
        f"{c}: sorry we missed you. Your parcel is at the {R.choice(CITIES)} depot, collect it with photo ID or rebook in the {c} app.",
        f"{c}: your driver {name()} is {R.randint(2, 9)} stops away.",
        f"{c}: parcel {order_no()} has been collected from the sender and is on its way.",
    ])


@hm
def real_hr_and_appointments():
    return R.choice([
        f"Thanks for coming to the interview today. We'll be in touch by {R.choice(DAYS)} with next steps. - {name()}, HR",
        f"Hi {name()}, this is {name()} from {R.choice(BRANDS_JOB)} recruiting. Your interview is confirmed for {R.choice(DAYS)} at {R.choice(TIMES)}. Reply if you need to reschedule.",
        f"Reminder: your appointment with Dr. {name()} is {R.choice(DAYS)} at {R.choice(TIMES)}. Reply C to confirm or R to reschedule.",
        f"Your prescription is ready for pickup at {R.choice(['CVS', 'Walgreens', 'Boots', 'Lloyds Pharmacy'])}, {R.randint(100, 9999)} Main St. Open {R.choice(TIMES)}-{R.choice(TIMES)}.",
        f"Your table for {R.randint(2, 8)} at {R.choice(['Nobu', 'Dishoom', 'Carbone', 'The Ivy', 'Osteria'])} is confirmed for {R.choice(DAYS)} {R.choice(TIMES)}. Reply CANCEL to cancel.",
        f"{R.choice(['Delta', 'United', 'British Airways', 'Ryanair', 'Emirates', 'Lufthansa'])}: flight {R.choice('ABDELU')}{R.choice('AEHKRU')}{R.randint(100, 9999)} to {R.choice(CITIES)} now departs from gate {R.choice('ABCD')}{R.randint(1, 40)}. Boarding at {R.choice(TIMES)}.",
        f"Your {R.choice(['Lyft', 'Uber', 'Bolt', 'Cabify'])} driver {name()} is arriving in a {R.choice(['gray Toyota Camry', 'black Prius', 'white Tesla', 'silver Skoda'])}, plate {R.choice('ABCDEFGH')}{R.randint(10, 99)}{R.choice('KLMNP')}{R.randint(100, 999)}.",
        f"Your Airbnb host {name()} sent you a message: check-in is after {R.choice(TIMES)}, the lockbox code is in the app.",
        f"Weather alert: {R.choice(['flash flood', 'severe thunderstorm', 'high wind', 'heat'])} warning for {R.choice(CITIES)} until {R.choice(TIMES)}. Avoid low-lying roads.",
        f"School: {R.choice(['early pickup', 'no school', 'sports day', 'parents evening'])} on {R.choice(DAYS)}. See the newsletter for details.",
    ])


@hm
def real_codes():
    b = R.choice(["Uber", "Google", "WhatsApp", "Apple", "Microsoft", "Amazon", "PayPal", "Venmo", "Doctolib", "Revolut", "Monzo", "Instagram", "Telegram", "Signal", "Steam", "Discord"] + BANKS)
    return R.choice([
        f"Your {b} code is {code()}. Never share this code with anyone.",
        f"{code()} is your {b} verification code. It expires in {R.choice([5, 10, 15])} minutes.",
        f"{b}: {code()} is your one-time passcode. Do not share it. We will never call to ask for it.",
        f"Use {code()} to sign in to {b}. If you didn't request this, ignore this message.",
        f"Votre code de connexion {b} est {code()}. Il expire dans 10 minutes.",
        f"Tu código de verificación de {b} es {code()}. No lo compartas.",
        f"Dein {b} Bestätigungscode lautet {code()}.",
        f"Seu código {b} é {code()}. Não compartilhe com ninguém.",
        f"{b}: رمز التحقق الخاص بك هو {code()}. لا تشاركه مع أحد.",
        f"{code()} is your {b} OTP. Valid for 10 mins. Do not share with anyone. -{b}",
    ])


@hm
def chat_i18n():
    return R.choice([
        f"Hola, ¿a qué hora llegas {R.choice(['mañana', 'hoy', 'el sábado'])}? Te espero en {R.choice(['la estación', 'casa', 'el bar de siempre'])}.",
        f"¿Has visto mis llaves? Creo que las dejé en tu coche",
        f"Je suis en retard de {R.randint(5, 30)} min, commande-moi un café stp",
        f"On se voit {R.choice(['ce soir', 'demain', 'samedi'])} ? {name()} vient aussi",
        f"Bin in {R.randint(5, 30)} Minuten da, die Bahn hatte Verspätung",
        f"Kannst du Brot mitbringen? Wir haben keins mehr",
        f"Oi, chegou o boleto do condomínio? Preciso pagar até {R.choice(['sexta', 'amanhã', 'dia 10'])}.",
        f"Tô chegando, {R.randint(5, 20)} min. Pede uma água pra mim?",
        f"سأتأخر قليلاً عن الاجتماع، ابدؤوا بدوني وسألحق بكم",
        f"وين انتو؟ أنا وصلت المطعم",
        f"आज शाम को घर आ रहे हो? मम्मी ने खाना बनाया है",
        f"Kal {R.choice(TIMES)} baje milte hain, station pe aa jana",
        f"Ciao, ci vediamo alle {R.randint(7, 21)} davanti al cinema?",
        f"Hoi, ben je nog thuis? Ik kom je pakje brengen",
    ])


BUILDERS = [(f, "phishing") for f in PH] + [(f, "spam") for f in SP] + [(f, "ham") for f in HM]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=250, help="rows per builder before dedup")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    R.seed(args.seed)
    seen, rows = set(), []
    for fn, label in BUILDERS:
        made = 0
        for _ in range(args.n * 3):
            if made >= args.n:
                break
            t = fn()
            if t and t not in seen:
                seen.add(t)
                rows.append({"text": t, "label": label, "category": fn.__name__})
                made += 1
    random.Random(args.seed).shuffle(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["text", "label", "category"])
        w.writeheader(); w.writerows(rows)
    from collections import Counter
    print(f"{len(rows)} rows -> {out}")
    print(Counter(r["label"] for r in rows)); print(Counter(r["category"] for r in rows))


if __name__ == "__main__":
    main()
