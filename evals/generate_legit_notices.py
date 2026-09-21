#!/usr/bin/env python3
"""Legitimate branded notices (bank, telco, courier, retailer, government, health)
with real domains, amounts and routine calls to action, plus lookalike phishing twins.

The distillation set built from the corpus has 844 ham rows with a link against
33,554 phishing rows with a link, and 4 ham rows with brand + link + amount
against 1,420 phishing rows. Model 2.9-s7 learned exactly that prior: every
"Verizon: your bill of $118 is available at myvzw.com" is blocked. This generator
composes legitimate notices from independent pools (brand, opener, event, amount,
action on the official domain, closer) so no two rows share a template, in 12
languages, and emits a minimal-pair phishing twin for a share of them: the same
notice with a lookalike domain and a threat or credential ask. Rows are intended
labels; evals/distill_labels.py asks TypeSafe about each one and keeps the row
only when the teacher agrees.

    python evals/generate_legit_notices.py --n 12000 --seed 23 \
        --out src/mBERT/training/model-training/dataset/curated/synthetic_legit_notices_v1.csv
"""
import argparse
import csv
import random
from pathlib import Path

R = random.Random()

# ----------------------------------------------------------------- brands + real domains
BANKS = {"Chase": "chase.com", "Wells Fargo": "wellsfargo.com", "Bank of America": "bankofamerica.com", "Citi": "citi.com",
         "Capital One": "capitalone.com", "TD Bank": "td.com", "US Bank": "usbank.com", "PNC": "pnc.com", "Truist": "truist.com",
         "Ally": "ally.com", "Discover": "discover.com", "American Express": "americanexpress.com", "Barclays": "barclays.co.uk",
         "HSBC": "hsbc.co.uk", "Lloyds": "lloydsbank.com", "NatWest": "natwest.com", "Halifax": "halifax.co.uk", "Monzo": "monzo.com",
         "Starling": "starlingbank.com", "Revolut": "revolut.com", "Santander": "santander.com", "BBVA": "bbva.es",
         "CaixaBank": "caixabank.es", "BNP Paribas": "bnpparibas.com", "Crédit Agricole": "credit-agricole.fr",
         "Société Générale": "societegenerale.fr", "Sparkasse": "sparkasse.de", "Commerzbank": "commerzbank.de",
         "Deutsche Bank": "deutsche-bank.de", "ING": "ing.com", "N26": "n26.com", "Rabobank": "rabobank.nl", "UniCredit": "unicredit.it",
         "Intesa Sanpaolo": "intesasanpaolo.com", "Itaú": "itau.com.br", "Nubank": "nubank.com.br", "Bradesco": "bradesco.com.br",
         "SBI": "onlinesbi.sbi", "HDFC Bank": "hdfcbank.com", "ICICI Bank": "icicibank.com", "Axis Bank": "axisbank.com",
         "Al Rajhi": "alrajhibank.com.sa", "Emirates NBD": "emiratesnbd.com", "QNB": "qnb.com", "DBS": "dbs.com.sg",
         "Maybank": "maybank2u.com.my", "BCA": "bca.co.id", "Scotiabank": "scotiabank.com", "RBC": "rbc.com", "TD Canada": "td.com",
         "Commonwealth Bank": "commbank.com.au", "ANZ": "anz.com.au", "Westpac": "westpac.com.au", "Sberbank": "sberbank.ru",
         "Tinkoff": "tinkoff.ru", "Garanti BBVA": "garantibbva.com.tr", "Ziraat": "ziraatbank.com.tr", "Alpha Bank": "alpha.gr",
         "Piraeus Bank": "piraeusbank.gr", "PKO BP": "pkobp.pl", "mBank": "mbank.pl", "Bank Mandiri": "bankmandiri.co.id"}
P2P = {"Venmo": "venmo.com", "PayPal": "paypal.com", "Zelle": "zellepay.com", "Cash App": "cash.app", "Revolut": "revolut.com",
       "Monzo": "monzo.com", "Wise": "wise.com", "Bizum": "bizum.es", "PIX": "bcb.gov.br", "Paytm": "paytm.com", "PhonePe": "phonepe.com",
       "Google Pay": "pay.google.com", "Apple Pay": "apple.com", "M-Pesa": "safaricom.co.ke", "GCash": "gcash.com", "Lydia": "lydia-app.com",
       "Tikkie": "tikkie.me", "MobilePay": "mobilepay.dk", "Swish": "swish.nu", "Satispay": "satispay.com"}
TELCOS = {"Verizon": "myvzw.com", "AT&T": "att.com", "T-Mobile": "t-mobile.com", "Vodafone": "vodafone.co.uk", "EE": "ee.co.uk",
          "O2": "o2.co.uk", "Three": "three.co.uk", "Orange": "orange.fr", "SFR": "sfr.fr", "Bouygues": "bouyguestelecom.fr",
          "Telekom": "telekom.de", "O2 Germany": "o2online.de", "Movistar": "movistar.es", "Telcel": "telcel.com", "Claro": "claro.com.br",
          "Vivo": "vivo.com.br", "TIM": "tim.it", "Airtel": "airtel.in", "Jio": "jio.com", "Vi": "myvi.in", "STC": "stc.com.sa",
          "Etisalat": "etisalat.ae", "Zain": "zain.com", "Telstra": "telstra.com.au", "Optus": "optus.com.au", "Rogers": "rogers.com",
          "Bell": "bell.ca", "Telus": "telus.com", "KPN": "kpn.com", "Turkcell": "turkcell.com.tr", "Telkomsel": "telkomsel.com",
          "MTS": "mts.ru", "Beeline": "beeline.ru", "Cosmote": "cosmote.gr", "Play": "play.pl"}
UTIL = {"ConEd": "coned.com", "PG&E": "pge.com", "Duke Energy": "duke-energy.com", "British Gas": "britishgas.co.uk", "EDF": "edfenergy.com",
        "Octopus Energy": "octopus.energy", "Enel": "enel.it", "Iberdrola": "iberdrola.es", "Endesa": "endesa.com", "E.ON": "eon.de",
        "Vattenfall": "vattenfall.de", "Engie": "engie.fr", "Naturgy": "naturgy.es", "Thames Water": "thameswater.co.uk",
        "National Grid": "nationalgrid.com", "Xcel Energy": "xcelenergy.com", "ΔΕΗ": "dei.gr", "Enedis": "enedis.fr", "Eneco": "eneco.nl",
        "Tata Power": "tatapower.com", "BSES": "bsesdelhi.com", "DEWA": "dewa.gov.ae", "PLN": "pln.co.id", "Mosenergosbyt": "mosenergosbyt.ru"}
COURIERS = {"USPS": "usps.com", "FedEx": "fedex.com", "UPS": "ups.com", "DHL": "dhl.com", "Royal Mail": "royalmail.com", "Evri": "evri.com",
            "DPD": "dpd.co.uk", "Correos": "correos.es", "La Poste": "laposte.fr", "Chronopost": "chronopost.fr", "Colissimo": "laposte.fr",
            "Correios": "correios.com.br", "Aramex": "aramex.com", "Canada Post": "canadapost-postescanada.ca",
            "Australia Post": "auspost.com.au", "PostNL": "postnl.nl", "Bpost": "bpost.be", "GLS": "gls-group.com", "Hermes": "myhermes.de",
            "Delhivery": "delhivery.com", "Blue Dart": "bluedart.com", "Yurtiçi Kargo": "yurticikargo.com", "JNE": "jne.co.id",
            "SDA": "sda.it", "BRT": "brt.it", "Amazon Logistics": "amazon.com", "InPost": "inpost.pl", "Poste Italiane": "poste.it",
            "СДЭК": "cdek.ru", "Почта России": "pochta.ru"}
SHOPS = {"Amazon": "amazon.com", "Walmart": "walmart.com", "Target": "target.com", "Best Buy": "bestbuy.com", "eBay": "ebay.com",
         "Apple": "apple.com", "Costco": "costco.com", "Home Depot": "homedepot.com", "Argos": "argos.co.uk", "Currys": "currys.co.uk",
         "John Lewis": "johnlewis.com", "Tesco": "tesco.com", "Sainsbury's": "sainsburys.co.uk", "ASOS": "asos.com", "Zara": "zara.com",
         "IKEA": "ikea.com", "MediaMarkt": "mediamarkt.de", "Otto": "otto.de", "Zalando": "zalando.de", "Fnac": "fnac.com",
         "Cdiscount": "cdiscount.com", "El Corte Inglés": "elcorteingles.es", "Mercado Livre": "mercadolivre.com.br",
         "Magazine Luiza": "magazineluiza.com.br", "Flipkart": "flipkart.com", "Myntra": "myntra.com", "Noon": "noon.com",
         "Jumia": "jumia.com", "Bol.com": "bol.com", "Coolblue": "coolblue.nl", "Trendyol": "trendyol.com", "Hepsiburada": "hepsiburada.com",
         "Tokopedia": "tokopedia.com", "Shopee": "shopee.com", "Lazada": "lazada.com", "Ozon": "ozon.ru", "Wildberries": "wildberries.ru",
         "Allegro": "allegro.pl", "Skroutz": "skroutz.gr", "Etsy": "etsy.com", "Wayfair": "wayfair.com", "Nike": "nike.com", "Uniqlo": "uniqlo.com"}
SUBS = {"Netflix": "netflix.com", "Spotify": "spotify.com", "Disney+": "disneyplus.com", "Hulu": "hulu.com", "Amazon Prime": "amazon.com",
        "Apple Music": "apple.com", "YouTube Premium": "youtube.com", "HBO Max": "max.com", "Xbox Game Pass": "xbox.com",
        "PlayStation Plus": "playstation.com", "Canva": "canva.com", "Dropbox": "dropbox.com", "iCloud": "icloud.com",
        "Google One": "one.google.com", "Adobe": "adobe.com", "Microsoft 365": "microsoft.com", "Audible": "audible.com",
        "Duolingo": "duolingo.com", "Strava": "strava.com", "Peloton": "onepeloton.com", "NYTimes": "nytimes.com", "Deezer": "deezer.com",
        "DAZN": "dazn.com", "Crunchyroll": "crunchyroll.com", "Globoplay": "globoplay.globo.com", "Hotstar": "hotstar.com"}
TECH = {"Google": "google.com", "Apple": "apple.com", "Microsoft": "microsoft.com", "Amazon": "amazon.com", "Meta": "facebook.com",
        "Instagram": "instagram.com", "WhatsApp": "whatsapp.com", "Telegram": "telegram.org", "Discord": "discord.com", "Steam": "steampowered.com",
        "Uber": "uber.com", "Lyft": "lyft.com", "Airbnb": "airbnb.com", "Booking.com": "booking.com", "LinkedIn": "linkedin.com",
        "GitHub": "github.com", "Slack": "slack.com", "Zoom": "zoom.us", "Dropbox": "dropbox.com", "Coinbase": "coinbase.com",
        "Binance": "binance.com", "Robinhood": "robinhood.com", "Twitter": "x.com", "TikTok": "tiktok.com", "Snapchat": "snapchat.com",
        "Yandex": "yandex.ru", "VK": "vk.com", "Mercado Pago": "mercadopago.com", "Grab": "grab.com", "Gojek": "gojek.com"}
GOV = {"IRS": "irs.gov", "HMRC": "gov.uk/hmrc", "Social Security": "ssa.gov", "Medicare": "medicare.gov", "DMV": "dmv.ca.gov", "DVLA": "gov.uk/dvla",
       "NHS": "nhs.uk", "CRA": "canada.ca/cra", "ATO": "ato.gov.au", "Services Australia": "servicesaustralia.gov.au", "Centrelink": "servicesaustralia.gov.au",
       "CAF": "caf.fr", "URSSAF": "urssaf.fr", "impots.gouv": "impots.gouv.fr", "Ameli": "ameli.fr", "Finanzamt": "elster.de",
       "Bundesagentur für Arbeit": "arbeitsagentur.de", "Agencia Tributaria": "agenciatributaria.gob.es", "Seguridad Social": "seg-social.es",
       "DGT": "dgt.es", "Receita Federal": "gov.br/receitafederal", "INSS": "gov.br/inss", "Detran": "detran.sp.gov.br", "Госуслуги": "gosuslugi.ru",
       "ФНС": "nalog.gov.ru", "e-Devlet": "turkiye.gov.tr", "SGK": "sgk.gov.tr", "gov.gr": "gov.gr", "ΑΑΔΕ": "aade.gr", "UIDAI": "uidai.gov.in",
       "Income Tax Dept": "incometax.gov.in", "EPFO": "epfindia.gov.in", "Dubai Police": "dubaipolice.gov.ae", "ICP": "icp.gov.ae",
       "Absher": "absher.sa", "DigiD": "digid.nl", "Belastingdienst": "belastingdienst.nl", "ePUAP": "epuap.gov.pl", "ZUS": "zus.pl",
       "BPJS": "bpjs-kesehatan.go.id", "Dukcapil": "dukcapil.kemendagri.go.id", "SPID": "spid.gov.it", "INPS": "inps.it", "Agenzia Entrate": "agenziaentrate.gov.it"}
HEALTH = {"Kaiser Permanente": "kp.org", "CVS Pharmacy": "cvs.com", "Walgreens": "walgreens.com", "Boots": "boots.com", "Lloyds Pharmacy": "lloydspharmacy.com",
          "MyChart": "mychart.com", "Quest Diagnostics": "questdiagnostics.com", "LabCorp": "labcorp.com", "Doctolib": "doctolib.fr", "Zocdoc": "zocdoc.com",
          "Practo": "practo.com", "Apollo Pharmacy": "apollopharmacy.in", "1mg": "1mg.com", "Medicover": "medicover.pl", "Sanitas": "sanitas.es",
          "Quirónsalud": "quironsalud.es", "Bupa": "bupa.co.uk", "Doctoralia": "doctoralia.com", "Jameda": "jameda.de", "Halodoc": "halodoc.com"}
INSURE = {"Geico": "geico.com", "State Farm": "statefarm.com", "Progressive": "progressive.com", "Allstate": "allstate.com", "Aviva": "aviva.co.uk",
          "AXA": "axa.com", "Allianz": "allianz.com", "Generali": "generali.com", "Mapfre": "mapfre.com", "LIC": "licindia.in", "Bajaj Finserv": "bajajfinserv.in",
          "Porto Seguro": "portoseguro.com.br", "Zurich": "zurich.com", "Lemonade": "lemonade.com", "Direct Line": "directline.com"}
TRAVEL = {"Delta": "delta.com", "United": "united.com", "American Airlines": "aa.com", "Southwest": "southwest.com", "British Airways": "ba.com",
          "Ryanair": "ryanair.com", "easyJet": "easyjet.com", "Lufthansa": "lufthansa.com", "Air France": "airfrance.com", "KLM": "klm.com",
          "Emirates": "emirates.com", "Qatar Airways": "qatarairways.com", "IndiGo": "goindigo.in", "LATAM": "latam.com", "Turkish Airlines": "turkishairlines.com",
          "Aeroflot": "aeroflot.ru", "Garuda": "garuda-indonesia.com", "Amtrak": "amtrak.com", "Eurostar": "eurostar.com", "SNCF": "sncf-connect.com",
          "Deutsche Bahn": "bahn.de", "Renfe": "renfe.com", "Trenitalia": "trenitalia.com", "Marriott": "marriott.com", "Hilton": "hilton.com",
          "IHG": "ihg.com", "Airbnb": "airbnb.com", "Booking.com": "booking.com", "Expedia": "expedia.com", "Uber": "uber.com", "Lyft": "lyft.com",
          "Bolt": "bolt.eu", "Cabify": "cabify.com", "Ola": "olacabs.com", "Careem": "careem.com", "99": "99app.com", "Yandex Go": "go.yandex"}

MERCHANTS = ["TARGET", "BEST BUY", "AMAZON.COM", "SHELL OIL", "WALMART", "UBER", "TESCO", "IKEA", "COSTCO", "APPLE.COM/BILL", "STARBUCKS",
             "NETFLIX", "LIDL", "ALDI", "CARREFOUR", "MERCADONA", "REWE", "EDEKA", "7-ELEVEN", "MCDONALDS", "DELTA AIR", "MARRIOTT", "ZARA",
             "H&M", "DECATHLON", "SPOTIFY", "GOOGLE *YOUTUBE", "PAYPAL *EBAY", "SQ *COFFEE HOUSE", "AMZN MKTP", "WHOLEFDS", "TRADER JOE'S", "CHEVRON",
             "BP", "EXXON", "ESSO", "PETROBRAS", "INDIAN OIL", "SWIGGY", "ZOMATO", "DMART", "BIG BAZAAR", "JUMBO", "ALBERT HEIJN", "MIGROS", "BIM"]
ITEMS = ["Bluetooth headphones", "a coffee maker", "running shoes", "a phone case", "a desk lamp", "2 books", "a backpack", "printer ink",
         "a monitor", "a kettle", "dog food", "a yoga mat", "3 items", "1 item", "a jacket", "an air fryer", "socks", "a laptop stand",
         "a birthday gift", "groceries", "a bike helmet", "sunglasses", "a tent", "toner cartridges", "a keyboard", "a water bottle"]
CITIES = ["Chicago", "Dallas", "Miami", "Seattle", "Denver", "Boston", "London", "Manchester", "Leeds", "Madrid", "Barcelona", "Paris", "Lyon",
          "Berlin", "Munich", "Hamburg", "Lisbon", "Porto", "Milan", "Rome", "Amsterdam", "Dubai", "Riyadh", "Mumbai", "Delhi", "Bangalore",
          "Toronto", "Vancouver", "Sydney", "Melbourne", "Lagos", "Nairobi", "Moscow", "Istanbul", "Athens", "Warsaw", "Jakarta", "Manila", "São Paulo",
          "Rio de Janeiro", "Mexico City", "Buenos Aires", "Singapore", "Kuala Lumpur"]
FIRST = ["Sarah", "Mike", "Emma", "Tom", "Liam", "Olivia", "Noah", "Ava", "Jake", "Lucy", "Ben", "Mia", "Chris", "Zoe", "Carlos", "Lucía", "Mateo",
         "Sofía", "Pierre", "Camille", "Louis", "Léa", "Jonas", "Lena", "Max", "Anna", "João", "Ana", "Pedro", "Beatriz", "Ahmed", "Fatima", "Omar",
         "Layla", "Rahul", "Priya", "Arjun", "Neha", "Yuki", "Kenji", "Wei", "Mei", "Ivan", "Olga", "Dmitri", "Natasha", "Alex Kim", "J. Patel",
         "M. Rossi", "S. Müller", "R. Silva", "K. Nguyen", "D. Okafor", "L. Dubois"]
DAYS = ["today", "tomorrow", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday", "Mon 22 Sep", "Tue 23 Sep", "Fri 26 Sep",
        "Sep 25", "Sep 28", "Sep 30", "Oct 1", "Oct 3", "Oct 10", "1 Oct", "3 Oct", "14 Oct", "10/02", "10/15", "09/30", "the 28th", "the 5th"]
TIMES = ["8am", "9am", "9:30am", "10am", "10:30am", "11:15am", "noon", "1pm", "2pm", "2:45pm", "3pm", "4pm", "5pm", "6pm", "7:30pm", "8pm",
         "08:00", "09:00", "10:30", "12:00", "13:00", "14:00", "15:30", "17:00", "18:00", "20:00"]
DOCTORS = ["Dr. Patel", "Dr. Nguyen", "Dr. Okafor", "Dr. Rossi", "Dr. Müller", "Dr. Silva", "Dr. Kim", "Dr. Ahmed", "Dr. García", "Dr. Dubois",
           "Dr. Ivanova", "Dr. Yilmaz", "Dr. Papadopoulos", "Dr. Kowalski", "Dr. Sharma", "Dr. Santos"]
STOP = ["Reply STOP to opt out.", "Msg&data rates may apply.", "Reply STOP to unsubscribe.", "Text STOP to end.", "Do not reply to this message.",
        "This is an automated message.", "Questions? Call the number on the back of your card.", "Thank you for banking with us.", "Thank you.",
        "Ref {ref}.", "Msg ID {ref}.", "Need help? Visit {url}.", "Manage alerts in the app.", "", "", "", "", ""]
BAD_TLDS = [".info", ".xyz", ".top", ".club", ".site", ".online", ".icu", ".live", ".click", ".link", ".buzz", ".cc", ".co", ".ru", ".cn", ".vip",
            ".pw", ".ws", ".tk", ".ml", ".gq", ".cf", ".net", ".org", ".shop", ".help", ".support", ".center", ".digital"]
SHORT = ["bit.ly", "tinyurl.com", "t.co", "is.gd", "cutt.ly", "rb.gy", "shorturl.at", "u.to", "v.gd", "ow.ly", "s.id", "goo.su"]

# ----------------------------------------------------------------- helpers
def pick(seq):
    return R.choice(seq)


def money(lang="en"):
    small = R.choice(["1.99", "2.45", "3.50", "4.20", "9.99", "12.40", "14.99", "18.50", "23.15", "29.00", "34.60", "45.00", "49.99", "58.20", "64.20",
                      "72.18", "85.12", "89.99", "92.41", "118.60", "129.90", "142.00", "184.20", "199.00", "230.00", "249.00", "312.45", "420.00",
                      "612.40", "780.00", "1,200.00", "1,480.50", "2,150.00", "2,399.00", "3,000.00"])
    cur = {"en": pick(["$", "£", "€", "C$", "A$", "$"]), "es": "€", "fr": "€", "de": "€", "it": "€", "pt": "R$", "nl": "€", "ru": "₽", "tr": "TL",
           "id": "Rp", "ar": pick(["AED", "SAR", "ر.س"]), "hi": "Rs.", "el": "€", "pl": "zł"}[lang]
    if cur in ("€", "TL", "zł", "₽"):
        s = small.replace(",", "").replace(".", ",")
        return f"{s} {cur}" if cur != "€" or R.random() < 0.5 else f"{s}€"
    if cur == "Rp":
        return "Rp" + f"{int(float(small.replace(',', '')) * 15000):,}".replace(",", ".")
    if cur == "Rs.":
        return f"Rs.{small}" if R.random() < 0.5 else f"INR {small}"
    if cur in ("AED", "SAR", "ر.س"):
        return f"{cur} {small}" if cur != "ر.س" else f"{small} ر.س"
    return f"{cur}{small}"


def last4():
    return f"{R.randint(1000, 9999)}"


def masked():
    return pick([f"ending {last4()}", f"ending in {last4()}", f"*{last4()}", f"x{last4()}", f"XX{last4()}", f"••{last4()}", f"ending with {last4()}"])


def ref():
    return pick([f"{R.randint(100000, 999999)}", f"{pick('ABCDEFGHJK')}{R.randint(10000, 99999)}", f"{R.randint(1000, 9999)}-{R.randint(10000, 99999)}",
                 f"#{R.randint(100, 999)}-{R.randint(1000000, 9999999)}-{R.randint(1000000, 9999999)}", f"{R.randint(10000000, 99999999)}",
                 f"{R.randint(1000000000, 9999999999)}", f"{pick('CDEJKRUZ')}{pick('ABDEHKMNPRSTVWXYZ')}{R.randint(100000000, 999999999)}"])


def code():
    return pick([f"{R.randint(100000, 999999)}", f"{R.randint(100, 999)} {R.randint(100, 999)}", f"{R.randint(1000, 9999)}", f"{R.randint(100, 999)}-{R.randint(100, 999)}",
                 f"{pick('ABCDEFGHJKLMNPQRSTUVWXYZ')}{pick('ABCDEFGHJKLMNPQRSTUVWXYZ')}{R.randint(1000, 9999)}", f"{R.randint(10000000, 99999999)}"])


def date():
    return pick(DAYS)


def time_():
    return pick(TIMES)


def good_url(brand, dom):
    path = pick(["", "", "", "/orders", "/account", "/track", "/pay", "/bill", "/security", "/login", "/statements", "/appointments", "/alerts",
                 "/myaccount", "/app", "/help", "/returns", "/manage", "/checkin", "/portal"])
    variants = [dom + path, f"https://{dom}{path}", f"https://www.{dom}{path}", f"www.{dom}{path}", f"the {brand} app", f"the {brand} app or {dom}",
                f"{dom} or the {brand} app", f"the app", f"online banking", f"your online account", f"the {brand} website", f"our app", dom, dom, dom]
    return pick(variants)


def bad_url(brand):
    slug = brand.lower().replace(" ", "").replace("'", "").replace("&", "").replace(".", "").replace("+", "plus")[:12]
    host = pick([f"{slug}-{pick(['verify', 'secure', 'update', 'claim', 'pay', 'help', 'support', 'alert', 'notice', 'refund', 'login', 'billing', 'auth'])}",
                 f"{pick(['my', 'get', 'go', 'e', 'app', 'online', 'secure', 'account'])}-{slug}", f"{slug}{R.randint(1, 99)}", f"{slug}-{pick(CITIES).lower().replace(' ', '')}",
                 f"{slug}.{pick(['secure', 'verify', 'account', 'login', 'billing'])}-{pick(['portal', 'center', 'service', 'online'])}",
                 f"{slug.replace('o', '0', 1) if 'o' in slug else slug + 'x'}", f"{slug}-{pick(['us', 'uk', 'eu', 'in', 'au'])}"])
    url = host + pick(BAD_TLDS)
    if R.random() < 0.3:
        url = f"https://{pick(SHORT)}/{pick('abcdefghjkmnpqrstuvwxyz')}{R.randint(1000, 99999)}"
    elif R.random() < 0.4:
        url = pick(["http://", "https://"]) + url
    if R.random() < 0.5:
        url += "/" + pick(["claim", "verify", "pay", "update", "login", "confirm", "id", "secure", "refund", "unlock", "reactivate"])
    return url


def opener(brand):
    return pick([f"{brand}:", f"{brand}:", f"{brand}:", f"{brand} -", f"[{brand}]", f"{brand} Alert:", f"{brand} Notice:", f"From {brand}:",
                 f"{brand.upper()}:", f"{brand}", f"({brand})", f"{brand} –", f"{brand}|", f"*{brand}*"])


def closer():
    c = pick(STOP)
    return c.replace("{ref}", ref()).replace("{url}", "the app")


def join(*parts):
    return " ".join(p for p in parts if p).replace("  ", " ").strip()


def with_brand(brand, body):
    body = body[0].upper() + body[1:] if R.random() < 0.7 else body
    return join(opener(brand), body, closer() if R.random() < 0.45 else "")


# ----------------------------------------------------------------- English builders  (fn -> (ham, phishing twin or None))
HAM = []


def hm(f):
    HAM.append(f)
    return f


def _twin_threat():
    return pick(["your account will be suspended", "your card will be blocked", "access will be restricted", "your funds will be frozen",
                 "your account will be permanently closed", "a fee of " + money() + " will be charged", "your parcel will be returned",
                 "your service will be disconnected", "legal action will follow", "your payment will be reversed"])


def _twin_generic(brand):
    """Lures whose link was stripped or that point at a callback number: the shape
    of much real-world smishing (IMC25) that a link-only feature misses."""
    return pick([
        f"Dear Customer, your {brand} account is disabled. Please click the link to update your PAN number.",
        f"Dear {brand} user, your A/C will be blocked {date()}. Update your PAN card for KYC immediately via given link.",
        f"your {brand} account will be suspended today due to KYC expiry. Update KYC via the link below. Thank you, Team {brand}.",
        f"{brand}: your account has been temporarily locked. Click the link below to {_twin_ask()}.",
        f"{brand}: we tried to deliver your parcel but no one was home. Reschedule to avoid it being sent back :",
        f"{brand}: your package could not be delivered because customs duties of {money()} are unpaid. Please confirm your payment to avoid delays :",
        f"{brand}: your electricity will be disconnected tonight at {time_()} because last month's bill was not updated. Call {pick(['1-800', '+44 20', '+91 98', '0800'])} {R.randint(100, 999)} {R.randint(1000, 9999)} immediately.",
        f"{brand}: suspicious activity has been noticed on your account. You must confirm your details to prevent service termination.",
        f"{brand}: your points worth {money()} expire today. Redeem now by clicking the link.",
        f"{brand}: your refund of {money()} is ready. Provide your card details to receive it.",
        f"{brand} ALERT: a new payee was added to your account. If this was not you, call {pick(['0800', '1-877', '+44 33', '1-866'])} {R.randint(100, 999)} {R.randint(1000, 9999)} now.",
        f"{brand}: your card has been temporarily restricted. Reply with your card number and expiry date to remove the restriction.",
        f"{brand}: you have {R.randint(1, 3)} pending {pick(['message', 'voicemail', 'document', 'invoice'])}s. Click here to view.",
        f"{brand}: due to a security update your login will stop working. Re-register via given link before {date()}.",
        f"{brand}: your order {ref()} for {money()} has been placed. If you did not make this purchase, call {pick(['1-888', '1-877', '1-866'])}-{R.randint(200, 999)}-{R.randint(1000, 9999)} to cancel.",
        f"{brand}: final reminder. Your account closes in 24h unless you {_twin_ask()}. Click the link.",
    ])


def _twin_ask():
    return pick(["verify your identity", "confirm your details", "update your payment information", "confirm your card number and PIN",
                 "re-enter your login details", "validate your account", "confirm your date of birth and card details", "update your billing address",
                 "verify your SSN", "confirm your password", "complete verification"])


@hm
def bank_card_txn():
    b, d = pick(list(BANKS.items()))
    m = pick(MERCHANTS)
    amt = money()
    ham = pick([
        f"{amt} was charged to your card {masked()} at {m} on {date()}. If you don't recognize this, call the number on the back of your card.",
        f"a purchase of {amt} at {m} was approved on your card {masked()}.",
        f"your debit card {masked()} was declined for {amt} at {m}. If you don't recognize this transaction, call us at the number on your card.",
        f"transaction alert: {amt} at {m}, card {masked()}, {date()} {time_()}. Available balance {money()}.",
        f"{amt} was debited from your account {masked()} on {date()} for {m}. Not you? Call the number on your card.",
        f"your card {masked()} was used at {m} in {pick(CITIES)} for {amt}. If this wasn't you, freeze your card in {good_url(b, d)}.",
        f"a {amt} ATM withdrawal was made from account {masked()} at {time_()}. Balance: {money()}.",
        f"{amt} refund from {m} has been credited to your card {masked()}.",
        f"payment of {amt} to {pick(FIRST)} was sent from your account {masked()}. Ref {ref()}.",
        f"{amt} was deposited into your checking account {masked()}. Available balance {money()}.",
        f"your card {masked()} was used online at {m} for {amt}. You set up this alert for purchases over {money()}.",
        f"we declined a {amt} charge at {m} because it looked unusual. If it was you, reply YES and try again.",
        f"your recurring payment of {amt} to {m} was processed on {date()}.",
        f"your available balance on account {masked()} is {money()} as of {time_()}. Manage alerts in {good_url(b, d)}.",
    ])
    tw = pick([
        f"a payment of {amt} to {m} was approved on card {masked()}. If this was not you, cancel it now: {bad_url(b)}",
        f"suspicious charge of {amt} at {m}. Your card is locked. Unlock: {bad_url(b)}",
        f"{amt} was sent from your account {masked()} to an unknown recipient. Stop this payment within 30 min: {bad_url(b)}",
        f"unusual activity on card {masked()}. {_twin_ask().capitalize()} at {bad_url(b)} or {_twin_threat()}.",
    ])
    return with_brand(b, ham), with_brand(b, tw)


@hm
def bank_fraud_check():
    b, d = pick(list(BANKS.items()))
    m = pick(MERCHANTS)
    amt = money()
    ham = pick([
        f"did you make a {amt} purchase at {m} on {R.randint(1, 12)}/{R.randint(1, 28)}? Reply YES or NO.",
        f"did you attempt a {amt} purchase at {m} on {date()}? Reply 1 for YES, 2 for NO.",
        f"fraud alert: was this you? {amt} at {m}, card {masked()}. Reply Y or N.",
        f"we noticed a {amt} charge at {m} in {pick(CITIES)}. Reply YES if this was you, NO if not. We'll never ask for your PIN.",
        f"please confirm: {amt} at {m} on {date()} {time_()}. Reply YES to approve or NO to decline. Msg&data rates may apply.",
        f"is this you? {amt} to {m}, card {masked()}. Reply YES/NO. If NO, we'll cancel the card and send a new one.",
        f"security check: a login to your online banking from a new device in {pick(CITIES)}. Reply YES if this was you.",
        f"an attempted {amt} transaction at {m} was blocked. Reply YES if it was you to allow it, or NO to keep the block.",
    ])
    tw = pick([
        f"did you make a {amt} purchase at {m}? If NO, {_twin_ask()} at {bad_url(b)} or {_twin_threat()}.",
        f"fraud alert: {amt} at {m}. Reply with your full card number and expiry to cancel.",
        f"was this you? {amt} at {m}. Cancel and {_twin_ask()}: {bad_url(b)}",
    ])
    return with_brand(b, ham), with_brand(b, tw)


@hm
def bank_account_service():
    b, d = pick(list(BANKS.items()))
    ham = pick([
        f"your statement for account {masked()} is ready. Sign on at {good_url(b, d)} to view.",
        f"your {pick(['monthly', 'quarterly', 'credit card', 'savings'])} statement is now available in {good_url(b, d)}.",
        f"a new device signed in to your {b} account from {pick(CITIES)} on {date()}. If this wasn't you, change your password in {good_url(b, d)}.",
        f"your password was changed on {date()} at {time_()}. If you didn't do this, call the number on the back of your card.",
        f"your new card {masked()} has been mailed and should arrive in 5-7 business days. Activate it in {good_url(b, d)} when it arrives.",
        f"your card {masked()} expires this month. A replacement is on its way. No action needed.",
        f"your direct debit of {money()} to {pick(list(TELCOS) + list(UTIL))} is due {date()}.",
        f"your loan payment of {money()} was received on {date()}. Thank you.",
        f"your mortgage payment of {money()} is scheduled for {date()}. Make sure funds are available.",
        f"we've updated our privacy policy. Read it at {good_url(b, d)}. No action is needed.",
        f"online banking will be unavailable {date()} {time_()}-{time_()} for scheduled maintenance. We apologize for the inconvenience.",
        f"your standing order of {money()} to {pick(FIRST)} was paid today. Ref {ref()}.",
        f"interest of {money()} was credited to your savings account {masked()}.",
        f"your credit card payment of {money()} is due on {date()}. Pay in {good_url(b, d)} to avoid a late fee.",
        f"your card {masked()} was successfully added to Apple Pay.",
        f"we'll never ask for your PIN, full password or one-time code by text, phone or email. If in doubt, hang up and call the number on your card.",
        f"reminder: your travel notice for {pick(CITIES)} starts {date()}. Your card will work abroad as usual.",
        f"thanks for calling us today. Your request (ref {ref()}) has been completed.",
        f"your overdraft limit was changed to {money()} as requested on {date()}.",
        f"your appointment at the {pick(CITIES)} branch is confirmed for {date()} at {time_()}.",
        f"your account {masked()} has a low balance of {money()}. This is the alert you set up.",
    ])
    tw = pick([
        f"your statement has a problem. {_twin_ask().capitalize()} within 24h at {bad_url(b)} or {_twin_threat()}.",
        f"a new device signed in from {pick(CITIES)}. If this wasn't you, secure your account now: {bad_url(b)}",
        f"your online banking is suspended due to failed verification. Restore access: {bad_url(b)}",
        f"your new card is ready but we need you to {_twin_ask()} before it can be activated: {bad_url(b)}",
        f"your account will be closed within 24 hours unless you {_twin_ask()}: {bad_url(b)}",
    ])
    return with_brand(b, ham), with_brand(b, tw)


@hm
def p2p_payment():
    b, d = pick(list(P2P.items()))
    who = pick(FIRST)
    amt = money()
    emoji = pick(["🍕", "🍻", "🎟️", "🏠", "⛽", "🎂", "☕", "🚕", "for dinner", "for rent", "for the tickets", "for groceries", "for the trip", "for gas", "for lunch"])
    ham = pick([
        f"{who} paid you {amt} {emoji}. Transfer to your bank anytime in the app.",
        f"you paid {who} {amt} {emoji}.",
        f"{who} sent you {amt}. It's in your {b} balance.",
        f"{who} requested {amt} {emoji}. Open the app to pay or decline.",
        f"you received {amt} from {who}. Ref {ref()}.",
        f"your transfer of {amt} to your bank account {masked()} is on its way. It usually takes 1-3 business days.",
        f"{amt} has been added to your balance from {who}.",
        f"your payment of {amt} to {pick(MERCHANTS).title()} was completed.",
        f"you sent {amt} to {who}. If you didn't authorize this, contact us in the app.",
        f"{who} paid you {amt} {emoji}. Your balance is {money()}.",
        f"instant transfer of {amt} to card {masked()} completed. Fee: {pick(['$0.25', '$1.50', '1.75%', 'no fee'])}.",
        f"reminder: {who} still owes you {amt} {emoji}. Send a reminder in the app.",
    ])
    tw = pick([
        f"{who} sent you {amt}. To receive it, {_twin_ask()} at {bad_url(b)}.",
        f"you have a pending payment of {amt}. Claim it within 24 hours: {bad_url(b)}",
        f"your account has been limited. {_twin_ask().capitalize()} to restore: {bad_url(b)}",
        f"unusual login detected. Confirm your password and PIN here: {bad_url(b)}",
    ])
    return with_brand(b, ham), with_brand(b, tw)


@hm
def telco_bill():
    b, d = pick(list(TELCOS.items()))
    amt = money()
    ham = pick([
        f"your bill of {amt} is now available. View or pay at {good_url(b, d)}. Autopay will process on {date()}.",
        f"your bill of {amt} is due on {date()}. Pay at {good_url(b, d)} or dial {pick(['#PMT', '*611', '#PAY', '150'])} from your phone.",
        f"thanks for your payment of {amt}. Your next bill is due {date()}.",
        f"your {pick(['August', 'September', 'October', 'monthly'])} bill is ready: {amt}. It will be paid by autopay on {date()}.",
        f"you've used {pick(['50', '75', '80', '90', '100'])}% of your {pick(['5', '10', '20', '50', '100'])}GB data. Your allowance resets on {date()}.",
        f"your recharge of {amt} was successful. Validity {R.randint(28, 84)} days. Talktime balance {money()}.",
        f"planned maintenance in your area {date()} {time_()}-{time_()}. You may lose signal briefly. Sorry for the inconvenience.",
        f"your plan renews on {date()} at {amt}/month. Manage your plan in {good_url(b, d)}.",
        f"your bill is higher than usual this month ({amt}). See the breakdown at {good_url(b, d)}.",
        f"welcome to {b}! Your number is active. Download the app at {good_url(b, d)} to manage your account.",
        f"your device payment of {amt} was processed on {date()}. {R.randint(1, 23)} payments remaining.",
        f"your roaming pass for {pick(CITIES)} is active until {date()}. Standard rates apply after that.",
        f"a payment of {amt} was received. Your account balance is now {money()} in credit.",
        f"your data add-on of {pick(['1', '2', '5'])}GB has been applied. Valid until {date()}.",
        f"your SIM will be delivered {date()}. Activate it at {good_url(b, d)} once it arrives.",
        f"5G is now available in your area. No action needed, your phone will connect automatically.",
    ])
    tw = pick([
        f"your bill of {amt} is overdue. Pay now to avoid disconnection: {bad_url(b)}",
        f"your last payment failed. Update your card within 24h or {_twin_threat()}: {bad_url(b)}",
        f"you are due a refund of {amt} for overcharging. Claim it here: {bad_url(b)}",
        f"your number will be deactivated today. {_twin_ask().capitalize()}: {bad_url(b)}",
    ])
    return with_brand(b, ham), with_brand(b, tw)


@hm
def utility_bill():
    b, d = pick(list(UTIL.items()))
    amt = money()
    ham = pick([
        f"your {pick(['electricity', 'gas', 'water', 'energy'])} bill of {amt} is due on {date()}. Pay in the {b} app or at {good_url(b, d)}. Thank you.",
        f"your bill of {amt} has been issued. Payment due {date()}. Pay at {good_url(b, d)}.",
        f"we received your payment of {amt}. Thank you.",
        f"your meter reading is due. Submit it at {good_url(b, d)} by {date()} to avoid an estimated bill.",
        f"a power cut is affecting {pick(CITIES)}. Engineers are on site, estimated restoration {time_()}.",
        f"your direct debit of {amt} will be collected on {date()}.",
        f"your monthly payment is changing to {amt} from {date()} based on your usage. Details at {good_url(b, d)}.",
        f"an engineer will visit {date()} between {time_()} and {time_()} to install your smart meter.",
        f"your annual statement is ready. You're {money()} in credit. View it at {good_url(b, d)}.",
        f"planned water supply interruption {date()} {time_()}-{time_()} in your street. Please store some water in advance.",
        f"your usage this month: {R.randint(120, 980)} kWh, about {pick(['5', '10', '15'])}% {pick(['less', 'more'])} than last month.",
        f"your account is in credit by {amt}. We'll apply it to your next bill.",
    ])
    tw = pick([
        f"your bill of {amt} is unpaid. Your supply will be cut off tomorrow unless you pay here: {bad_url(b)}",
        f"you are owed a refund of {amt}. Enter your bank details to receive it: {bad_url(b)}",
        f"final notice: pay {amt} now to avoid disconnection: {bad_url(b)}",
    ])
    return with_brand(b, ham), with_brand(b, tw)


@hm
def delivery_notice():
    b, d = pick(list(COURIERS.items()))
    trk = ref()
    ham = pick([
        f"your shipment {trk} is scheduled for delivery {date()} between {time_()}-{time_()}. Manage delivery: {good_url(b, d)}",
        f"your parcel {trk} will be delivered {date()}. No action needed.",
        f"we delivered your package to {pick(['your front door', 'the mailroom', 'a neighbour at No. ' + str(R.randint(1, 99)), 'your safe place', 'reception', 'the porch', 'your parcel locker'])} at {time_()}.",
        f"sorry we missed you. Your parcel is at the {pick(CITIES)} depot. Collect it with photo ID or rebook at {good_url(b, d)}.",
        f"your driver {pick(FIRST)} is {R.randint(2, 9)} stops away. Estimated arrival {time_()}.",
        f"parcel {trk} has been collected from the sender and is on its way. Track at {good_url(b, d)}.",
        f"your package is out for delivery today. Track it at {good_url(b, d)}.",
        f"your parcel {trk} is ready for collection at the {pick(CITIES)} post office. Bring ID. It will be held for 10 days.",
        f"delivered: your package {trk} was left {pick(['at the front door', 'in the mailbox', 'with a neighbour', 'at the back door'])}. Photo in the app.",
        f"your parcel is running late and will now arrive {date()}. Sorry for the delay.",
        f"your return {trk} was received by the sender. Your refund will be processed by them.",
        f"delivery attempted at {time_()}. We'll try again {date()}. To change delivery, visit {good_url(b, d)}.",
        f"your package from {pick(list(SHOPS))} ({ref()}) has shipped via {b}. Expected {date()}.",
        f"your parcel is waiting at the {pick(['Tesco', 'Co-op', 'Shell', '7-Eleven', 'Spar'])} pickup point, {R.randint(1, 200)} {pick(['High St', 'Main St', 'Station Rd'])}. Collect within 7 days.",
    ])
    tw = pick([
        f"your parcel {trk} is held, a customs fee of {money()} is unpaid. Pay to release: {bad_url(b)}",
        f"we could not deliver your package. Reschedule and pay the {money()} redelivery fee: {bad_url(b)}",
        f"your parcel will be returned to sender unless you confirm your address here: {bad_url(b)}",
        f"delivery failed: incomplete address. Update your details within 24h: {bad_url(b)}",
    ])
    return with_brand(b, ham), with_brand(b, tw)


@hm
def order_notice():
    b, d = pick(list(SHOPS.items()))
    item = pick(ITEMS)
    amt = money()
    ham = pick([
        f"your order {ref()} ({item}) has shipped and will arrive {date()}. Track it at {good_url(b, d)}.",
        f"your package with {item} was delivered. It was {pick(['handed to a resident', 'left at the front door', 'left in the mailroom', 'left with the concierge', 'placed in your parcel box'])}.",
        f"thanks for your order! {item.capitalize()}, {amt}. Est. delivery {date()}. Manage your order at {good_url(b, d)}.",
        f"{item.capitalize()} is out for delivery today. Someone should be available to sign.",
        f"your return for {item} was received. Refund of {amt} will appear on your card in 3-5 business days.",
        f"order {ref()} is ready for {pick(['pickup', 'collection'])} at the {pick(CITIES)} store. Bring your order confirmation.",
        f"your order {ref()} has been confirmed. Total {amt}. You'll get another text when it ships.",
        f"good news, {item} is back in stock. Your backorder {ref()} will ship {date()}.",
        f"your refund of {amt} for order {ref()} has been issued to your original payment method.",
        f"your subscribe & save delivery of {item} is scheduled for {date()}. Skip or change it at {good_url(b, d)}.",
        f"your order {ref()} was cancelled as requested. Any charge will be reversed within 5 days.",
        f"your {b} gift card of {amt} from {pick(FIRST)} has been added to your account.",
        f"a delivery driver will arrive between {time_()} and {time_()} with your order {ref()}.",
        f"your order arrives {date()}. It's currently {pick(['in transit', 'at the local depot', 'with the courier'])}. Track at {good_url(b, d)}.",
        f"your recent order {ref()} was delivered. How did we do? Rate your delivery in the app.",
        f"your prescription glasses order {ref()} is ready for collection in store.",
    ])
    tw = pick([
        f"your order {ref()} for {item} ({amt}) has shipped. If you did not place this order, call {pick(['1-888', '1-877', '1-866'])}-{R.randint(200, 999)}-{R.randint(1000, 9999)} within 24h.",
        f"your account has been charged {amt} for {item}. Cancel the order here: {bad_url(b)}",
        f"we could not confirm your payment for order {ref()}. Update your card within 12h: {bad_url(b)}",
        f"your order of {item} is on hold. {_twin_ask().capitalize()} to release it: {bad_url(b)}",
    ])
    return with_brand(b, ham), with_brand(b, tw)


@hm
def subscription_notice():
    b, d = pick(list(SUBS.items()))
    amt = money()
    ham = pick([
        f"your payment of {amt} was processed. Next billing date {date()}. Manage your plan in Account settings.",
        f"a new device signed in to your account from {pick(CITIES)}. If this wasn't you, change your password in Account settings.",
        f"your free trial ends {date()}. You'll be charged {amt}/month unless you cancel in the app.",
        f"we couldn't process your payment. Please update your payment method in the {b} app to keep your plan.",
        f"receipt: {amt} for your {pick(['monthly', 'annual', 'family', 'student', 'premium', 'standard'])} plan. Thank you!",
        f"your plan renews on {date()} for {amt}. No action needed.",
        f"your password was changed. If you didn't do this, go to {good_url(b, d)} right away.",
        f"your subscription has been cancelled as requested. You'll have access until {date()}.",
        f"your annual plan will renew on {date()} at {amt}. To change or cancel, visit {good_url(b, d)}.",
        f"welcome back! Your {b} membership is active again. Enjoy.",
        f"your gift subscription from {pick(FIRST)} has been activated. It runs until {date()}.",
        f"your payment method {masked()} expires soon. Update it in Account settings to avoid interruption.",
        f"price update: from {date()} your plan will be {amt}/month. Details at {good_url(b, d)}.",
        f"your download of {pick(['the new season', 'your playlist', 'the album'])} is ready for offline listening.",
    ])
    tw = pick([
        f"your payment failed and your account is suspended. Update your card to restore access: {bad_url(b)}",
        f"your membership has expired. Renew within 24h to keep your profile: {bad_url(b)}",
        f"we detected an unusual login. Confirm your password here or {_twin_threat()}: {bad_url(b)}",
        f"your account is on hold due to a billing problem. {_twin_ask().capitalize()}: {bad_url(b)}",
    ])
    return with_brand(b, ham), with_brand(b, tw)


@hm
def account_security():
    b, d = pick(list(TECH.items()))
    ham = pick([
        f"your {b} account was used to sign in on a new {pick(['iPhone 15', 'Windows PC', 'Android device', 'Mac', 'iPad', 'Samsung Galaxy'])}. If you didn't sign in, change your password at {good_url(b, d)}.",
        f"your password was changed on {date()}. If you didn't do this, go to {good_url(b, d)} to secure your account.",
        f"two-step verification is now on for your account. You'll be asked for a code when you sign in on a new device.",
        f"a new sign-in from {pick(CITIES)} at {time_()}. Was this you? Review recent activity in your account settings.",
        f"your recovery phone number was updated. If this wasn't you, review your security settings at {good_url(b, d)}.",
        f"security alert: a new app was granted access to your account. Review it at {good_url(b, d)}.",
        f"your email address was verified. Thanks for keeping your account secure.",
        f"a backup of your account data has been completed.",
        f"your storage is {pick(['80', '90', '95'])}% full. Manage storage in the app.",
        f"your account is now protected with a passkey. You can sign in without a password on this device.",
        f"you signed out of all devices as requested. Sign in again on the devices you use.",
        f"reminder: we never ask for your password by text or email. Learn more at {good_url(b, d)}.",
        f"your ride with {pick(FIRST)} is complete. {money()} was charged to card {masked()}. Rate your trip in the app.",
        f"your booking at {pick(['The Grand', 'Hotel Central', 'a private room in ' + pick(CITIES)])} is confirmed for {date()}. Check-in after {time_()}.",
    ])
    tw = pick([
        f"your account will be deactivated in 24 hours due to a policy violation. Appeal here: {bad_url(b)}",
        f"unusual sign-in detected. {_twin_ask().capitalize()} to keep your account: {bad_url(b)}",
        f"your account has been compromised. Reset your password immediately: {bad_url(b)}",
        f"your account is locked. Confirm your password and phone number here: {bad_url(b)}",
    ])
    return with_brand(b, ham), with_brand(b, tw)


@hm
def otp_code():
    b, d = pick(list(TECH.items()) + list(BANKS.items()) + list(P2P.items()) + list(SUBS.items()))
    c = code()
    ham = pick([
        f"your {b} code is {c}. Never share this code with anyone.",
        f"{c} is your {b} verification code. It expires in {pick([5, 10, 15])} minutes.",
        f"{b}: {c} is your one-time passcode. Do not share it. We will never call to ask for it.",
        f"use {c} to sign in to {b}. If you didn't request this, ignore this message.",
        f"{c} is your {b} OTP for a payment of {money()} to {pick(MERCHANTS).title()}. Valid 10 mins. Do not share.",
        f"your {b} security code: {c}. Don't share this with anyone, including {b} staff.",
        f"G-{c} is your Google verification code.",
        f"<#> {c} is your {b} code. {pick(['H5s3Xk9', 'FA+9qCX9VSu', 'kJ2mN8pQ'])}",
        f"{b}: your login code is {c}. If you didn't try to log in, change your password.",
        f"{c} — your {b} confirmation code for {pick(['a new device', 'password reset', 'a transfer of ' + money(), 'adding a payee'])}. Valid 5 min.",
        f"your {b} authentication code is {c}. Enter it on the {pick(['website', 'app', 'checkout page'])} to continue.",
        f"{b}: {c} is the code to link your new phone. Enter it in the app.",
    ])
    tw = pick([
        f"{b}: we sent you a code by mistake. Please reply with the code {c} to cancel the transaction.",
        f"your {b} code is {c}. A support agent will call to collect it and secure your account.",
        f"your one-time code {c} was used to authorize {money()}. If not you, {_twin_ask()}: {bad_url(b)}",
    ])
    return ham if R.random() < 0.6 else with_brand(b, ham.replace(f"{b}: ", "")), tw


@hm
def appointment():
    doc = pick(DOCTORS)
    h, hd = pick(list(HEALTH.items()))
    ham = pick([
        f"Reminder: your {pick(['dentist', 'dental', 'eye', 'physio', 'GP', 'blood test', 'MOT', 'hair', 'vet'])} appointment with {doc} is {date()} at {time_()}. Reply C to confirm or R to reschedule.",
        f"{h}: your appointment with {doc} is confirmed for {date()} at {time_()}. Please arrive 10 minutes early.",
        f"Your appointment at {pick(CITIES)} {pick(['Dental', 'Medical Centre', 'Health Clinic', 'Eye Care'])} is {date()} at {time_()}. Reply 1 to confirm, 2 to cancel.",
        f"{h}: you have a new test result. Sign in to {good_url(h, hd)} to view it.",
        f"{h}: your prescription is ready for pickup at {R.randint(100, 9999)} {pick(['Main St', 'High St', 'Market St', 'Station Rd'])}. Reply STOP to opt out.",
        f"{h}: your prescription for {pick(['amoxicillin', 'your regular medication', 'lisinopril', 'your inhaler'])} is ready. Store hours {time_()}-{time_()}.",
        f"{h}: your {pick(['flu', 'COVID', 'travel', 'HPV'])} vaccine appointment is booked for {date()} at {time_()}. Bring your ID.",
        f"{doc}'s office: we need to reschedule your {date()} appointment. Please call us at your convenience to rebook.",
        f"{h}: your lab results are ready. Your doctor will call to discuss them by {date()}.",
        f"Your {pick(['car', 'boiler', 'washing machine', 'broadband'])} service visit is booked for {date()} between {time_()} and {time_()}. Engineer: {pick(FIRST)}.",
        f"{h}: a message from {doc} is waiting in your patient portal at {good_url(h, hd)}.",
        f"Your {pick(['haircut', 'massage', 'nail', 'facial'])} appointment at {pick(['Bliss Salon', 'Studio 9', 'The Barber Shop', 'Urban Spa'])} is {date()} {time_()}. Reply CANCEL to cancel.",
        f"{h}: it's time for your annual check-up. Book online at {good_url(h, hd)} or call the surgery.",
        f"NHS: your {pick(['GP', 'hospital', 'outpatient'])} appointment is on {date()} at {time_()} at {pick(CITIES)} Hospital. Please bring any medication you take.",
    ])
    tw = pick([
        f"{h}: your test results show an urgent issue. View them here immediately: {bad_url(h)}",
        f"{h}: you have an unpaid bill of {money()}. Pay within 24h to avoid collections: {bad_url(h)}",
        f"NHS: you are eligible for a new vaccine. Apply for your pass here and enter your bank details for the {money()} fee: {bad_url('nhs')}",
    ])
    return ham, tw


@hm
def government_notice():
    b, d = pick(list(GOV.items()))
    amt = money()
    ham = pick([
        f"your tax return has been received. Ref {ref()}. No action is needed unless we contact you.",
        f"your refund of {amt} has been approved and will be paid into your bank account within 5 working days.",
        f"your application {ref()} has been received. You can check its status at {good_url(b, d)}.",
        f"reminder: your {pick(['vehicle tax', 'driving licence', 'passport', 'vehicle registration', 'MOT'])} expires on {date()}. Renew at {good_url(b, d)}.",
        f"your {pick(['passport', 'ID card', 'driving licence', 'residence permit'])} is ready for collection at the {pick(CITIES)} office. Bring your receipt.",
        f"your payment of {amt} was received. Thank you.",
        f"your {pick(['benefit', 'pension', 'child benefit', 'universal credit'])} payment of {amt} will be paid on {date()}.",
        f"we will never contact you by text asking for bank details. Report suspicious messages to {pick(['60599', '7726', 'phishing@irs.gov', 'report@phishing.gov.uk'])}.",
        f"your appointment at the {pick(CITIES)} office is confirmed for {date()} at {time_()}. Bring photo ID.",
        f"your vehicle registration renewal has been processed. Your new sticker will arrive by mail within 2 weeks.",
        f"your jury service is confirmed for {date()}. Report to {pick(CITIES)} Courthouse at {time_()}.",
        f"polling day is {date()}. Your polling station is {pick(['St Mary\'s Hall', 'the Community Centre', 'Lincoln Elementary', 'the Town Hall'])}.",
        f"your {pick(['council tax', 'property tax', 'water rates'])} for this year is {amt}. Pay in instalments at {good_url(b, d)}.",
        f"a document is waiting in your secure inbox. Sign in at {good_url(b, d)} to read it.",
        f"your appeal {ref()} has been logged. We aim to respond within 30 days.",
        f"weather alert: {pick(['flash flood', 'severe thunderstorm', 'high wind', 'extreme heat', 'winter storm'])} warning for {pick(CITIES)} until {time_()}. Avoid unnecessary travel.",
    ])
    tw = pick([
        f"you are owed a tax refund of {amt}. Claim it within 48h or forfeit: {bad_url(b)}",
        f"you have an unpaid fine of {amt}. Pay today to avoid prosecution: {bad_url(b)}",
        f"your benefits will be stopped unless you {_twin_ask()} here: {bad_url(b)}",
        f"a warrant has been issued in your name. Call {pick(['1-888', '1-877', '+44 20', '+1 202'])} {R.randint(200, 999)} {R.randint(1000, 9999)} immediately.",
    ])
    return with_brand(b, ham), with_brand(b, tw)


@hm
def insurance_loan():
    b, d = pick(list(INSURE.items()) + list(BANKS.items()))
    amt = money()
    ham = pick([
        f"your premium of {amt} was received. Your policy {ref()} is active until {date()}.",
        f"your policy renews on {date()}. Your new premium is {amt}. View your documents at {good_url(b, d)}.",
        f"your EMI of {amt} for loan {ref()} was debited on {date()}. {R.randint(3, 48)} EMIs remaining.",
        f"your claim {ref()} has been approved. {amt} will be paid to your account within 7 days.",
        f"your claim {ref()} was received. An adjuster will contact you within 2 business days.",
        f"reminder: your auto insurance payment of {amt} is due {date()}. Pay at {good_url(b, d)}.",
        f"your ID card is ready. Download it from the app or {good_url(b, d)}.",
        f"your loan application {ref()} has been approved. The funds will be in your account by {date()}.",
        f"your monthly loan payment of {amt} was received. Thank you.",
        f"your policy documents have been updated. No action needed.",
        f"your credit score updated: {R.randint(620, 810)}. See what changed at {good_url(b, d)}.",
        f"your card payment due date is {date()}. Minimum due {money()}. Pay at {good_url(b, d)}.",
    ])
    tw = pick([
        f"your policy has lapsed due to a failed payment. Reinstate within 24h: {bad_url(b)}",
        f"you are pre-approved for a {amt} loan. {_twin_ask().capitalize()} to receive it today: {bad_url(b)}",
        f"your claim payout of {amt} is waiting. Enter your card details to receive it: {bad_url(b)}",
    ])
    return with_brand(b, ham), with_brand(b, tw)


@hm
def travel_notice():
    b, d = pick(list(TRAVEL.items()))
    ham = pick([
        f"flight {pick('ABDELUQ')}{pick('AEHKRUS')}{R.randint(100, 9999)} to {pick(CITIES)} is now departing from gate {pick('ABCDE')}{R.randint(1, 45)}. Boarding begins at {time_()}.",
        f"check-in is open for your flight to {pick(CITIES)} on {date()}. Check in at {good_url(b, d)}.",
        f"your flight {pick('ABDELUQ')}{pick('AEHKRUS')}{R.randint(100, 9999)} is delayed by {R.randint(20, 180)} minutes. New departure {time_()}. We're sorry.",
        f"your booking {ref()} is confirmed. {pick(CITIES)} to {pick(CITIES)}, {date()}. Manage your booking at {good_url(b, d)}.",
        f"your driver {pick(FIRST)} is arriving in a {pick(['gray Toyota Camry', 'black Prius', 'white Tesla Model 3', 'silver Skoda Octavia', 'blue Honda Civic'])}, plate {pick('ABCDEFGH')}{R.randint(10, 99)}{pick('KLMNP')}{R.randint(100, 999)}.",
        f"your ride is complete. {money()} charged to card {masked()}. Rate {pick(FIRST)} in the app.",
        f"your host {pick(FIRST)} sent you a message: check-in is after {time_()}, the lockbox code is in the app.",
        f"your reservation at {pick(['Hotel Central', 'The Grand', 'Park Inn', 'Riverside Suites'])} in {pick(CITIES)} is confirmed for {date()}. Check-in from {time_()}.",
        f"your train {pick(['ICE', 'TGV', 'AVE', 'Frecciarossa', 'Acela'])} {R.randint(100, 9999)} departs {time_()} from platform {R.randint(1, 20)}. Seat {R.randint(1, 80)}{pick('ABCDF')}, coach {R.randint(1, 12)}.",
        f"your e-ticket for {date()} is attached in the app. Show it at the gate.",
        f"gate change: your flight to {pick(CITIES)} now departs from gate {pick('ABCDE')}{R.randint(1, 45)}.",
        f"your baggage has been loaded on flight {pick('ABDELUQ')}{pick('AEHKRUS')}{R.randint(100, 9999)}. Collect it at belt {R.randint(1, 12)} on arrival.",
        f"thanks for riding with us. Your receipt for {money()} is in the app.",
        f"your booking is cancelled as requested. Refund of {money()} will be processed within 7 days.",
        f"your trip to {pick(CITIES)} starts {date()}. Passport valid? Check entry requirements at {good_url(b, d)}.",
    ])
    tw = pick([
        f"your flight has been cancelled. Claim your refund of {money()} here: {bad_url(b)}",
        f"your booking could not be confirmed. Re-enter your card details within 2 hours: {bad_url(b)}",
        f"your ride receipt has a {money()} overcharge. Get your refund: {bad_url(b)}",
    ])
    return with_brand(b, ham), with_brand(b, tw)


@hm
def reservation_community():
    ham = pick([
        f"Your table for {R.randint(2, 8)} at {pick(['Nobu', 'Dishoom', 'Carbone', 'The Ivy', 'Osteria Francescana', 'Olive & Thyme', 'Le Bernardin', 'Sushi Nakazawa', 'The Fat Duck'])} is confirmed for {date()} {time_()}. Reply CANCEL to cancel.",
        f"OpenTable: your reservation at {pick(['Nobu', 'Dishoom', 'Carbone', 'Zuma'])} for {R.randint(2, 6)} on {date()} at {time_()} is confirmed.",
        f"Your {pick(['gym', 'yoga', 'spin', 'pilates'])} class at {time_()} {date()} is booked. Cancel up to 2 hours before in the app.",
        f"School: {pick(['early pickup', 'no school', 'sports day', 'parents evening', 'school photos'])} on {date()}. See the newsletter for details.",
        f"{pick(['Lincoln', 'Oakwood', 'St Mary\'s', 'Riverside'])} School: {pick(FIRST)} was marked absent today. Reply with the reason or call the office.",
        f"Library: your reserved book is ready for pickup. Hold expires {date()}.",
        f"Library: 2 items are due {date()}. Renew online or in the app.",
        f"Your car is ready for collection from {pick(['Kwik Fit', 'Halfords', 'Jiffy Lube', 'the dealership', 'Midas'])}. Total {money()}.",
        f"Practice is cancelled tonight because of the {pick(['rain', 'heat', 'storm'])}. Coach will send the new schedule tomorrow.",
        f"Your parcel locker code is {code()}. Locker {R.randint(1, 60)} at {pick(CITIES)} station. Valid 3 days.",
        f"Your dry cleaning is ready for pickup. Total {money()}. Open until {time_()}.",
        f"Your {pick(['gym', 'club', 'co-working'])} membership renews on {date()} at {money()}. Manage it at reception or in the app.",
        f"Building management: water will be shut off {date()} {time_()}-{time_()} for repairs.",
        f"Your {pick(['Amazon Fresh', 'Instacart', 'Ocado', 'Tesco', 'Getir'])} grocery delivery is arriving in {R.randint(5, 40)} minutes.",
        f"Your food order from {pick(['Chipotle', 'Nando\'s', 'Pizza Express', 'Wagamama', 'Five Guys'])} is on its way with {pick(FIRST)}. ETA {R.randint(5, 35)} min.",
        f"Ticketmaster: your tickets for {pick(['Coldplay', 'the match', 'Hamilton', 'the comedy show'])} on {date()} are in your account. Show the barcode at the gate.",
        f"Your parking session at {pick(CITIES)} {pick(['Central', 'Station', 'Market'])} car park expires in 15 min. Extend in the app.",
        f"Your {pick(['Tesla', 'BMW', 'Toyota'])} is fully charged. Charging session ended at {time_()}. Cost {money()}.",
        f"Vet: {pick(['Max', 'Bella', 'Luna', 'Charlie'])}'s vaccination is due. Book at your convenience.",
        f"Your interview with {pick(['Deloitte', 'Google', 'Tesla', 'Marriott', 'IKEA'])} is confirmed for {date()} at {time_()}. Reply if you need to reschedule. - {pick(FIRST)}, Recruiting",
        f"Thanks for coming to the interview today. We'll be in touch by {date()} with next steps. - {pick(FIRST)}, HR",
        f"Payroll: your payslip for {pick(['August', 'September', 'this month'])} is available in Workday.",
        f"HR: open enrollment for benefits closes {date()}. Make your selections in the HR portal.",
        f"IT: your password expires in {R.randint(1, 7)} days. Change it from a company device.",
    ])
    return ham, None


# ----------------------------------------------------------------- non-English pools
I18N = {
    "es": [
        lambda: (f"{pick(['BBVA', 'CaixaBank', 'Santander', 'Bankinter', 'ING'])}: compra de {money('es')} en {pick(MERCHANTS).title()} con tarjeta {masked()}. Si no la reconoce, llame al número del reverso de su tarjeta.",
                 f"{pick(['BBVA', 'CaixaBank', 'Santander'])}: cargo sospechoso de {money('es')}. Cancélelo ahora: {bad_url('bbva')}"),
        lambda: (f"{pick(['Movistar', 'Vodafone', 'Orange', 'Yoigo'])}: su factura de {money('es')} ya está disponible. Puede consultarla en la app o en {pick(['movistar.es', 'vodafone.es', 'orange.es'])}.",
                 f"{pick(['Movistar', 'Vodafone'])}: factura impagada de {money('es')}. Evite el corte de línea pagando aquí: {bad_url('movistar')}"),
        lambda: (f"Correos: su envío {ref()} será entregado {pick(['hoy', 'mañana'])} entre las {time_()} y las {time_()}. No es necesario hacer nada.",
                 f"Correos: su paquete está retenido por una tasa de aduana de {money('es')}. Páguela aquí: {bad_url('correos')}"),
        lambda: (f"Bizum: {pick(FIRST)} te ha enviado {money('es')}. Ya está en tu cuenta.",
                 f"Bizum: tienes un pago pendiente de {money('es')}. Acéptalo en 24h aquí: {bad_url('bizum')}"),
        lambda: (f"{pick(['Iberdrola', 'Endesa', 'Naturgy'])}: su factura de {money('es')} se cargará en su cuenta el {date()}. Consúltela en la app.",
                 f"{pick(['Iberdrola', 'Endesa'])}: factura pendiente de {money('es')}. Evite el corte de suministro: {bad_url('iberdrola')}"),
        lambda: (f"Recordatorio: tiene cita con {pick(DOCTORS)} el {date()} a las {time_()} en {pick(['Sanitas', 'Quirónsalud', 'el centro de salud'])}. Responda C para confirmar.",
                 None),
        lambda: (f"Amazon: su pedido {ref()} ha sido enviado y llegará {pick(['mañana', 'el ' + date()])}. Siga el envío en amazon.es.",
                 f"Amazon: no hemos podido cobrar su pedido {ref()}. Actualice su tarjeta en 12h: {bad_url('amazon')}"),
        lambda: (f"Agencia Tributaria: su declaración ha sido presentada correctamente. Ref {ref()}. No es necesaria ninguna acción.",
                 f"Agencia Tributaria: tiene una devolución pendiente de {money('es')}. Solicítela aquí: {bad_url('aeat')}"),
        lambda: (f"Tu código de verificación de {pick(['BBVA', 'Google', 'WhatsApp', 'Glovo'])} es {code()}. No lo compartas con nadie.", None),
        lambda: (f"DGT: su permiso de conducir caduca el {date()}. Renuévelo en dgt.es o en su jefatura provincial.",
                 f"DGT: tiene una multa sin pagar de {money('es')}. Pague hoy para evitar el recargo: {bad_url('dgt')}"),
        lambda: (f"Glovo: tu pedido de {pick(['Burger King', 'Telepizza', 'Domino\'s'])} está en camino con {pick(FIRST)}. Llega en {R.randint(5, 30)} min.", None),
        lambda: (f"Renfe: su tren AVE {R.randint(2000, 9999)} sale a las {time_()} por la vía {R.randint(1, 20)}. Coche {R.randint(1, 12)}, plaza {R.randint(1, 80)}{pick('ABCD')}.", None),
    ],
    "fr": [
        lambda: (f"{pick(['BNP Paribas', 'Société Générale', 'Crédit Agricole', 'LCL', 'Boursorama'])} : paiement de {money('fr')} chez {pick(MERCHANTS).title()} avec la carte {masked()}. En cas de doute, appelez le numéro au dos de votre carte.",
                 f"{pick(['BNP Paribas', 'Crédit Agricole'])} : opération suspecte de {money('fr')}. Annulez-la ici : {bad_url('bnp')}"),
        lambda: (f"{pick(['Orange', 'SFR', 'Bouygues Telecom', 'Free'])} : votre facture de {money('fr')} est disponible dans votre espace client. Prélèvement le {date()}.",
                 f"{pick(['Orange', 'SFR', 'Free'])} : facture impayée de {money('fr')}. Régularisez sous 24h pour éviter la suspension : {bad_url('orange')}"),
        lambda: (f"Colissimo : votre colis {ref()} sera livré {pick(['aujourd\'hui', 'demain'])} entre {time_()} et {time_()}. Suivi sur laposte.fr.",
                 f"Colissimo : votre colis est en attente, frais de douane de {money('fr')} à régler : {bad_url('laposte')}"),
        lambda: (f"Chronopost : votre colis a été livré à {time_()} et remis {pick(['en main propre', 'au gardien', 'en boîte aux lettres'])}.",
                 f"Chronopost : livraison impossible, adresse incomplète. Mettez à jour vos coordonnées sous 24h : {bad_url('chronopost')}"),
        lambda: (f"Ameli : votre remboursement de {money('fr')} a été effectué le {date()}. Détail sur ameli.fr.",
                 f"Ameli : votre nouvelle carte Vitale est disponible. Commandez-la et réglez les frais de {money('fr')} : {bad_url('ameli')}"),
        lambda: (f"impots.gouv : votre déclaration a bien été enregistrée. Aucune action n'est nécessaire.",
                 f"impots.gouv : vous avez un remboursement de {money('fr')} en attente. Réclamez-le ici : {bad_url('impots')}"),
        lambda: (f"Doctolib : rappel de votre rendez-vous avec {pick(DOCTORS)} le {date()} à {time_()}. Pour annuler, connectez-vous à doctolib.fr.", None),
        lambda: (f"Votre code de connexion {pick(['Doctolib', 'Lydia', 'Google', 'La Poste'])} est {code()}. Il expire dans 10 minutes.", None),
        lambda: (f"Lydia : {pick(FIRST)} vous a envoyé {money('fr')} pour {pick(['le resto', 'les courses', 'le loyer', 'les billets'])}.",
                 f"Lydia : un paiement de {money('fr')} est en attente. Confirmez votre identité pour le recevoir : {bad_url('lydia')}"),
        lambda: (f"{pick(['EDF', 'Engie', 'TotalEnergies'])} : votre facture de {money('fr')} sera prélevée le {date()}. Consultez-la dans votre espace client.",
                 f"{pick(['EDF', 'Engie'])} : facture impayée. Évitez la coupure en réglant {money('fr')} ici : {bad_url('edf')}"),
        lambda: (f"SNCF Connect : votre TGV {R.randint(6000, 8999)} part à {time_()} voie {R.randint(1, 20)}. Voiture {R.randint(1, 18)}, place {R.randint(1, 100)}.", None),
        lambda: (f"Uber : votre chauffeur {pick(FIRST)} arrive dans {R.randint(2, 8)} min en {pick(['Toyota Prius grise', 'Tesla blanche', 'Peugeot 508 noire'])}, plaque {pick('ABCDEFGH')}{pick('ABCDEFGH')}-{R.randint(100, 999)}-{pick('KLMNP')}{pick('KLMNP')}.", None),
    ],
    "de": [
        lambda: (f"{pick(['Sparkasse', 'Commerzbank', 'Deutsche Bank', 'ING', 'DKB', 'N26'])}: {money('de')} wurden heute von Ihrer Karte {masked()} bei {pick(MERCHANTS).title()} abgebucht. Bei Fragen nutzen Sie die Nummer auf Ihrer Karte.",
                 f"{pick(['Sparkasse', 'Commerzbank', 'ING'])}: Verdächtige Abbuchung von {money('de')}. Jetzt stornieren: {bad_url('sparkasse')}"),
        lambda: (f"{pick(['Telekom', 'Vodafone', 'O2', '1&1'])}: Ihre Rechnung über {money('de')} ist online verfügbar. Abbuchung am {date()}.",
                 f"{pick(['Telekom', 'Vodafone', 'O2'])}: Ihre Rechnung ist überfällig. Zahlen Sie innerhalb von 24 Std., sonst wird Ihr Anschluss gesperrt: {bad_url('telekom')}"),
        lambda: (f"DHL: Ihr Paket {ref()} wird {pick(['heute', 'morgen'])} zwischen {time_()} und {time_()} Uhr zugestellt. Sendungsverfolgung unter dhl.de.",
                 f"DHL: Ihr Paket wartet im Zolllager. Zollgebühr von {money('de')} bezahlen: {bad_url('dhl')}"),
        lambda: (f"Hermes: Ihr Paket wurde um {time_()} Uhr {pick(['beim Nachbarn', 'im Paketshop', 'an der Haustür'])} abgegeben.",
                 f"Hermes: Zustellung fehlgeschlagen. Adresse innerhalb von 24 Std. bestätigen: {bad_url('hermes')}"),
        lambda: (f"PayPal: Sie haben {money('de')} von {pick(FIRST)} erhalten. Der Betrag ist in Ihrem Guthaben.",
                 f"PayPal: Ihr Konto wurde eingeschränkt. Bestätigen Sie Ihre Daten: {bad_url('paypal')}"),
        lambda: (f"{pick(['E.ON', 'Vattenfall', 'EnBW', 'Stadtwerke'])}: Ihr Abschlag von {money('de')} wird am {date()} abgebucht.",
                 f"{pick(['E.ON', 'Vattenfall'])}: Offene Rechnung über {money('de')}. Stromsperre vermeiden: {bad_url('eon')}"),
        lambda: (f"ELSTER: Ihre Steuererklärung ist eingegangen. Es ist nichts weiter zu tun.",
                 f"Finanzamt: Steuererstattung von {money('de')} verfügbar. Jetzt beantragen: {bad_url('elster')}"),
        lambda: (f"Terminerinnerung: {date()} um {time_()} Uhr bei {pick(DOCTORS)}. Bitte antworten Sie mit JA zur Bestätigung.", None),
        lambda: (f"Dein {pick(['Google', 'WhatsApp', 'Sparkasse', 'DKB', 'Lieferando'])} Bestätigungscode lautet {code()}. Gib ihn niemals weiter.", None),
        lambda: (f"Deutsche Bahn: Ihr ICE {R.randint(100, 999)} fährt um {time_()} Uhr von Gleis {R.randint(1, 20)}. Wagen {R.randint(1, 12)}, Platz {R.randint(1, 80)}.", None),
        lambda: (f"Amazon: Ihre Bestellung {ref()} wurde versandt und kommt {pick(['morgen', 'am ' + date()])} an. Sendung verfolgen auf amazon.de.",
                 f"Amazon: Zahlung für Bestellung {ref()} fehlgeschlagen. Zahlungsdaten innerhalb von 12 Std. aktualisieren: {bad_url('amazon')}"),
        lambda: (f"Lieferando: Deine Bestellung bei {pick(['Pizza Hut', 'Burgerme', 'Sushi Palace'])} ist unterwegs mit {pick(FIRST)}. Ankunft in {R.randint(5, 30)} Min.", None),
    ],
    "pt": [
        lambda: (f"{pick(['Itaú', 'Nubank', 'Bradesco', 'Banco do Brasil', 'Caixa'])}: compra de {money('pt')} aprovada no cartão final {last4()} em {pick(MERCHANTS).title()}. Não reconhece? Ligue para o número no verso do cartão.",
                 f"{pick(['Itaú', 'Nubank', 'Bradesco'])}: compra suspeita de {money('pt')}. Cancele agora: {bad_url('itau')}"),
        lambda: (f"PIX recebido: {money('pt')} de {pick(FIRST)}. Já está na sua conta.",
                 f"PIX pendente de {money('pt')}. Confirme seus dados para receber: {bad_url('pix')}"),
        lambda: (f"Correios: seu objeto {ref()} saiu para entrega. Acompanhe em correios.com.br.",
                 f"Correios: objeto retido, taxa de {money('pt')} pendente. Pague para liberar: {bad_url('correios')}"),
        lambda: (f"{pick(['Vivo', 'Claro', 'TIM', 'Oi'])}: sua fatura de {money('pt')} vence em {date()}. Pague pelo app ou em {pick(['vivo.com.br', 'claro.com.br', 'tim.com.br'])}.",
                 f"{pick(['Vivo', 'Claro', 'TIM'])}: fatura em atraso. Evite o bloqueio da linha pagando aqui: {bad_url('vivo')}"),
        lambda: (f"Mercado Livre: seu pedido {ref()} foi enviado e chega {pick(['amanhã', 'em ' + date()])}. Acompanhe no app.",
                 f"Mercado Livre: problema no pagamento do pedido {ref()}. Atualize seu cartão em 12h: {bad_url('mercadolivre')}"),
        lambda: (f"Seu código de verificação {pick(['Nubank', 'WhatsApp', 'iFood', 'Google'])} é {code()}. Não compartilhe com ninguém.", None),
        lambda: (f"iFood: seu pedido do {pick(['McDonald\'s', 'Habib\'s', 'Outback'])} está a caminho com {pick(FIRST)}. Chega em {R.randint(5, 35)} min.", None),
        lambda: (f"Receita Federal: sua declaração foi recebida. Nenhuma ação é necessária.",
                 f"Receita Federal: você tem uma restituição de {money('pt')} pendente. Solicite aqui: {bad_url('receita')}"),
        lambda: (f"Lembrete: consulta com {pick(DOCTORS)} em {date()} às {time_()}. Responda SIM para confirmar.", None),
        lambda: (f"{pick(['Enel', 'Light', 'CPFL', 'Sabesp'])}: sua conta de {money('pt')} vence em {date()}. Pague pelo app ou débito automático.",
                 f"{pick(['Enel', 'Light'])}: conta em atraso. Evite o corte pagando {money('pt')} aqui: {bad_url('enel')}"),
    ],
    "it": [
        lambda: (f"{pick(['Intesa Sanpaolo', 'UniCredit', 'Poste Italiane', 'Fineco'])}: addebito di {money('it')} sulla carta {masked()} presso {pick(MERCHANTS).title()}. Se non lo riconosci, chiama il numero sul retro della carta.",
                 f"{pick(['Intesa Sanpaolo', 'UniCredit'])}: operazione sospetta di {money('it')}. Blocca subito: {bad_url('intesa')}"),
        lambda: (f"{pick(['TIM', 'Vodafone', 'WindTre', 'Iliad'])}: la tua fattura di {money('it')} è disponibile nell'area clienti. Addebito il {date()}.",
                 f"{pick(['TIM', 'Vodafone'])}: fattura non pagata. Evita la sospensione della linea: {bad_url('tim')}"),
        lambda: (f"Poste Italiane: la tua spedizione {ref()} è in consegna oggi. Traccia su poste.it.",
                 f"Poste Italiane: pacco in giacenza, spese doganali di {money('it')} da pagare: {bad_url('poste')}"),
        lambda: (f"Il tuo codice di verifica {pick(['Intesa', 'Satispay', 'Google', 'PosteID'])} è {code()}. Non condividerlo con nessuno.", None),
        lambda: (f"Promemoria: appuntamento con {pick(DOCTORS)} il {date()} alle {time_()}. Rispondi OK per confermare.", None),
        lambda: (f"Trenitalia: il tuo Frecciarossa {R.randint(9000, 9999)} parte alle {time_()} dal binario {R.randint(1, 20)}. Carrozza {R.randint(1, 11)}, posto {R.randint(1, 80)}{pick('ABCD')}.", None),
        lambda: (f"Satispay: {pick(FIRST)} ti ha inviato {money('it')} per {pick(['la cena', 'i biglietti', 'la spesa'])}.",
                 f"Satispay: pagamento di {money('it')} in sospeso. Conferma la tua identità: {bad_url('satispay')}"),
        lambda: (f"Agenzia Entrate: la tua dichiarazione è stata ricevuta. Nessuna azione richiesta.",
                 f"Agenzia Entrate: rimborso di {money('it')} disponibile. Richiedilo entro 48h: {bad_url('agenziaentrate')}"),
    ],
    "nl": [
        lambda: (f"{pick(['ING', 'Rabobank', 'ABN AMRO', 'bunq'])}: {money('nl')} afgeschreven van rekening {masked()} bij {pick(MERCHANTS).title()}. Herkent u dit niet? Bel het nummer op uw pas.",
                 f"{pick(['ING', 'Rabobank', 'ABN AMRO'])}: verdachte afschrijving van {money('nl')}. Annuleer direct: {bad_url('ing')}"),
        lambda: (f"PostNL: uw pakket {ref()} wordt {pick(['vandaag', 'morgen'])} tussen {time_()} en {time_()} bezorgd. Volg het via postnl.nl.",
                 f"PostNL: uw pakket ligt bij de douane. Betaal {money('nl')} invoerkosten: {bad_url('postnl')}"),
        lambda: (f"Tikkie: {pick(FIRST)} heeft je {money('nl')} betaald voor {pick(['het eten', 'de tickets', 'de boodschappen'])}.",
                 f"Tikkie: je hebt nog een openstaand betaalverzoek van {money('nl')}. Betaal hier: {bad_url('tikkie')}"),
        lambda: (f"{pick(['KPN', 'Odido', 'Vodafone', 'Ziggo'])}: uw factuur van {money('nl')} staat klaar in Mijn {pick(['KPN', 'Odido', 'Ziggo'])}. Incasso op {date()}.",
                 f"{pick(['KPN', 'Ziggo'])}: uw factuur is niet betaald. Voorkom afsluiting: {bad_url('kpn')}"),
        lambda: (f"Uw DigiD-code is {code()}. Deel deze code met niemand.", None),
        lambda: (f"Belastingdienst: uw aangifte is ontvangen. U hoeft niets te doen.",
                 f"Belastingdienst: u krijgt {money('nl')} terug. Vraag het hier aan: {bad_url('belastingdienst')}"),
        lambda: (f"Herinnering: afspraak met {pick(DOCTORS)} op {date()} om {time_()}. Antwoord JA om te bevestigen.", None),
        lambda: (f"bol.com: je bestelling {ref()} is onderweg en wordt {pick(['vandaag', 'morgen'])} bezorgd. Volg je pakket in de app.",
                 f"bol.com: betaling mislukt voor bestelling {ref()}. Werk je gegevens binnen 12 uur bij: {bad_url('bol')}"),
    ],
    "ru": [
        lambda: (f"{pick(['Сбербанк', 'Тинькофф', 'Альфа-Банк', 'ВТБ'])}: покупка {money('ru')} в {pick(MERCHANTS).title()}, карта {masked()}. Баланс: {money('ru')}.",
                 f"{pick(['Сбербанк', 'Тинькофф', 'ВТБ'])}: подозрительная операция на {money('ru')}. Отмените здесь: {bad_url('sber')}"),
        lambda: (f"Сбербанк: перевод {money('ru')} от {pick(['Ивана К.', 'Ольги М.', 'Дмитрия С.', 'Анны П.'])} зачислен на карту {masked()}. Баланс: {money('ru')}.",
                 f"Сбербанк: перевод {money('ru')} ожидает подтверждения. Подтвердите данные карты: {bad_url('sber')}"),
        lambda: (f"Госуслуги: ваше заявление №{R.randint(10000, 99999)} принято. Статус можно посмотреть в личном кабинете на gosuslugi.ru.",
                 f"Госуслуги: вам положена выплата {money('ru')}. Получите её здесь: {bad_url('gosuslugi')}"),
        lambda: (f"{pick(['МТС', 'Билайн', 'МегаФон', 'Tele2'])}: счёт за {pick(['сентябрь', 'октябрь'])} — {money('ru')}. Оплатите в приложении до {date()}.",
                 f"{pick(['МТС', 'Билайн'])}: задолженность {money('ru')}. Во избежание блокировки номера оплатите: {bad_url('mts')}"),
        lambda: (f"СДЭК: ваш заказ {ref()} прибыл в пункт выдачи по адресу {pick(['ул. Ленина', 'пр. Мира', 'ул. Пушкина'])} {R.randint(1, 99)}. Хранение 7 дней.",
                 f"СДЭК: посылка задержана, оплатите пошлину {money('ru')}: {bad_url('cdek')}"),
        lambda: (f"Ваш код подтверждения {pick(['Сбербанк', 'Госуслуги', 'Ozon', 'Яндекс'])}: {code()}. Никому его не сообщайте.", None),
        lambda: (f"Ozon: заказ {ref()} доставлен в пункт выдачи. Заберите до {date()}.",
                 f"Ozon: оплата заказа {ref()} не прошла. Обновите данные карты в течение 12 часов: {bad_url('ozon')}"),
        lambda: (f"Напоминание: приём у {pick(['терапевта', 'стоматолога', 'врача'])} {date()} в {time_()}. Ответьте ДА для подтверждения.", None),
        lambda: (f"Яндекс Go: водитель {pick(['Иван', 'Сергей', 'Алексей'])} на {pick(['белой Kia Rio', 'серой Skoda Octavia', 'чёрном Hyundai Solaris'])} подъедет через {R.randint(2, 8)} мин.", None),
    ],
    "tr": [
        lambda: (f"{pick(['Garanti BBVA', 'Ziraat', 'İş Bankası', 'Akbank', 'Yapı Kredi'])}: {last4()} ile biten kartınızdan {pick(MERCHANTS).title()} işlem yerinde {money('tr')} harcama yapıldı. Tanımıyorsanız kartınızın arkasındaki numarayı arayın.",
                 f"{pick(['Garanti BBVA', 'Ziraat', 'Akbank'])}: şüpheli işlem {money('tr')}. Hemen iptal edin: {bad_url('garanti')}"),
        lambda: (f"{pick(['Turkcell', 'Vodafone', 'Türk Telekom'])}: {money('tr')} tutarındaki faturanız hazır. Son ödeme tarihi {date()}. Uygulamadan ödeyebilirsiniz.",
                 f"{pick(['Turkcell', 'Vodafone'])}: faturanız gecikti. Hattınızın kapanmaması için hemen ödeyin: {bad_url('turkcell')}"),
        lambda: (f"Yurtiçi Kargo: {ref()} numaralı gönderiniz bugün dağıtıma çıktı. Takip: yurticikargo.com",
                 f"Yurtiçi Kargo: gönderiniz gümrükte bekliyor. {money('tr')} vergi ödeyin: {bad_url('yurtici')}"),
        lambda: (f"e-Devlet: başvurunuz alınmıştır. Başvuru no {ref()}. Durumunu turkiye.gov.tr üzerinden takip edebilirsiniz.",
                 f"e-Devlet: {money('tr')} iadeniz var. Almak için bilgilerinizi doğrulayın: {bad_url('edevlet')}"),
        lambda: (f"{pick(['Garanti', 'Trendyol', 'Google', 'WhatsApp'])} doğrulama kodunuz: {code()}. Kimseyle paylaşmayın.", None),
        lambda: (f"Trendyol: {ref()} numaralı siparişiniz kargoya verildi. {pick(['Yarın', date()])} teslim edilecek.",
                 f"Trendyol: siparişinizin ödemesi alınamadı. 12 saat içinde kartınızı güncelleyin: {bad_url('trendyol')}"),
        lambda: (f"Hatırlatma: {pick(DOCTORS)} ile randevunuz {date()} saat {time_()}. Onaylamak için EVET yazın.", None),
    ],
    "id": [
        lambda: (f"{pick(['BCA', 'Mandiri', 'BRI', 'BNI'])}: transaksi {money('id')} di {pick(MERCHANTS).title()} pada kartu {masked()} berhasil. Jika bukan Anda, hubungi nomor di belakang kartu.",
                 f"{pick(['BCA', 'Mandiri', 'BRI'])}: transaksi mencurigakan {money('id')}. Batalkan sekarang: {bad_url('bca')}"),
        lambda: (f"{pick(['Telkomsel', 'Indosat', 'XL'])}: tagihan Anda {money('id')} jatuh tempo {date()}. Bayar lewat aplikasi My{pick(['Telkomsel', 'IM3', 'XL'])}.",
                 f"{pick(['Telkomsel', 'XL'])}: tagihan belum dibayar. Hindari pemblokiran nomor: {bad_url('telkomsel')}"),
        lambda: (f"JNE: paket {ref()} sedang dalam pengiriman dan tiba {pick(['hari ini', 'besok'])}. Lacak di jne.co.id.",
                 f"JNE: paket tertahan di bea cukai. Bayar {money('id')} untuk pelepasan: {bad_url('jne')}"),
        lambda: (f"Tokopedia: pesanan {ref()} telah dikirim. Cek status di aplikasi.",
                 f"Tokopedia: pembayaran pesanan {ref()} gagal. Perbarui kartu dalam 12 jam: {bad_url('tokopedia')}"),
        lambda: (f"GoPay: Anda menerima {money('id')} dari {pick(FIRST)}. Saldo Anda {money('id')}.",
                 f"GoPay: saldo {money('id')} tertunda. Verifikasi akun Anda: {bad_url('gopay')}"),
        lambda: (f"Kode verifikasi {pick(['BCA', 'Gojek', 'Tokopedia', 'WhatsApp'])} Anda: {code()}. Jangan bagikan kepada siapa pun.", None),
        lambda: (f"BPJS Kesehatan: pembayaran iuran {money('id')} telah diterima. Terima kasih.",
                 f"BPJS: kepesertaan Anda akan dinonaktifkan. Verifikasi data: {bad_url('bpjs')}"),
        lambda: (f"Pengingat: janji temu dengan {pick(DOCTORS)} pada {date()} pukul {time_()}. Balas YA untuk konfirmasi.", None),
    ],
    "ar": [
        lambda: (f"{pick(['الراجحي', 'بنك الإمارات دبي الوطني', 'بنك قطر الوطني', 'البنك الأهلي'])}: تم خصم {money('ar')} من بطاقتك {masked()} لدى {pick(MERCHANTS).title()}. إذا لم تتعرف على العملية اتصل بالرقم خلف البطاقة.",
                 f"{pick(['الراجحي', 'البنك الأهلي'])}: عملية مشبوهة بقيمة {money('ar')}. ألغها الآن: {bad_url('alrajhi')}"),
        lambda: (f"{pick(['STC', 'اتصالات', 'زين', 'موبايلي'])}: فاتورتك بقيمة {money('ar')} جاهزة. يمكنك السداد عبر التطبيق قبل {date()}.",
                 f"{pick(['STC', 'زين'])}: فاتورتك متأخرة. تجنب قطع الخدمة بالدفع هنا: {bad_url('stc')}"),
        lambda: (f"أرامكس: شحنتك {ref()} خرجت للتوصيل اليوم. تتبع عبر aramex.com",
                 f"أرامكس: شحنتك محتجزة في الجمارك. ادفع {money('ar')} للإفراج عنها: {bad_url('aramex')}"),
        lambda: (f"رمز التحقق الخاص بك من {pick(['الراجحي', 'STC Pay', 'نون', 'أبشر'])} هو {code()}. لا تشاركه مع أحد.", None),
        lambda: (f"أبشر: تم تجديد رخصة القيادة بنجاح. رقم الطلب {ref()}.",
                 f"أبشر: لديك مخالفة غير مدفوعة بقيمة {money('ar')}. ادفع اليوم لتجنب الحجز: {bad_url('absher')}"),
        lambda: (f"نون: تم شحن طلبك {ref()} وسيصل {pick(['غداً', date()])}. تابع الطلب عبر التطبيق.",
                 f"نون: فشل الدفع للطلب {ref()}. حدّث بطاقتك خلال 12 ساعة: {bad_url('noon')}"),
        lambda: (f"تذكير: موعدك مع {pick(DOCTORS)} يوم {date()} الساعة {time_()}. أرسل نعم للتأكيد.", None),
        lambda: (f"كريم: الكابتن {pick(['أحمد', 'محمد', 'خالد'])} في الطريق إليك بسيارة {pick(['كامري بيضاء', 'هيونداي رمادية'])} ويصل خلال {R.randint(2, 8)} دقائق.", None),
    ],
    "hi": [
        lambda: (f"{pick(['SBI', 'HDFC Bank', 'ICICI Bank', 'Axis Bank', 'PNB'])}: Rs.{pick(['250', '1,200', '2,500', '4,999', '12,000'])}.00 debited from A/c XX{last4()} on {R.randint(1, 28)}-{R.randint(1, 12)}-26 to VPA {pick(['grocery', 'swiggy', 'paytm', 'amazon'])}@ok{pick(['axis', 'hdfc', 'sbi'])}. Not you? Call 1800 {R.randint(100, 999)} {R.randint(1000, 9999)}.",
                 f"{pick(['SBI', 'HDFC Bank', 'ICICI Bank'])}: your account will be blocked today due to KYC expiry. Update PAN/Aadhaar here: {bad_url('sbi')}"),
        lambda: (f"{pick(['Airtel', 'Jio', 'Vi'])}: your recharge of Rs.{pick(['239', '299', '479', '719'])} is successful. Validity {R.randint(28, 84)} days. Data {pick(['1.5', '2', '3'])}GB/day.",
                 f"{pick(['Airtel', 'Jio', 'Vi'])}: your SIM will be blocked in 24 hours. Complete KYC: {bad_url('airtel')}"),
        lambda: (f"Delhivery: your {pick(['Flipkart', 'Myntra', 'Amazon'])} order {ref()} is out for delivery. OTP {code()}. Delivery by {time_()}.",
                 f"Delhivery: delivery failed, update address and pay Rs.{R.randint(10, 99)} redelivery fee: {bad_url('delhivery')}"),
        lambda: (f"{code()} is your {pick(['UPI', 'PhonePe', 'Paytm', 'SBI', 'IRCTC'])} OTP. Valid for 10 mins. Do not share with anyone. -{pick(['SBI', 'HDFC', 'PhonePe'])}", None),
        lambda: (f"PhonePe: Rs.{pick(['150', '500', '1,200', '2,000'])} received from {pick(FIRST)}. UPI Ref {ref()}.",
                 f"PhonePe: Rs.{pick(['2,500', '5,000'])} cashback pending. Claim by entering your UPI PIN here: {bad_url('phonepe')}"),
        lambda: (f"EPFO: your PF contribution of Rs.{pick(['1,800', '3,600', '5,400'])} for {pick(['Aug', 'Sep'])} has been credited. Balance available at epfindia.gov.in.",
                 f"EPFO: your PF account is blocked. Verify Aadhaar here to unblock: {bad_url('epfo')}"),
        lambda: (f"Reminder: your appointment with {pick(DOCTORS)} at Apollo Hospital is on {date()} at {time_()}. Reply YES to confirm.", None),
        lambda: (f"Kal {time_()} baje {pick(DOCTORS)} ke saath appointment hai. Confirm karne ke liye YES reply karein.", None),
        lambda: (f"IRCTC: your ticket PNR {R.randint(1000000000, 9999999999)} is confirmed. Train {R.randint(10000, 22999)}, coach {pick('SB')}{R.randint(1, 9)}, berth {R.randint(1, 72)}. Departs {time_()}.", None),
        lambda: (f"Swiggy: your order from {pick(['Domino\'s', 'Biryani Blues', 'Haldiram\'s'])} is on the way with {pick(['Rahul', 'Suresh', 'Amit'])}. Arriving in {R.randint(5, 35)} min.", None),
    ],
    "el": [
        lambda: (f"{pick(['Alpha Bank', 'Πειραιώς', 'Eurobank', 'Εθνική'])}: χρέωση {money('el')} στην κάρτα {masked()} στο {pick(MERCHANTS).title()}. Αν δεν την αναγνωρίζετε, καλέστε τον αριθμό στο πίσω μέρος της κάρτας.",
                 f"{pick(['Alpha Bank', 'Πειραιώς'])}: ύποπτη συναλλαγή {money('el')}. Ακυρώστε τώρα: {bad_url('alpha')}"),
        lambda: (f"ΔΕΗ: Ο λογαριασμός σας ύψους {money('el')} εκδόθηκε. Λήξη πληρωμής {date()}. Πληρώστε στο dei.gr",
                 f"ΔΕΗ: ληξιπρόθεσμος λογαριασμός. Αποφύγετε τη διακοπή ρεύματος: {bad_url('dei')}"),
        lambda: (f"{pick(['Cosmote', 'Vodafone', 'Nova'])}: ο λογαριασμός σας {money('el')} είναι διαθέσιμος στο My {pick(['Cosmote', 'Vodafone'])}. Λήξη {date()}.",
                 f"{pick(['Cosmote', 'Vodafone'])}: απλήρωτος λογαριασμός. Πληρώστε σε 24 ώρες: {bad_url('cosmote')}"),
        lambda: (f"ΕΛΤΑ Courier: το δέμα σας {ref()} παραδίδεται σήμερα. Παρακολούθηση στο elta-courier.gr",
                 f"ΕΛΤΑ: το δέμα σας κρατείται στο τελωνείο. Πληρώστε {money('el')}: {bad_url('elta')}"),
        lambda: (f"Ο κωδικός επαλήθευσης {pick(['gov.gr', 'Alpha', 'Skroutz', 'Viber'])} είναι {code()}. Μην τον μοιραστείτε.", None),
        lambda: (f"gov.gr: η αίτησή σας {ref()} καταχωρήθηκε. Δεν απαιτείται καμία ενέργεια.",
                 f"ΑΑΔΕ: έχετε επιστροφή φόρου {money('el')}. Δηλώστε ΙΒΑΝ εδώ: {bad_url('aade')}"),
    ],
    "pl": [
        lambda: (f"{pick(['PKO BP', 'mBank', 'ING', 'Santander', 'Pekao'])}: transakcja {money('pl')} kartą {masked()} w {pick(MERCHANTS).title()}. Jeśli to nie Ty, zadzwoń pod numer z odwrotu karty.",
                 f"{pick(['PKO BP', 'mBank'])}: podejrzana transakcja {money('pl')}. Anuluj teraz: {bad_url('pko')}"),
        lambda: (f"InPost: Twoja paczka {ref()} czeka w Paczkomacie {pick(['WAW', 'KRA', 'GDA'])}{R.randint(10, 99)}{pick('ABM')}. Kod odbioru: {code()}. Odbierz w ciągu 48h.",
                 f"InPost: paczka wstrzymana, dopłata {money('pl')}: {bad_url('inpost')}"),
        lambda: (f"{pick(['Orange', 'Play', 'Plus', 'T-Mobile'])}: faktura na {money('pl')} jest dostępna. Termin płatności {date()}.",
                 f"{pick(['Orange', 'Play'])}: zaległa faktura. Uniknij blokady numeru: {bad_url('play')}"),
        lambda: (f"Twój kod {pick(['BLIK', 'mBank', 'Allegro', 'ePUAP'])}: {code()}. Nie udostępniaj go nikomu.", None),
        lambda: (f"Allegro: zamówienie {ref()} zostało wysłane. Dostawa {pick(['jutro', date()])}.",
                 f"Allegro: płatność za zamówienie {ref()} nie powiodła się. Zaktualizuj kartę: {bad_url('allegro')}"),
        lambda: (f"Przypomnienie: wizyta u {pick(DOCTORS)} {date()} o {time_()}. Odpowiedz TAK, aby potwierdzić.", None),
    ],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12000, help="total legitimate rows to aim for")
    ap.add_argument("--twin-frac", type=float, default=0.3, help="share of ham rows that also emit a phishing twin")
    ap.add_argument("--i18n-frac", type=float, default=0.35, help="share of rows drawn from non-English pools")
    ap.add_argument("--seed", type=int, default=23)
    ap.add_argument("--twins-only", action="store_true", help="emit only phishing twins (a second file, minimal pairs for the ham set)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    R.seed(args.seed)
    langs = list(I18N)
    seen, rows = set(), []
    counts = {}

    def add(text, label, cat, lang):
        text = " ".join(text.split())
        k = text.lower()
        if k in seen or len(text) < 20:
            return False
        seen.add(k)
        rows.append({"text": text, "label": label, "category": cat, "language": lang})
        counts[(label, lang)] = counts.get((label, lang), 0) + 1
        return True

    made = 0
    attempts = 0
    while made < args.n and attempts < args.n * 20:
        attempts += 1
        if R.random() < args.i18n_frac:
            lang = pick(langs)
            ham, tw = pick(I18N[lang])()
            cat = "i18n_notice"
        else:
            lang = "en"
            fn = pick(HAM)
            ham, tw = fn()
            cat = fn.__name__
        if args.twins_only:
            if lang == "en" and R.random() < 0.5:
                brand = ham.split(":")[0].strip("[]*() ") if ":" in ham[:30] else pick(list(BANKS) + list(COURIERS) + list(TELCOS) + list(SHOPS))
                tw = with_brand(brand, _twin_generic(brand)) if R.random() < 0.5 else _twin_generic(brand)
            if tw and add(tw, "phishing", cat + "_twin", lang):
                made += 1
            continue
        if add(ham, "ham", cat, lang):
            made += 1
            if tw and R.random() < args.twin_frac:
                add(tw, "phishing", cat + "_twin", lang)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["text", "label", "category", "language"])
        w.writeheader()
        w.writerows(rows)
    by_label = {}
    for (label, lang), n in counts.items():
        by_label[label] = by_label.get(label, 0) + n
    print(f"{len(rows)} rows -> {out}")
    print("by label:", by_label)
    print("ham by language:", {l: n for (lab, l), n in sorted(counts.items()) if lab == "ham"})


if __name__ == "__main__":
    main()
