#!/usr/bin/env python3
"""Generate notice-shaped attacks paired with legitimate twins.

Model 2.7 blocks too many real bank, parcel and code messages, and adding real
notices alone taught the 2.8c candidate to pass notice-shaped lures (see
evals/results/DATA_CLEANUP_2.8.md). Both sides of that boundary need examples,
so every scam template here has a benign twin about the same brand and topic:
the difference is the behaviour, not the vocabulary.

Wording is written fresh rather than copied from the fable5 suite, and rows too
close to any eval message are dropped, so the benchmarks stay honest.

    python evals/generate_notice_pairs.py --pairs 500

Output: dataset/curated/synthetic_notice_pairs_v1.csv, in the candidate schema
(id, pool, text, label), so evals/label_audit.py checks it like any other
addition: only rows TypeSafe agrees with are used.
"""

import argparse
import csv
import json
import random
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_CSV = REPO_ROOT / "src/mBERT/training/model-training/dataset/curated/synthetic_notice_pairs_v1.csv"
EVAL_FILES = [REPO_ROOT / "evals/datasets/fable5_adversarial_v1.csv",
              REPO_ROOT / "evals/datasets/hard_legit_a2p_v1.csv",
              REPO_ROOT / "evals/datasets/mishra_soni_5971.csv"]
csv.field_size_limit(10 * 1024 * 1024)

BANKS = {"en": ["Halifax", "TSB", "Nationwide", "Citizens Bank", "Regions Bank"],
         "es": ["Bankinter", "Abanca", "Unicaja", "Banco Falabella"],
         "de": ["Volksbank", "Targobank", "Postbank", "Norisbank"],
         "fr": ["Crédit Mutuel", "Banque Populaire", "LCL", "Caisse d'Epargne"],
         "it": ["Banca Sella", "Credem", "BPER", "Banco Desio"],
         "pt": ["Millennium", "Novo Banco", "Banco CTT"],
         "nl": ["Knab", "SNS", "Regiobank"],
         "ru": ["Уралсиб", "Совкомбанк", "Росбанк"],
         "tr": ["Denizbank", "Vakıfbank", "Şekerbank"],
         "id": ["Bank Mega", "Bank Permata", "OCBC NISP"],
         "ar": ["بنك الرياض", "بنك الجزيرة", "بنك دبي"]}
COURIERS = {"en": ["Evri", "Yodel", "OnTrac"], "es": ["MRW", "Nacex", "GLS"], "de": ["GLS", "Hermes", "Trans-o-flex"],
            "fr": ["Colissimo", "Mondial Relay", "GLS"], "it": ["SDA", "GLS", "Poste Delivery"],
            "pt": ["CTT Expresso", "Rangel"], "nl": ["DHL Parcel", "GLS", "DPD"], "ru": ["Боксберри", "СДЭК", "Почта"],
            "tr": ["Yurtiçi Kargo", "MNG Kargo", "Sürat Kargo"], "id": ["SiCepat", "AnterAja", "Ninja Xpress"],
            "ar": ["أرامكس", "سمسا", "زاجل"]}
SHADY_HOSTS = ["-verify-now.top", "-secure-id.info", "-account-check.xyz", "-billing-update.online",
               "-portal-login.cc", "-customer-verify.live", "-id-confirm.site"]
SHORTENERS = ["cutt.ly/", "rb.gy/", "t.ly/", "shorturl.at/", "is.gd/"]


def slug(name):
    return re.sub(r"[^a-z0-9]", "", name.lower())[:12] or "service"


class Gen:
    def __init__(self, rng):
        self.rng = rng

    def pick(self, seq):
        return self.rng.choice(seq)

    def bad_url(self, brand):
        if self.rng.random() < 0.4:
            return self.pick(SHORTENERS) + "".join(self.rng.choices("abcdefghjkmnpqrstuvwxyz23456789", k=6))
        return slug(brand) + self.pick(SHADY_HOSTS)

    def good_url(self, brand, lang):
        tld = {"en": "com", "es": "es", "de": "de", "fr": "fr", "it": "it", "pt": "pt", "nl": "nl",
               "ru": "ru", "tr": "com.tr", "id": "co.id", "ar": "com"}[lang]
        return f"{slug(brand)}.{tld}"

    def code(self):
        return str(self.rng.randint(1000, 999999))

    def amount(self, lang):
        n = self.rng.choice([1.99, 2.45, 3.50, 9.90, 12.40, 19.99, 24.50, 48.00, 129.90, 249.00])
        return {"en": f"£{n}", "es": f"{n}€", "de": f"{n} €", "fr": f"{n} €", "it": f"{n} €", "pt": f"{n}€",
                "nl": f"€{n}", "ru": f"{int(n * 90)} ₽", "tr": f"{int(n * 34)} TL", "id": f"Rp{int(n * 16000)}",
                "ar": f"{n} ريال"}[lang]

    def phone(self):
        return self.pick(["+44 7", "+1 8", "+34 6", "+49 15", "+62 8"]) + "".join(self.rng.choices("0123456789", k=8))

    def last4(self):
        return str(self.rng.randint(1000, 9999))

    def tracking(self):
        return "".join(self.rng.choices("0123456789", k=self.rng.choice([10, 12])))


# Each entry: scenario -> {lang: (lure, legitimate twin)}. The twin keeps the
# brand and topic, and differs in what it asks the reader to do.
SCENARIOS = {
    "account_block": {
        "en": ("{bank}: we could not verify your details, so online banking is suspended. Restore access within 24h: {bad}",
               "{bank}: online banking will be unavailable on Sunday 02:00-05:00 for maintenance. No action needed."),
        "es": ("{bank}: no hemos podido verificar sus datos y su banca online queda suspendida. Restablezca el acceso en 24h: {bad}",
               "{bank}: su banca online estará en mantenimiento el domingo de 02:00 a 05:00. No necesita hacer nada."),
        "de": ("{bank}: Ihre Daten konnten nicht bestätigt werden, das Online-Banking ist gesperrt. Zugang in 24 Std. freischalten: {bad}",
               "{bank}: Am Sonntag von 02:00 bis 05:00 Uhr ist das Online-Banking wegen Wartung nicht erreichbar."),
        "fr": ("{bank} : vos informations n'ont pas pu être vérifiées, votre banque en ligne est suspendue. Rétablissez l'accès sous 24h : {bad}",
               "{bank} : maintenance de la banque en ligne dimanche de 02h00 à 05h00. Aucune action de votre part."),
        "it": ("{bank}: non siamo riusciti a verificare i tuoi dati, l'home banking è sospeso. Ripristina l'accesso entro 24h: {bad}",
               "{bank}: domenica dalle 02:00 alle 05:00 l'home banking sarà in manutenzione. Nessuna azione richiesta."),
        "ru": ("{bank}: не удалось подтвердить ваши данные, интернет-банк заблокирован. Восстановите доступ за 24 часа: {bad}",
               "{bank}: в воскресенье с 02:00 до 05:00 интернет-банк будет недоступен из-за техработ."),
        "tr": ("{bank}: bilgileriniz doğrulanamadı, internet bankacılığınız askıya alındı. 24 saat içinde erişimi açın: {bad}",
               "{bank}: Pazar günü 02:00-05:00 arası internet bankacılığı bakımda olacaktır. İşlem yapmanıza gerek yok."),
        "id": ("{bank}: data Anda tidak dapat diverifikasi, internet banking diblokir. Pulihkan akses dalam 24 jam: {bad}",
               "{bank}: internet banking dalam pemeliharaan Minggu pukul 02.00-05.00. Tidak ada tindakan yang diperlukan."),
        "ar": ("{bank}: تعذر التحقق من بياناتك وتم تعليق الخدمة المصرفية. استعد الوصول خلال 24 ساعة: {bad}",
               "{bank}: ستكون الخدمة المصرفية عبر الإنترنت تحت الصيانة يوم الأحد من 02:00 حتى 05:00."),
    },
    "card_charge": {
        "en": ("{bank}: a payment of {amt} to a new merchant was approved on card ending {l4}. Cancel it here: {bad}",
               "{bank}: {amt} was debited from card ending {l4} today. If you do not recognise it, call the number on your card."),
        "es": ("{bank}: se aprobó un pago de {amt} a un comercio nuevo con la tarjeta terminada en {l4}. Cancélelo aquí: {bad}",
               "{bank}: cargo de {amt} en su tarjeta terminada en {l4}. Si no lo reconoce, llame al número del reverso."),
        "de": ("{bank}: Zahlung über {amt} an einen neuen Händler mit Karte {l4} freigegeben. Hier stornieren: {bad}",
               "{bank}: {amt} wurden heute von Karte {l4} abgebucht. Bei Fragen nutzen Sie die Nummer auf Ihrer Karte."),
        "fr": ("{bank} : paiement de {amt} autorisé chez un nouveau commerçant avec la carte {l4}. Annulez ici : {bad}",
               "{bank} : {amt} débités aujourd'hui de la carte {l4}. En cas de doute, appelez le numéro au dos de la carte."),
        "it": ("{bank}: pagamento di {amt} autorizzato presso un nuovo esercente con la carta {l4}. Annulla qui: {bad}",
               "{bank}: addebito di {amt} sulla carta {l4}. Se non lo riconosci, chiama il numero sul retro della carta."),
        "ru": ("{bank}: платёж {amt} новому получателю по карте {l4} одобрен. Отмените здесь: {bad}",
               "{bank}: списание {amt} по карте {l4}. Если это не вы, позвоните по номеру на обороте карты."),
        "tr": ("{bank}: {l4} ile biten kartınızdan yeni bir üye iş yerine {amt} ödeme onaylandı. Buradan iptal edin: {bad}",
               "{bank}: {l4} ile biten kartınızdan {amt} tutarında işlem yapıldı. Tanımıyorsanız kartınızın arkasındaki numarayı arayın."),
        "id": ("{bank}: pembayaran {amt} ke merchant baru dari kartu {l4} disetujui. Batalkan di sini: {bad}",
               "{bank}: transaksi {amt} pada kartu {l4} berhasil. Jika bukan Anda, hubungi nomor di belakang kartu."),
        "ar": ("{bank}: تمت الموافقة على دفعة {amt} لتاجر جديد من البطاقة {l4}. ألغها هنا: {bad}",
               "{bank}: تم خصم {amt} من بطاقتك {l4}. إذا لم تتعرف على العملية اتصل بالرقم خلف البطاقة."),
    },
    "parcel_fee": {
        "en": ("{courier}: parcel {trk} is held, a customs fee of {amt} is unpaid. Pay to release: {bad}",
               "{courier}: parcel {trk} is out for delivery today between 14:00 and 18:00."),
        "es": ("{courier}: el paquete {trk} está retenido por una tasa aduanera de {amt}. Pague para liberarlo: {bad}",
               "{courier}: su paquete {trk} sale hoy a reparto entre las 14:00 y las 18:00."),
        "de": ("{courier}: Paket {trk} wird zurückgehalten, Zollgebühr von {amt} offen. Zum Freigeben zahlen: {bad}",
               "{courier}: Paket {trk} wird heute zwischen 14:00 und 18:00 Uhr zugestellt."),
        "fr": ("{courier} : le colis {trk} est bloqué, frais de douane de {amt} impayés. Payez pour le libérer : {bad}",
               "{courier} : votre colis {trk} sera livré aujourd'hui entre 14h00 et 18h00."),
        "it": ("{courier}: il pacco {trk} è fermo, tassa doganale di {amt} non pagata. Paga per liberarlo: {bad}",
               "{courier}: il pacco {trk} è in consegna oggi tra le 14:00 e le 18:00."),
        "nl": ("{courier}: pakket {trk} is vastgehouden, douanekosten van {amt} openstaand. Betaal om vrij te geven: {bad}",
               "{courier}: uw pakket {trk} wordt vandaag tussen 14:00 en 18:00 bezorgd."),
        "ru": ("{courier}: посылка {trk} задержана, таможенный сбор {amt} не оплачен. Оплатите для выдачи: {bad}",
               "{courier}: посылка {trk} будет доставлена сегодня с 14:00 до 18:00."),
        "tr": ("{courier}: {trk} numaralı kargo gümrükte, {amt} tutarında ödeme bekleniyor. Serbest bırakmak için ödeyin: {bad}",
               "{courier}: {trk} numaralı kargonuz bugün 14:00-18:00 arasında teslim edilecek."),
        "id": ("{courier}: paket {trk} tertahan, biaya bea masuk {amt} belum dibayar. Bayar untuk melepaskan: {bad}",
               "{courier}: paket {trk} dalam pengiriman hari ini pukul 14.00-18.00."),
        "ar": ("{courier}: الطرد {trk} محتجز بسبب رسوم جمركية {amt}. ادفع للإفراج عنه: {bad}",
               "{courier}: سيتم تسليم طردك {trk} اليوم بين الساعة 14:00 و18:00."),
    },
    "code_theft": {
        "en": ("{bank}: to stop the transfer we just blocked, reply with the {code} code we sent you.",
               "{bank}: {code} is your login code. We will never ask you to share it."),
        "es": ("{bank}: para detener la transferencia que hemos bloqueado, responda con el código {code} que le enviamos.",
               "{bank}: {code} es su código de acceso. Nunca le pediremos que lo comparta."),
        "de": ("{bank}: um die soeben gestoppte Überweisung zu stornieren, antworten Sie mit dem Code {code}.",
               "{bank}: {code} ist Ihr Anmeldecode. Wir fragen Sie nie nach diesem Code."),
        "fr": ("{bank} : pour annuler le virement que nous avons bloqué, répondez avec le code {code}.",
               "{bank} : {code} est votre code de connexion. Nous ne vous le demanderons jamais."),
        "it": ("{bank}: per bloccare il bonifico appena fermato, rispondi con il codice {code}.",
               "{bank}: {code} è il tuo codice di accesso. Non te lo chiederemo mai."),
        "ru": ("{bank}: чтобы отменить заблокированный перевод, отправьте в ответ код {code}.",
               "{bank}: {code} — ваш код для входа. Мы никогда не просим его сообщить."),
        "id": ("{bank}: untuk membatalkan transfer yang kami blokir, balas dengan kode {code}.",
               "{bank}: {code} adalah kode masuk Anda. Kami tidak pernah meminta kode ini."),
        "ar": ("{bank}: لإيقاف التحويل الذي حجبناه، أرسل الرمز {code} الذي وصلك.",
               "{bank}: {code} هو رمز الدخول الخاص بك. لن نطلب منك مشاركته أبدًا."),
    },
    "subscription": {
        "en": ("{brand}: your plan renews today for {amt} unless you cancel. Stop the charge: {bad}",
               "{brand}: your plan renews on the 28th for {amt}. Manage it in the app or at {good}."),
        "es": ("{brand}: su plan se renueva hoy por {amt} si no lo cancela. Detenga el cobro: {bad}",
               "{brand}: su plan se renovará el día 28 por {amt}. Gestiónelo en la app o en {good}."),
        "de": ("{brand}: Ihr Abo verlängert sich heute für {amt}, wenn Sie nicht kündigen. Abbuchung stoppen: {bad}",
               "{brand}: Ihr Abo verlängert sich am 28. für {amt}. Verwalten Sie es in der App oder auf {good}."),
        "fr": ("{brand} : votre abonnement se renouvelle aujourd'hui pour {amt} sauf annulation. Arrêtez le prélèvement : {bad}",
               "{brand} : votre abonnement se renouvelle le 28 pour {amt}. Gérez-le dans l'application ou sur {good}."),
        "it": ("{brand}: il tuo piano si rinnova oggi a {amt} se non lo annulli. Blocca l'addebito: {bad}",
               "{brand}: il tuo piano si rinnova il 28 a {amt}. Gestiscilo nell'app o su {good}."),
        "ru": ("{brand}: подписка продлится сегодня за {amt}, если не отменить. Остановить списание: {bad}",
               "{brand}: подписка продлится 28-го за {amt}. Управление в приложении или на {good}."),
        "id": ("{brand}: paket Anda diperpanjang hari ini seharga {amt} kecuali dibatalkan. Hentikan tagihan: {bad}",
               "{brand}: paket Anda diperpanjang tanggal 28 seharga {amt}. Kelola di aplikasi atau {good}."),
    },
    "utility_bill": {
        "en": ("{brand}: your electricity will be cut off today over an unpaid bill of {amt}. Settle now: {bad}",
               "{brand}: your electricity bill of {amt} is due on the 28th. Pay in the app or at {good}."),
        "es": ("{brand}: hoy se cortará la luz por una factura impagada de {amt}. Regularice ahora: {bad}",
               "{brand}: su factura de luz de {amt} vence el día 28. Pague en la app o en {good}."),
        "de": ("{brand}: Ihr Strom wird heute wegen einer offenen Rechnung über {amt} abgestellt. Jetzt begleichen: {bad}",
               "{brand}: Ihre Stromrechnung über {amt} ist am 28. fällig. Zahlung in der App oder auf {good}."),
        "fr": ("{brand} : votre électricité sera coupée aujourd'hui pour une facture impayée de {amt}. Régularisez : {bad}",
               "{brand} : votre facture d'électricité de {amt} est due le 28. Payez dans l'application ou sur {good}."),
        "tr": ("{brand}: {amt} tutarındaki ödenmemiş fatura nedeniyle elektriğiniz bugün kesilecek. Şimdi ödeyin: {bad}",
               "{brand}: {amt} tutarındaki elektrik faturanızın son ödeme günü ayın 28'i. Uygulamadan ödeyebilirsiniz."),
        "id": ("{brand}: listrik Anda akan diputus hari ini karena tagihan {amt} belum dibayar. Bayar sekarang: {bad}",
               "{brand}: tagihan listrik {amt} jatuh tempo tanggal 28. Bayar di aplikasi atau {good}."),
        "ar": ("{brand}: سيتم قطع الكهرباء اليوم بسبب فاتورة غير مدفوعة بقيمة {amt}. سدد الآن: {bad}",
               "{brand}: فاتورة الكهرباء بقيمة {amt} تستحق في الـ28. ادفع عبر التطبيق أو {good}."),
    },
    "refund_claim": {
        "en": ("{brand}: a refund of {amt} is waiting for you. Enter your card details to receive it: {bad}",
               "{brand}: your refund of {amt} was sent back to the card ending {l4}. It can take 5 working days."),
        "es": ("{brand}: tiene un reembolso de {amt} pendiente. Introduzca los datos de su tarjeta para recibirlo: {bad}",
               "{brand}: su reembolso de {amt} se ha enviado a la tarjeta terminada en {l4}. Puede tardar 5 días hábiles."),
        "de": ("{brand}: eine Rückerstattung von {amt} wartet auf Sie. Kartendaten eingeben zum Erhalt: {bad}",
               "{brand}: Ihre Rückerstattung über {amt} ging an die Karte {l4}. Die Gutschrift dauert bis zu 5 Werktage."),
        "fr": ("{brand} : un remboursement de {amt} vous attend. Saisissez vos coordonnées bancaires : {bad}",
               "{brand} : votre remboursement de {amt} a été envoyé sur la carte {l4}. Comptez 5 jours ouvrés."),
        "it": ("{brand}: un rimborso di {amt} ti aspetta. Inserisci i dati della carta per riceverlo: {bad}",
               "{brand}: il rimborso di {amt} è stato inviato sulla carta {l4}. Possono servire 5 giorni lavorativi."),
        "ru": ("{brand}: вам положен возврат {amt}. Введите данные карты для получения: {bad}",
               "{brand}: возврат {amt} отправлен на карту {l4}. Зачисление занимает до 5 рабочих дней."),
        "id": ("{brand}: pengembalian dana {amt} menunggu Anda. Masukkan data kartu untuk menerimanya: {bad}",
               "{brand}: dana {amt} telah dikembalikan ke kartu {l4}. Proses hingga 5 hari kerja."),
    },
    "delivery_address": {
        "en": ("{courier}: we could not deliver parcel {trk} because the address is incomplete. Confirm it here: {bad}",
               "{courier}: parcel {trk} could not be delivered. It is waiting at your local depot for 7 days."),
        "es": ("{courier}: no pudimos entregar el paquete {trk} porque la dirección está incompleta. Confírmela aquí: {bad}",
               "{courier}: no pudimos entregar el paquete {trk}. Queda 7 días en su oficina local."),
        "de": ("{courier}: Paket {trk} konnte wegen unvollständiger Adresse nicht zugestellt werden. Hier bestätigen: {bad}",
               "{courier}: Paket {trk} konnte nicht zugestellt werden und liegt 7 Tage in Ihrer Filiale bereit."),
        "fr": ("{courier} : colis {trk} non livré, adresse incomplète. Confirmez-la ici : {bad}",
               "{courier} : colis {trk} non livré. Il reste 7 jours à votre point relais."),
        "nl": ("{courier}: pakket {trk} is niet bezorgd omdat het adres onvolledig is. Bevestig het hier: {bad}",
               "{courier}: pakket {trk} is niet bezorgd en ligt 7 dagen klaar in het depot."),
        "it": ("{courier}: pacco {trk} non consegnato per indirizzo incompleto. Confermalo qui: {bad}",
               "{courier}: pacco {trk} non consegnato. Resta 7 giorni presso il punto di ritiro."),
        "ru": ("{courier}: посылку {trk} не доставили, адрес неполный. Подтвердите его здесь: {bad}",
               "{courier}: посылку {trk} не доставили. Она ждёт вас в отделении 7 дней."),
        "id": ("{courier}: paket {trk} gagal dikirim karena alamat tidak lengkap. Konfirmasi di sini: {bad}",
               "{courier}: paket {trk} gagal dikirim dan menunggu di gerai selama 7 hari."),
    },
}
BRANDS = {"subscription": ["Spotify", "Disney+", "Audible", "Strava", "Duolingo"],
          "utility_bill": ["Iberdrola", "EDF", "Enel", "E.ON", "PLN"],
          "refund_claim": ["Zalando", "AliExpress", "Booking", "Etsy", "Shein"]}


def build_rows(pairs, seed):
    rng = random.Random(seed)
    gen = Gen(rng)
    rows, seen = [], set()
    scenarios = list(SCENARIOS)
    while len(rows) < pairs * 2:
        scenario = rng.choice(scenarios)
        lang = rng.choice(list(SCENARIOS[scenario]))
        lure_t, legit_t = SCENARIOS[scenario][lang]
        brand = rng.choice(BRANDS.get(scenario, BANKS.get(lang, ["Service"])))
        fields = {"bank": rng.choice(BANKS.get(lang, ["Bank"])), "courier": rng.choice(COURIERS.get(lang, ["Courier"])),
                  "brand": brand, "amt": gen.amount(lang), "l4": gen.last4(), "trk": gen.tracking(),
                  "code": gen.code(), "phone": gen.phone()}
        anchor = fields["bank"] if "{bank}" in lure_t else fields["courier"] if "{courier}" in lure_t else brand
        fields["bad"] = gen.bad_url(anchor)
        fields["good"] = gen.good_url(anchor, lang)
        lure, legit = lure_t.format(**fields), legit_t.format(**fields)
        if lure in seen or legit in seen:
            continue
        seen.add(lure)
        seen.add(legit)
        rows.append(("notice_pair_lure", lure, "phishing"))
        rows.append(("notice_pair_legit", legit, "ham"))
    return rows


def eval_keys():
    keys = set()
    for path in EVAL_FILES:
        if path.exists():
            with open(path, newline="", encoding="utf-8", errors="replace") as f:
                for row in csv.DictReader(f):
                    keys.add(frozenset(re.findall(r"\w+", (row.get("text") or "").lower())))
    return keys


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", type=int, default=500)
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()

    rows = build_rows(args.pairs, args.seed)
    evals = eval_keys()
    kept, dropped = [], 0
    for pool, text, label in rows:
        words = frozenset(re.findall(r"\w+", text.lower()))
        if any(len(words & e) / max(1, len(words | e)) >= 0.5 for e in evals):
            dropped += 1
            continue
        kept.append((pool, text, label))
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "pool", "text", "label"])
        for i, (pool, text, label) in enumerate(kept):
            w.writerow([f"n{i:05d}", pool, text, label])
    print(f"wrote {OUT_CSV.relative_to(REPO_ROOT)}: {len(kept)} rows "
          f"({sum(1 for p, _, _ in kept if p == 'notice_pair_lure')} lures, "
          f"{sum(1 for p, _, _ in kept if p == 'notice_pair_legit')} legitimate), "
          f"{dropped} dropped for looking like an eval message")


if __name__ == "__main__":
    main()
