#!/usr/bin/env python3
"""
build_training.py — gap-fix TRAINING set for OpenTextShield (false-positive repair).

WHAT THIS IS
------------
Realistic, branded A2P messages to teach the model that legitimate transactional
SMS (government, delivery, bank, appointment, telecom, otp) is HAM — paired with
hard-negative SCAM twins so the fix does NOT blind the model to real attacks.
Built as a generator (templates + varied slots) for volume and reproducibility,
mirroring the project's own generate_synthetic.py approach.

WHY THESE CATEGORIES / WEIGHTS
------------------------------
Baseline FP rate on model 2.7 (88-msg benchmark): government 94%, delivery 92%,
appointment/telecom 75%, bank 67%, otp 50%; personal 0%. Root cause in v2.4 data:
transactional ham is thin (government 174 vs 2,208 spam; appointment 11) AND too
short/templated to generalise. So this set is gap-weighted and deliberately
realistic/branded/verbose.

*** SEPARATION FROM THE BENCHMARK ***
This file auto-loads ham_benchmark_v1.csv (if present) and removes any exact-text
collisions, so the train/test split stays leakage-free.

PROVENANCE
----------
Constructed-realistic (standard A2P templates + real per-locale institution names,
fictional URLs/refs/amounts). Not scraped. es/it/pt author-verified Romance. Scam
URLs are fictional dedicated domains, never real shorteners.

OUTPUT: ham_training_v1.csv  (text,label,category,language)  label in {ham,phishing}
"""
import csv, os, random

random.seed(42)
LANGS = ["en", "es", "it", "pt"]

# ---------- slot pools ----------
AMOUNTS = ["9,99", "12,40", "19,90", "29,90", "36,90", "42,50", "61,20", "85,00", "129,99", "250,00", "1,99"]
def money(lang):
    a = random.choice(AMOUNTS)
    return "£" + a.replace(",", ".") if lang == "en" else a + "\u20ac"
def code():  return str(random.randint(100000, 999999))
def last4(): return random.choice(["4471", "7782", "2231", "6650", "1180", "9043", "3517"])
def ref():   return random.choice(["RM", "DT", "ZX", "AB", "PT", "CN"]) + str(random.randint(1000000, 9999999))
def name():  return random.choice(["J. SMITH", "A. GARCIA", "M. ROSSI", "P. SILVA", "L. MARTIN", "C. FERREIRA"])
def slug():  return str(random.randint(1000, 99999))

TIME = ["09:30", "10:15", "11:45", "14:00", "16:30", "08:50"]
DATE = {
    "en": ["31 Jan", "1 March", "30 June", "14 May", "5 April"],
    "es": ["31 de enero", "1 de marzo", "30 de junio", "14 de mayo", "5 de abril"],
    "it": ["31 gennaio", "1 marzo", "30 giugno", "14 maggio", "5 aprile"],
    "pt": ["31 de janeiro", "1 de março", "30 de junho", "14 de maio", "5 de abril"],
}
# scam URLs: fictional dedicated domains only (never real shorteners)
SCAM_HOST = ["secure-verify", "account-update", "gov-refund", "redelivery-fee",
             "pay-now-secure", "id-confirm", "billing-portal", "parcel-reschedule"]
def scam_url():
    return f"http://{random.choice(SCAM_HOST)}-{slug()}.{random.choice(['top','xyz','live','vip'])}/login"

BRANDS = {
 "bank_alert": {"en":["Barclays","Lloyds","NatWest","Halifax","Nationwide"],
                "es":["Santander","BBVA","CaixaBank","Bankinter","Sabadell"],
                "it":["Intesa Sanpaolo","UniCredit","BPER","Banco BPM","Poste Italiane"],
                "pt":["Caixa Geral de Dep\u00f3sitos","Millennium BCP","Novo Banco","Santander Totta","BPI"]},
 "delivery":   {"en":["Royal Mail","DPD","Evri","UPS","Yodel"],
                "es":["Correos","SEUR","MRW","GLS","Amazon"],
                "it":["Poste Italiane","BRT","GLS","SDA","Amazon"],
                "pt":["CTT","DPD","GLS","DHL","Amazon"]},
 "government": {"en":["HMRC","DVLA","DWP","TV Licensing","HM Passport Office"],
                "es":["Agencia Tributaria","DGT","Seguridad Social","SEPE"],
                "it":["Agenzia delle Entrate","INPS","Motorizzazione","Comune"],
                "pt":["Autoridade Tribut\u00e1ria","Seguran\u00e7a Social","IMT","SNS"]},
 "telecom":    {"en":["Vodafone","EE","O2","Three","Sky Mobile"],
                "es":["Movistar","Orange","Vodafone","Yoigo"],
                "it":["TIM","Vodafone","WindTre","Iliad"],
                "pt":["MEO","NOS","Vodafone"]},
 "appointment":{"en":["Greenway Surgery","Specsavers","Boots Opticians","City Dental"],
                "es":["tu centro de salud","Cl\u00ednica Dental Sonrisa","Sanitas"],
                "it":["il poliambulatorio","Studio Dentistico Sorriso","ASL"],
                "pt":["o centro de sa\u00fade","Cl\u00ednica Dent\u00e1ria Sorriso","SNS"]},
 "otp":        {"en":["Google","Microsoft","PayPal","Amazon"],
                "es":["Google","Microsoft","PayPal","Amazon"],
                "it":["Google","Microsoft","PayPal","Amazon"],
                "pt":["Google","Microsoft","PayPal","Amazon"]},
}

GOV_THING = {
 "en":["Self Assessment","vehicle tax","passport application","benefit review","TV Licence"],
 "es":["la declaraci\u00f3n de la renta","el impuesto de circulaci\u00f3n","la renovaci\u00f3n del DNI","tu prestaci\u00f3n"],
 "it":["la dichiarazione dei redditi","il bollo auto","il rinnovo della carta d'identit\u00e0","la tua prestazione"],
 "pt":["a entrega do IRS","o imposto do ve\u00edculo","a renova\u00e7\u00e3o do Cart\u00e3o de Cidad\u00e3o","a tua presta\u00e7\u00e3o"],
}
GOV_URL = {"en":"gov.uk","es":"sede.administracion.gob.es","it":"agenziaentrate.gov.it","pt":"portaldasfinancas.gov.pt"}

# ---------- templates: {cat: {lang: {"ham":[...], "scam":[...]}}} ----------
T = {
 "government": {
  "en":{"ham":[
        "{brand}: Your {gov} deadline is {date}. Sign in at {gov_url} to complete it. We never ask for payment by text.",
        "{brand}: An update on {gov} (ref {ref}) is available in your online account at {gov_url}. No payment is needed.",
        "{brand}: Your {gov} has been received and is being processed. Track progress at {gov_url}."],
       "scam":[
        "{brand}: You have an outstanding {gov} payment of {amt}. Pay within 24h to avoid penalty: {surl}",
        "{brand} FINAL NOTICE: your {gov} is overdue. Confirm your card details now to avoid legal action: {surl}"]},
  "es":{"ham":[
        "{brand}: El plazo para {gov} finaliza el {date}. Accede en {gov_url}. Nunca pedimos datos bancarios por SMS.",
        "{brand}: Hay novedades sobre {gov} (ref {ref}) en tu \u00e1rea personal en {gov_url}. No es necesario ning\u00fan pago.",
        "{brand}: Tu tr\u00e1mite de {gov} se ha registrado y est\u00e1 en proceso. Consulta el estado en {gov_url}."],
       "scam":[
        "{brand}: Tienes un pago pendiente de {gov} de {amt}. Paga en 24h para evitar recargo: {surl}",
        "{brand} \u00daLTIMO AVISO: {gov} est\u00e1 vencido. Confirma los datos de tu tarjeta ahora: {surl}"]},
  "it":{"ham":[
        "{brand}: La scadenza per {gov} \u00e8 il {date}. Accedi su {gov_url}. Non chiediamo mai dati bancari via SMS.",
        "{brand}: Aggiornamento su {gov} (rif {ref}) disponibile nella tua area personale su {gov_url}. Nessun pagamento richiesto.",
        "{brand}: La tua pratica di {gov} \u00e8 stata registrata ed \u00e8 in lavorazione. Stato su {gov_url}."],
       "scam":[
        "{brand}: Hai un pagamento in sospeso di {gov} di {amt}. Paga entro 24h per evitare la sanzione: {surl}",
        "{brand} ULTIMO AVVISO: {gov} \u00e8 scaduto. Conferma subito i dati della carta: {surl}"]},
  "pt":{"ham":[
        "{brand}: O prazo para {gov} termina a {date}. Acede em {gov_url}. Nunca pedimos dados banc\u00e1rios por SMS.",
        "{brand}: H\u00e1 novidades sobre {gov} (ref {ref}) na tua \u00e1rea pessoal em {gov_url}. N\u00e3o \u00e9 necess\u00e1rio qualquer pagamento.",
        "{brand}: O teu pedido de {gov} foi registado e est\u00e1 em processamento. Estado em {gov_url}."],
       "scam":[
        "{brand}: Tens um pagamento pendente de {gov} de {amt}. Paga em 24h para evitar penaliza\u00e7\u00e3o: {surl}",
        "{brand} \u00daLTIMO AVISO: {gov} est\u00e1 em atraso. Confirma os dados do teu cart\u00e3o agora: {surl}"]},
 },
 "delivery": {
  "en":{"ham":[
        "{brand}: Your parcel {ref} will be delivered today between {time} and {time2}. Track it in the {brand} app.",
        "{brand}: We tried to deliver {ref} but no one was home. Collect from your local depot or rebook online. No fee.",
        "{brand}: Your order has been dispatched and arrives tomorrow. No action needed."],
       "scam":[
        "{brand}: Your parcel {ref} is held. A {amt} customs fee is required to release it: {surl}",
        "{brand}: Delivery failed. Pay the {amt} redelivery fee within 24h or your parcel returns: {surl}"]},
  "es":{"ham":[
        "{brand}: Tu paquete {ref} se entregar\u00e1 hoy entre las {time} y las {time2}. S\u00edguelo en la app de {brand}.",
        "{brand}: Intentamos entregar {ref} pero no hab\u00eda nadie. Rec\u00f3gelo en tu oficina o concierta otra entrega. Sin coste.",
        "{brand}: Tu pedido ha sido enviado y llega ma\u00f1ana. No es necesario hacer nada."],
       "scam":[
        "{brand}: Tu paquete {ref} est\u00e1 retenido. Se requiere una tasa de aduana de {amt} para liberarlo: {surl}",
        "{brand}: Entrega fallida. Paga la tasa de reenv\u00edo de {amt} en 24h o se devolver\u00e1: {surl}"]},
  "it":{"ham":[
        "{brand}: Il tuo pacco {ref} sar\u00e0 consegnato oggi tra le {time} e le {time2}. Segui la spedizione nell'app {brand}.",
        "{brand}: Abbiamo tentato la consegna di {ref} ma non c'era nessuno. Ritira in filiale o riprenota online. Nessun costo.",
        "{brand}: Il tuo ordine \u00e8 stato spedito e arriva domani. Nessuna azione necessaria."],
       "scam":[
        "{brand}: Il tuo pacco {ref} \u00e8 in giacenza. \u00c8 richiesta una tassa doganale di {amt} per liberarlo: {surl}",
        "{brand}: Consegna fallita. Paga la tariffa di riconsegna di {amt} entro 24h o sar\u00e0 reso: {surl}"]},
  "pt":{"ham":[
        "{brand}: A tua encomenda {ref} ser\u00e1 entregue hoje entre as {time} e as {time2}. Segue na app dos {brand}.",
        "{brand}: Tent\u00e1mos entregar {ref} mas n\u00e3o estava ningu\u00e9m. Levanta no ponto ou remarca online. Sem custo.",
        "{brand}: O teu pedido foi enviado e chega amanh\u00e3. N\u00e3o \u00e9 necess\u00e1ria qualquer a\u00e7\u00e3o."],
       "scam":[
        "{brand}: A tua encomenda {ref} est\u00e1 retida. \u00c9 necess\u00e1ria uma taxa alfandeg\u00e1ria de {amt} para a libertar: {surl}",
        "{brand}: Entrega falhada. Paga a taxa de reenvio de {amt} em 24h ou ser\u00e1 devolvida: {surl}"]},
 },
 "bank_alert": {
  "en":{"ham":[
        "{brand}: A payment of {amt} to a card ending {last4} was made today. Not you? Call the number on the back of your card.",
        "{brand}: You received {amt} from {name}. Your statement is available in the app.",
        "{brand}: Your monthly statement for the account ending {last4} is ready to view in the app."],
       "scam":[
        "{brand}: Your account ending {last4} is locked. Verify your identity now to restore access: {surl}",
        "{brand} ALERT: a payment of {amt} is pending. If you don't recognise it, cancel here: {surl}"]},
  "es":{"ham":[
        "{brand}: Pago de {amt} con la tarjeta acabada en {last4} hoy. Si no fuiste t\u00fa, llama al n\u00famero de tu tarjeta.",
        "{brand}: Has recibido {amt} de {name}. Tu extracto est\u00e1 disponible en la app.",
        "{brand}: Tu extracto mensual de la cuenta acabada en {last4} ya est\u00e1 disponible en la app."],
       "scam":[
        "{brand}: Tu cuenta acabada en {last4} est\u00e1 bloqueada. Verifica tu identidad ahora para recuperar el acceso: {surl}",
        "{brand} AVISO: hay un pago de {amt} pendiente. Si no lo reconoces, canc\u00e9lalo aqu\u00ed: {surl}"]},
  "it":{"ham":[
        "{brand}: Pagamento di {amt} con la carta che termina con {last4} oggi. Se non sei stato tu, chiama il numero della carta.",
        "{brand}: Hai ricevuto {amt} da {name}. L'estratto \u00e8 disponibile nell'app.",
        "{brand}: Il tuo estratto conto mensile del conto che termina con {last4} \u00e8 disponibile nell'app."],
       "scam":[
        "{brand}: Il tuo conto che termina con {last4} \u00e8 bloccato. Verifica subito la tua identit\u00e0 per ripristinare l'accesso: {surl}",
        "{brand} AVVISO: un pagamento di {amt} \u00e8 in sospeso. Se non lo riconosci, annullalo qui: {surl}"]},
  "pt":{"ham":[
        "{brand}: Pagamento de {amt} com o cart\u00e3o terminado em {last4} hoje. Se n\u00e3o foste tu, liga para o n\u00famero do cart\u00e3o.",
        "{brand}: Recebeste {amt} de {name}. O teu extrato est\u00e1 dispon\u00edvel na app.",
        "{brand}: O teu extrato mensal da conta terminada em {last4} j\u00e1 est\u00e1 dispon\u00edvel na app."],
       "scam":[
        "{brand}: A tua conta terminada em {last4} est\u00e1 bloqueada. Verifica a tua identidade agora para recuperar o acesso: {surl}",
        "{brand} AVISO: um pagamento de {amt} est\u00e1 pendente. Se n\u00e3o o reconheces, cancela aqui: {surl}"]},
 },
 "appointment": {
  "en":{"ham":[
        "Reminder: your appointment at {brand} is on {date} at {time}. Reply C to confirm or call to reschedule.",
        "{brand}: your check-up is booked for {date} at {time}. Reply 1 to confirm, 2 to cancel."],
       "scam":[
        "{brand}: confirm your appointment by paying a {amt} booking fee here within 12h or it will be cancelled: {surl}"]},
  "es":{"ham":[
        "Recordatorio: tu cita en {brand} es el {date} a las {time}. Responde C para confirmar o llama para cambiarla.",
        "{brand}: tu revisi\u00f3n est\u00e1 fijada para el {date} a las {time}. Responde 1 para confirmar, 2 para cancelar."],
       "scam":[
        "{brand}: confirma tu cita pagando una tasa de reserva de {amt} aqu\u00ed en 12h o se cancelar\u00e1: {surl}"]},
  "it":{"ham":[
        "Promemoria: il tuo appuntamento presso {brand} \u00e8 il {date} alle {time}. Rispondi C per confermare o chiama per spostarlo.",
        "{brand}: la tua visita \u00e8 fissata per il {date} alle {time}. Rispondi 1 per confermare, 2 per annullare."],
       "scam":[
        "{brand}: conferma il tuo appuntamento pagando una tassa di prenotazione di {amt} qui entro 12h o sar\u00e0 annullato: {surl}"]},
  "pt":{"ham":[
        "Lembrete: a tua consulta em {brand} \u00e9 no dia {date} \u00e0s {time}. Responde C para confirmar ou liga para remarcar.",
        "{brand}: a tua consulta est\u00e1 marcada para {date} \u00e0s {time}. Responde 1 para confirmar, 2 para cancelar."],
       "scam":[
        "{brand}: confirma a tua consulta pagando uma taxa de reserva de {amt} aqui em 12h ou ser\u00e1 cancelada: {surl}"]},
 },
 "telecom": {
  "en":{"ham":[
        "{brand}: You've used 80% of your monthly data. Top up or manage your plan in the {brand} app.",
        "{brand}: Your bill of {amt} is ready and will be taken on the 3rd. View it in the {brand} app."],
       "scam":[
        "{brand}: Your account will be suspended today. Verify your details now to keep your number: {surl}",
        "{brand}: You've earned a {amt} loyalty reward. Claim it within 24h here: {surl}"]},
  "es":{"ham":[
        "{brand}: Has consumido el 80% de tus datos del mes. Recarga o gestiona tu tarifa en la app de {brand}.",
        "{brand}: Tu factura de {amt} est\u00e1 disponible y se cobrar\u00e1 el d\u00eda 3. Consúltala en la app de {brand}."],
       "scam":[
        "{brand}: Tu cuenta ser\u00e1 suspendida hoy. Verifica tus datos ahora para mantener tu n\u00famero: {surl}",
        "{brand}: Has ganado un premio de fidelidad de {amt}. Recl\u00e1malo en 24h aqu\u00ed: {surl}"]},
  "it":{"ham":[
        "{brand}: Hai usato l'80% dei tuoi giga del mese. Ricarica o gestisci l'offerta nell'app {brand}.",
        "{brand}: La tua fattura di {amt} \u00e8 disponibile e sar\u00e0 addebitata il giorno 3. Consultala nell'app {brand}."],
       "scam":[
        "{brand}: Il tuo account sar\u00e0 sospeso oggi. Verifica subito i tuoi dati per mantenere il numero: {surl}",
        "{brand}: Hai vinto un premio fedelt\u00e0 di {amt}. Riscuotilo entro 24h qui: {surl}"]},
  "pt":{"ham":[
        "{brand}: J\u00e1 usaste 80% dos teus dados do m\u00eas. Carrega ou gere o teu tarif\u00e1rio na app {brand}.",
        "{brand}: A tua fatura de {amt} est\u00e1 dispon\u00edvel e ser\u00e1 debitada no dia 3. Consulta na app {brand}."],
       "scam":[
        "{brand}: A tua conta ser\u00e1 suspensa hoje. Verifica os teus dados agora para manter o n\u00famero: {surl}",
        "{brand}: Ganhaste um pr\u00e9mio de fidelidade de {amt}. Resgata em 24h aqui: {surl}"]},
 },
 "otp": {
  "en":{"ham":[
        "{brand}: {code} is your verification code. It expires in 10 minutes. We'll never ask for it by phone or email.",
        "Your {brand} security code is {code}. Do not share it with anyone."],
       "scam":[
        "{brand}: someone tried to access your account. Reply with the code we just sent to stop it: {code}",
        "{brand}: to verify your account, share the {code} code our agent will request by call."]},
  "es":{"ham":[
        "{brand}: {code} es tu c\u00f3digo de verificaci\u00f3n. Caduca en 10 minutos. Nunca te lo pediremos por tel\u00e9fono ni email.",
        "Tu c\u00f3digo de seguridad de {brand} es {code}. No lo compartas con nadie."],
       "scam":[
        "{brand}: alguien intent\u00f3 acceder a tu cuenta. Responde con el c\u00f3digo que te enviamos para detenerlo: {code}",
        "{brand}: para verificar tu cuenta, comparte el c\u00f3digo {code} que nuestro agente te pedir\u00e1 por llamada."]},
  "it":{"ham":[
        "{brand}: {code} \u00e8 il tuo codice di verifica. Scade tra 10 minuti. Non te lo chiederemo mai per telefono o email.",
        "Il tuo codice di sicurezza {brand} \u00e8 {code}. Non condividerlo con nessuno."],
       "scam":[
        "{brand}: qualcuno ha tentato di accedere al tuo account. Rispondi con il codice appena inviato per bloccarlo: {code}",
        "{brand}: per verificare il tuo account, comunica il codice {code} che il nostro operatore ti chieder\u00e0 per telefono."]},
  "pt":{"ham":[
        "{brand}: {code} \u00e9 o teu c\u00f3digo de verifica\u00e7\u00e3o. Expira em 10 minutos. Nunca o pediremos por telefone ou email.",
        "O teu c\u00f3digo de seguran\u00e7a {brand} \u00e9 {code}. N\u00e3o o partilhes com ningu\u00e9m."],
       "scam":[
        "{brand}: algu\u00e9m tentou aceder \u00e0 tua conta. Responde com o c\u00f3digo que enviámos para o bloquear: {code}",
        "{brand}: para verificar a tua conta, partilha o c\u00f3digo {code} que o nosso agente pedir\u00e1 por chamada."]},
 },
}

# how many variations per template, weighted by how bad the gap is
WEIGHT = {"government": 6, "delivery": 6, "appointment": 5, "telecom": 4, "bank_alert": 4, "otp": 3}
SCAM_LABEL = "phishing"   # transactional impersonation scams are phishing-class

def fill(tmpl, lang, cat):
    t1, t2 = sorted(random.sample(TIME, 2))   # ordered window: t1 < t2
    return (tmpl
        .replace("{brand}", random.choice(BRANDS[cat][lang]))
        .replace("{amt}", money(lang))
        .replace("{code}", code())
        .replace("{last4}", last4())
        .replace("{ref}", ref())
        .replace("{name}", name())
        .replace("{date}", random.choice(DATE[lang]))
        .replace("{time2}", t2)
        .replace("{time}", t1)
        .replace("{gov}", random.choice(GOV_THING.get(lang, [""])))
        .replace("{gov_url}", GOV_URL.get(lang, ""))
        .replace("{surl}", scam_url()))

def generate():
    rows, seen = [], set()
    for cat, langs in T.items():
        k = WEIGHT[cat]
        for lang, kinds in langs.items():
            for kind, tmpls in kinds.items():
                label = "ham" if kind == "ham" else SCAM_LABEL
                for tmpl in tmpls:
                    made = 0; tries = 0
                    while made < k and tries < k * 8:
                        tries += 1
                        text = fill(tmpl, lang, cat)
                        if text in seen:
                            continue
                        seen.add(text)
                        rows.append((text, label, cat, lang))
                        made += 1
    return rows

def drop_benchmark_overlap(rows):
    bench = set()
    for p in ["ham_benchmark_v1.csv", os.path.join("..", "ham_benchmark_v1.csv")]:
        if os.path.exists(p):
            for r in csv.DictReader(open(p, encoding="utf-8")):
                bench.add(r["text"].strip())
    if not bench:
        print("note: ham_benchmark_v1.csv not found next to this script — skipping leakage check.")
        return rows
    kept = [r for r in rows if r[0].strip() not in bench]
    print(f"leakage check: removed {len(rows)-len(kept)} rows overlapping the benchmark")
    return kept

def main():
    rows = drop_benchmark_overlap(generate())
    out = "ham_training_v1.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["text", "label", "category", "language"])
        for r in rows:
            w.writerow(r)
    from collections import Counter
    print(f"wrote {out}: {len(rows)} rows")
    print("by label    :", dict(Counter(r[1] for r in rows)))
    print("by category :", dict(Counter(r[2] for r in rows)))
    print("by language :", dict(Counter(r[3] for r in rows)))

if __name__ == "__main__":
    main()