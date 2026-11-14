# Generated data for model 2.12

9,698 SMS rows were written in-house for the weak spots measured on model 2.9,
one file per theme, in 27 languages, in two rounds (7,276 rows, then 2,422
after the first retraining showed which gaps were still open). Every row was written to an intended
label, then re-graded blind against `docs/LABELING_GUIDE.md` by a second pass
that saw only the text (`evals/curate_synthetic.py`). A row is kept only when
both passes agree and it duplicates neither an existing training row nor any
eval set. The 151 rows that were dropped, with both labels and the reason, are
in `synthetic_v2.12_dropped.csv`.

| | Rows |
|---|---|
| Generated | 9,698 |
| Dropped: second pass disagreed | 60 (phishing→spam 46, phishing→ham 10, phishing→unsure 3, ham→phishing 1) |
| Dropped: duplicate of an existing training row | 6 |
| Dropped: too close to an eval set | 84 |
| Dropped: not graded | 1 |
| **Kept** | **9,547** (phishing 5,033 / ham 3,611 / spam 903) |

## What each file covers

Round 1:

| File | Kept | Why it exists |
|---|---|---|
| `legit_en`, `legit_west_europe`, `legit_ar_tr_ru_pl_el`, `legit_east_asia`, `legit_south_asia_africa` | 1,802 ham | Real branded notices on the brand's real domain or with no link, in 21 languages, many with an amount and a payment link. Model 2.9 blocked 24% of such notices in the probe and 4 of 8 in the live smoke test. |
| `lures_latin`, `lures_nonlatin` | 734 phishing | The minimal-pair twins of the notices above: lookalike or shortener domain, invented problem, credential or payment ask. Non-Latin scripts were the weakest languages on IMC25. |
| `linkfree_en`, `linkfree_multi` | 722 phishing | Lures with no URL: family new-number, code relay, callback vishing, KYC/SIM-block, boss gift cards, job-task openers. The template-free shapes 2.9 passed most often. |
| `wrong_number` | 206 phishing / 149 ham | Pig-butchering openers with a hook against genuine misdials, written to share vocabulary. 2.9 passed 97% of IMC25's wrong-number attacks. |
| `conversational_en`, `conversational_multi` | 743 ham | Person-to-person chat, most of it about money, codes, cards or links in an innocent way. 2.9 blocked "can I get a dollar loan from you" as phishing. |
| `codes_alerts` | 224 ham / 108 phishing | Delivered codes and security alerts (ham) against messages that ask for the code (phishing). |
| `prize_premium` | 329 phishing | Prize, compensation and premium-rate callback lures, which 2.9 passed more often than 2.7 did on UCI and Mishra. |
| `government_toll`, `fintech_crypto`, `delivery`, `jobs` | 875 phishing / 396 ham / 122 spam | Themed pairs: real notices and real recruiter follow-ups against impersonations, with job ads as spam. |
| `spam_en`, `spam_multi` | 722 spam | Modern advertising that impersonates nobody and promises no prize, so the spam class keeps a boundary with phishing. |

Round 2, after the round-1 model fell behind 2.9 on long messages whose only
link is a URL shortener (worst in banking, Dutch and Spanish) and on calm,
routine-sounding lures:

| File | Kept | Why it exists |
|---|---|---|
| `shortlink_bank_es_nl` | 320 phishing | Bank impersonation in Spanish and Dutch with a shortener or opaque link. |
| `shortlink_services_en`, `shortlink_services_multi`, `shortlink_other_langs` | 941 phishing | Courier, carrier, utility, government, streaming and tax impersonation with shortener links, in English and about 20 other languages; the other-languages file also has bank lures. |
| `callback_kyc_utility` | 299 phishing | No-link lures that ask for a call back: KYC expiry, SIM block, electricity or gas disconnection. |
| `low_key_notices` | 280 phishing | Lures written like a routine notice: no capitals, no urgency words, one quiet ask. |
| `prize_callback` | 219 phishing / 59 spam | "You have won" and premium-rate callback messages (phishing) against plain competition ads (spam). |
| `ham_links` | 297 ham | Real notices and personal messages that carry a link, shorteners included, so a link alone does not mean phishing. |

Two planned round-2 files were not produced and are not in the data: bank
lures in English, and bank lures in French, Italian, German and Portuguese.

Languages kept: en 4,016, es 650, nl 389, fr 340, de 308, pt 298, hi 292,
id 283, ja 266, it 264, zh 263, ar 261, tl 248, tr 231, ko 212, ru 198, th 183,
pl 175, el 161, ms 144, sw 131, vi 125, ta 74, cs 10, sv 10, da 10, ur 5.

All brands' real domains are used only in legitimate notices; every phishing
domain, phone number, amount and reference is invented. No client traffic was
used. Files are `text,label,category,language`.
