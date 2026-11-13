# Generated data for model 2.12

7,276 SMS rows were written in-house for the weak spots measured on model 2.9,
one file per theme, in 23 languages. Every row was written to an intended
label, then re-graded blind against `docs/LABELING_GUIDE.md` by a second pass
that saw only the text (`evals/curate_synthetic.py`). A row is kept only when
both passes agree and it duplicates neither an existing training row nor any
eval set. The 144 rows that were dropped, with both labels and the reason, are
in `synthetic_v2.12_dropped.csv`.

| | Rows |
|---|---|
| Generated | 7,276 |
| Dropped: second pass disagreed | 58 (phishing→spam 45, phishing→ham 10, phishing→unsure 2, ham→phishing 1) |
| Dropped: duplicate of an existing training row | 6 |
| Dropped: too close to an eval set | 80 |
| **Kept** | **7,132** (ham 3,314 / phishing 2,974 / spam 844) |

## What each file covers

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

Languages kept: en 2,979, es 245, pt 239, de 238, ar 236, fr 235, zh 233,
hi 232, id 228, ja 226, tr 216, ko 212, tl 209, ru 198, it 184, pl 155, nl 149,
th 143, el 141, sw 131, vi 125, ms 104, ta 74.

All brands' real domains are used only in legitimate notices; every phishing
domain, phone number, amount and reference is invented. No client traffic was
used. Files are `text,label,category,language`.
