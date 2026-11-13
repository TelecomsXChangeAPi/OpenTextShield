# OTS probe report

Generated: 2026-09-21 21:50 UTC. Messages invented daily by claude-haiku-4-5, classified by the live API, graded against `docs/LABELING_GUIDE.md` by claude-opus-5. 200 usable messages over 1 day(s); model version(s) seen: 2.9.

## Overall

| Messages | Accuracy | Block accuracy | False block rate | Missed threat rate | Median latency |
|---|---|---|---|---|---|
| 200 | 86.0% | 89.0% | 24.1% | 2.5% | 179 ms |

## By day

| Day | Messages | Accuracy | Block accuracy | False blocks | Missed threats |
|---|---|---|---|---|---|
| 2026-09-21 | 200 | 86.0% | 89.0% | 24.1% | 2.5% |

## Where it fails

### By error type

| Key | Misses | Of | Miss rate |
|---|---|---|---|
| false_block | 19 | 19 | 100.0% |
| category_swap | 6 | 6 | 100.0% |
| missed_threat | 3 | 3 | 100.0% |

### By category

| Key | Misses | Of | Miss rate |
|---|---|---|---|
| utility_bill | 3 | 3 | 100.0% |
| subscription_renewal | 3 | 3 | 100.0% |
| ecommerce_transaction_check | 1 | 1 | 100.0% |
| gym_membership_billing | 1 | 1 | 100.0% |
| investment_scam | 1 | 2 | 50.0% |
| password_theft | 1 | 1 | 100.0% |
| app_trial_notice | 1 | 1 | 100.0% |
| work_from_home_ad | 1 | 6 | 16.7% |
| service_reminder | 1 | 1 | 100.0% |
| family_impersonation | 1 | 2 | 50.0% |
| utility_account | 1 | 1 | 100.0% |
| gadget_giveaway | 1 | 1 | 100.0% |

### By language

| Key | Misses | Of | Miss rate |
|---|---|---|---|
| en | 18 | 125 | 14.4% |
| ja | 3 | 7 | 42.9% |
| tr | 2 | 5 | 40.0% |
| ru | 2 | 9 | 22.2% |
| ar | 1 | 8 | 12.5% |
| es | 1 | 6 | 16.7% |
| ta | 1 | 2 | 50.0% |

## What the next training round needs

Grader notes, grouped by error type, most recent first. Severity 3 means high-confidence mistakes.

### false block (19)

- sev3 gold=ham ots=phishing@0.959 [ja] "あなたのAmazonアカウントへようこそ。この取引を確認してください：¥45,200の購入が1月15日に行われました。返信YES/NO。" → Japanese legitimate transaction-confirmation alerts (brand name + amount + reply YES/NO, no link and no sensitive data requested) must be ham, like the Chase fraud-alert pattern.
- sev3 gold=ham ots=phishing@0.9505 [ja] "来月のジム会員費1,980円の引き落とし予定です。FitNext会員ページで確認できます。" → Routine Japanese subscription/direct-debit billing notices with no link or urgency are ham; the model over-weights currency amounts plus account-page mentions in non-English text.
- sev3 gold=ham ots=phishing@0.9938 [en] "Your Vodafone bill of £42.99 is due on 22 Jan. Pay securely: myvodafone.co.uk/billing or reply PAY" → Routine billing reminders with a normal due date and the carrier's real domain (myvodafone.co.uk) must not be treated as phishing just because they mention payment.
- sev3 gold=ham ots=phishing@0.9987 [tr] "Turkcell: Faturanız 89 TL ödenmek için bekliyor. Öde: myturkcell.com.tr/odemeler" → A routine bill notice on the carrier's genuine domain (myturkcell.com.tr) is ham; the model treats any billing+link message as phishing instead of checking domain legitimacy and absence of invented urgency.
- sev3 gold=ham ots=phishing@0.9958 [en] "Netflix subscription £9.99 renews on 20 Jan. Change payment method: netflix.com/account/billing" → Subscription renewal notice linking to the real netflix.com domain with no threat or credential request is ham; needs brand-domain whitelisting.
- sev3 gold=ham ots=phishing@0.9991 [en] "Verizon: Your bill of $156.43 is now available. View and pay securely: myvzw.com/billpay" → Legitimate bill-available notice on myvzw.com is ham; model over-weights '$ amount + pay link' without checking the official domain.
- sev3 gold=ham ots=phishing@0.9952 [en] "National Lottery: Check your ticket numbers at national-lottery.org.uk - Play responsibly" → Lottery-branded message on the official national-lottery.org.uk domain that makes no win claim is ham; 'lottery' keyword alone should not trigger phishing.
- sev3 gold=ham ots=phishing@0.994 [en] "AT&T: Your prepaid balance is $12.43. Recharge at: att.com/recharge or dial *611 for assistance." → Prepaid balance notice on att.com with the real *611 helpline is ham; needs official-domain recognition for carrier account notices.
- sev3 gold=ham ots=phishing@0.9277 [en] "Tinder match alert: Maria liked you! See who else swiped right: tinder.com/app/likes" → App engagement notifications on the genuine first-party domain (tinder.com) are real service notices, not lures; treat legitimate brand domains as a strong ham signal.
- sev3 gold=ham ots=phishing@0.9674 [en] "Netflix: Your password was changed on Jan 26. If this wasn't you, reset it at netflix.com/security immediately." → Security notices pointing to the real netflix.com domain with no requested credentials are ham; the model over-weights words like 'password changed' and 'immediately'.
- sev3 gold=ham ots=phishing@0.9762 [en] "Your subscription to Premium will renew on Feb 15 for $9.99. Manage at: spotify.com/account/subscription" → Subscription renewal notices on the brand's own domain (spotify.com) with no urgency or credential request are legitimate service notices, not phishing.
- sev3 gold=ham ots=phishing@0.9957 [ta] "நீங்கள் மொபைல் பாக்கி கட்டும் நேரம் வந்துவிட்டது. தொகை: ₹1,250. பணம் செலுத்துக: airtel.com/pay" → Tamil-language billing reminders pointing to the carrier's genuine domain (airtel.com) are ham; the model treats any non-English payment request as phishing.
- sev3 gold=ham ots=phishing@0.9678 [en] "Wells Fargo: We detected a sign-in attempt from a new device. If this was you, no action needed. If not, change your pas" → Legitimate bank sign-in notices on the real first-party domain (wellsfargo.com) with 'no action needed' must be ham; key on domain legitimacy and absence of urgency, not on bank+login keywords.
- sev3 gold=ham ots=phishing@0.971 [en] "Marriott Rewards: You've earned 1,250 points from your stay at our Los Angeles property. Check balance at marriott.com/r" → Loyalty-points earned notices on the real brand domain are routine ham; 'points/rewards' tokens should not alone trigger phishing.
- sev3 gold=ham ots=phishing@0.9931 [en] "Google Play: Your payment method was declined for order #GPA87934. Update here: play.google.com/account/paymentmethods" → Billing/payment-declined notices pointing to the genuine play.google.com path are real service notices; treat first-party domains plus order references as ham.

### missed threat (3)

- sev3 gold=phishing ots=ham@0.8219 [ja] "銀行員です。あなたのアカウントの確認が必要です。パスワードを返信して下さい。" → Japanese impersonation of a bank employee with a direct request to reply with the password was read as benign; need credential-request cues (パスワードを返信) in non-Latin scripts.
- sev3 gold=phishing ots=ham@0.9606 [en] "SEND THIS CODE TO CUSTOMER SERVICE TO UNLOCK YOUR ACCOUNT: 384762" → Messages instructing the user to forward/send an OTP code to anyone are phishing, even when they contain no link or brand name.
- sev2 gold=phishing ots=ham@0.8159 [es] "Hola! Es tu tía María. Cambié de número. ¿Puedo contar contigo? Mi teléfono se rompió y necesito urgente." → Family-impersonation 'new number / phone broke / urgent help' scripts, including non-English variants, must be phishing even without links.

### category swap (6)

- sev1 gold=phishing ots=spam@0.9475 [ar] "جنة الذهب: استثمر 5,000 درهم واربح 50,000 درهم في شهر واحد! إرسال الأموال الآن: TransferWise+966..." → Investment pitches that instruct the reader to wire/transfer money to a named service are phishing, not merely money-making spam.
- sev1 gold=spam ots=phishing@0.9866 [tr] "İş buldum! Haftalık $1,500 kazanın evden. Tıkla: t.me/joboffers" → Generic work-from-home earnings ads with no impersonation and no 'you were selected' claim are spam under decision 2, even with a t.me link.
- sev1 gold=spam ots=phishing@0.8996 [en] "FREE iPhone 15 Pro Max! 100 winners announced daily. Claim yours: giveaway.site/apple99" → Giveaway ads that announce future winners ('100 winners announced daily') do not claim the reader already won, so they stay spam under decision 1.
- sev1 gold=spam ots=phishing@0.9062 [en] "Earn $$$ CRYPTO FAST!! B i g gains, n o risk. Z e r o fees f o r new members. Join: tg.me/cryptogang" → Generic crypto-group recruitment ads with no impersonation and no 'you were selected/won' claim are spam; letter-spacing obfuscation should not by itself push to phishing.
- sev1 gold=phishing ots=spam@0.8636 [ru] "Бухгалтер Гугл, полностью удалённо! 150,000 рублей в месяц. Заявку отправьте на google-jobs-russia.org" → Job ads that impersonate a real brand (Google) via a lookalike domain are phishing, not generic work-from-home spam, including in Russian.
- sev1 gold=spam ots=phishing@0.9042 [ru] "ВЫИГРЫВАЙ! 500,000 рублей и iPhone 15! Перейди: bit.ly/winning2024russia" → Imperative 'win X' prize-draw ads with no claim/already-won wording are spam, not phishing; distinguish 'ВЫИГРЫВАЙ' (invitation) from 'вы выиграли/заберите приз'.

## Files

- `logs/<day>.jsonl`: every message with the OTS answer and the grader verdict.
- `logs/improvements.log`: one line per miss, appended daily.
- `logs/training_additions.csv`: 28 graded misses with their gold label, ready to add as a source in `evals/distill_labels.py`.
