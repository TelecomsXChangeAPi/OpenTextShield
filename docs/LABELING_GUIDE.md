# SMS Labeling Guide

**Status: approved 2026-09-17.**

One rule for every label in OpenTextShield: training data, test sets, and reviews of flagged rows. It follows how the fable5 adversarial suite is already labeled, so existing test labels stay valid.

## Decisions

1. **A message that says you already won or are owed something is phishing**, even if it looks like a cheap promo. Example: "You won a $1,352 Amazon gift card! Redeem at tinyurl.com/abc". About 7,500 training rows are labeled spam today and would change.
2. **Job and money-making ads are spam unless they pretend to be a real company or say you were picked.** "Earn $2,000 a week from home, start today" is spam. "You've been selected for a remote position at Amazon" is phishing. About 1,850 "Job offer: Earn $X/month" rows are labeled phishing today and would change.
3. **A login or verification code is ham.** It only becomes phishing when the message asks you to send the code to someone. About 1,800 "WhatsApp: New security code" rows are labeled phishing today and would change.
4. **Rows with no real message are removed, not labeled:** a bare link, a single word, translator or AI notes, gibberish. About 4,300 rows.

## The three labels

| Label | In one line |
|---|---|
| **ham** | Normal chat, or a real notice from a real service. |
| **spam** | Unwanted advertising. It may be pushy or shady, but it doesn't pretend to be someone else. |
| **phishing** | A trick to steal money, codes, passwords, or personal details. |

## How to decide

Ask these questions in order and stop at the first "yes".

1. **Is there no real message?** A bare link, a single word like "Alert!", a translator note like "Sorry, this text has no meaning", or gibberish.
   → **Remove the row.**
2. **Does it try to trick the reader?** Any one of these:
   - pretends to be a bank, delivery company, carrier, government office, well-known brand, employer, or family member;
   - says the reader already won, is owed, or was picked for money, a prize, a refund, or a job;
   - invents a problem (locked account, held parcel, unpaid bill or toll, legal trouble) and pushes the reader to click, call, pay, or reply;
   - asks for a password, card number, code, ID number, or money transfer.

   → **phishing**
3. **Is it advertising?** A shop, casino, loan, crypto group, health product, MLM, political ad, or any offer to buy, join, or sign up.
   → **spam**
4. **Otherwise** → **ham**

## Hard cases

| Message | Label | Why |
|---|---|---|
| "Your Uber code is 7782. Never share this code." | ham | A real code delivery. |
| "Hi, can you send me the code you just received?" | phishing | Asks for the code. |
| "Chase: Did you make a $612 purchase at BEST BUY? Reply YES or NO." | ham | A real fraud alert. It asks nothing sensitive. |
| "Chase Alert: account LOCKED. Verify now: chase-secure-login.com" | phishing | Invented problem, fake link. |
| "Netflix: your password was changed. If this wasn't you, go to netflix.com/security" | ham | A real security notice on the real domain. |
| "HMRC will never text you asking for bank details." | ham | A scam warning, not a scam. |
| "URGENT: Your prize is waiting! Call 0906 123 4567" | phishing | Says the reader already won. |
| "WIN BIG! 200 FREE spins, no deposit! Play now: luckyspin.bet" | spam | A casino ad. It doesn't say the reader already won. |
| "FREE iPhone 15! Enter code 1007 at bit.ly/123" | spam | A giveaway ad. |
| "You need to send money to the IRS NOW at tinyurl.com/9065" | phishing | Pretends to be the IRS and asks for money. |
| "Earn $2,464 per week working from home!" | spam | A money-making ad. |
| "You've been selected for a remote position at Amazon. $450/day. Join: t.me/..." | phishing | Pretends to be Amazon, says you were picked. |
| "Get a free data booster with every new phone plan at MobileNet. Call 800-555-0199" | spam | A normal carrier ad. |
| "Hi Mum, I dropped my phone, this is my new number" | phishing | Pretends to be family. |
| "Hello, wrong number?" | ham | Normal chat. |

## Other rules

- **Label the meaning, not the wording.** Disguised text (fake letters, spaces between letters, "p4ssw0rd") gets the label it would have if written normally.
- **Any language, same rule.** A translated row keeps the label of its meaning. If the translation broke the meaning, remove the row.
- **Templates:** when one message pattern repeats with only names, amounts, codes, or links changed, all copies get the same label.
- **When unsure between two labels,** leave the row out of training and add it to the review list. Don't guess.

## How TypeSafe fits in

TypeSafe is only a reviewer. It flags rows whose label looks wrong, and a person decides using this guide. Its questions must use the definitions above: its earlier audit used a stricter phishing definition, so some of its spam→phishing flags (like "FREE iPhone") don't apply under decision 1. No client messages or audit logs are ever sent to it, and the shipped product never calls it.
