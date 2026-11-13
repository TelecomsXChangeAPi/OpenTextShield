#!/usr/bin/env python3
"""Smoke-test a running OpenTextShield API after a deploy.

Checks /health reports the expected model version, then classifies a fixed set
of known ham, spam and phishing messages in several languages and fails if the
block decision (ham vs spam/phishing) is wrong on more than --max-misses of them.

    python evals/smoke_live.py --url https://ots.telecomsxchange.com --expect-model 2.12
"""
import argparse
import json
import sys
import time
import urllib.request

# (text, expected label) — the firewall decision is what matters: ham passes, spam and phishing block.
SAMPLES = [
    ("Hey, running 10 min late, order me a flat white?", "ham"),
    ("Your Uber code is 7782. Never share this code.", "ham"),
    ("Chase: Did you make a $612 purchase at BEST BUY? Reply YES or NO.", "ham"),
    ("Netflix: your password was changed. If this wasn't you, go to netflix.com/security", "ham"),
    ("Verizon: Your bill of $84.20 is ready. Pay by Nov 28 at myvzw.com/bill", "ham"),
    ("DHL: Ihr Paket 4582 1190 wird heute zwischen 10 und 14 Uhr zugestellt. Verfolgen: dhl.de", "ham"),
    ("Mamá, ya llegué a casa, te llamo después de cenar.", "ham"),
    ("ヤマト運輸：お荷物を本日お届け予定です。詳細はkuronekoyamato.co.jpでご確認ください。", "ham"),
    ("WIN BIG! 200 FREE spins, no deposit! Play now: luckyspin.bet 18+", "spam"),
    ("Earn $2,000 a week from home, no experience needed. Start today: bit.ly/wfh2k", "spam"),
    ("¡Préstamo rápido hasta 50.000 MXN sin buró! Solicita hoy en creditoya.mx", "spam"),
    ("URGENT: Your prize is waiting! Call 0906 123 4567 now. £1.50/min", "phishing"),
    ("Chase Alert: your account is LOCKED. Verify now: chase-secure-login.com", "phishing"),
    ("Hi Mum, I dropped my phone, this is my new number. Can you pay a bill for me? I'll pay you back tonight", "phishing"),
    ("Royal Mail: your parcel is held, pay the £1.99 redelivery fee at royalmail-redeliver.com", "phishing"),
    ("Dear customer your SBI KYC has expired, A/C will be blocked within 24hrs. Call 9876543210", "phishing"),
    ("Votre colis est en attente. Confirmez l'adresse sous 24h : laposte-suivi-colis.net", "phishing"),
    ("Amazonアカウントが一時停止されました。24時間以内に確認してください: amazon-jp-verify.com", "phishing"),
    ("تم تعليق حسابك في البنك. للتفعيل أدخل بياناتك هنا: alrajhi-verify.com", "phishing"),
]


def post(url, text, timeout=30):
    req = urllib.request.Request(f"{url.rstrip('/')}/predict/", method="POST",
                                 data=json.dumps({"text": text, "model": "ots-mbert"}).encode("utf-8"),
                                 headers={"Content-Type": "application/json", "accept": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    return body, (time.time() - t0) * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8002")
    ap.add_argument("--expect-model", help="model version /health must report")
    ap.add_argument("--max-misses", type=int, default=1)
    args = ap.parse_args()

    with urllib.request.urlopen(f"{args.url.rstrip('/')}/health", timeout=30) as resp:
        health = json.loads(resp.read().decode("utf-8"))
    model = health.get("model") or {}
    version = model.get("version") if isinstance(model, dict) else None
    print(f"health: status={health.get('status')} api={health.get('api_version')} model={version}")
    if args.expect_model and str(version) != str(args.expect_model):
        print(f"FAIL: expected model {args.expect_model}, /health reports {version}")
        return 2

    misses, lat = [], []
    for text, expected in SAMPLES:
        body, ms = post(args.url, text)
        lat.append(ms)
        got = body.get("label")
        ok = (got != "ham") == (expected != "ham")
        mark = "ok  " if ok else "MISS"
        print(f"{mark} {expected:8} -> {got:8} {body.get('probability', 0):.2f} {ms:6.0f}ms  {text[:60]}")
        if not ok:
            misses.append((text, expected, got))
    lat.sort()
    print(f"{len(SAMPLES) - len(misses)}/{len(SAMPLES)} block decisions right; "
          f"latency median {lat[len(lat) // 2]:.0f} ms, p95 {lat[int(len(lat) * 0.95) - 1]:.0f} ms")
    if len(misses) > args.max_misses:
        print(f"FAIL: {len(misses)} misses, allowed {args.max_misses}")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
