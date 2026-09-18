# Soak test: v2.11.0-rc.2 on an Apple M1 Pro, 2026-09-18

Image `ots:2.11.0-rc.2` built from the tag in a clean checkout, model 2.7, traffic through the
SMPP proxy with DLR relay. Stopped by hand after 1.2 hours; the run was scheduled for 48.
Harness: `evals/soak/`. Raw data: `~/ots-soak/run-2026-09-18/` on the test machine.


18482 messages over 1.2 h (2026-09-18T20:19:08.407Z to 2026-09-18T21:32:35.029Z)

## Outcomes by class

| class | sent | forwarded | rejected | error | timeout | block rate | false block |
|---|---|---|---|---|---|---|---|
| ham | 15176 | 14084 | 1092 | 0 | 0 | 7.2% | 7.20% |
| spam | 1799 | 41 | 1758 | 0 | 0 | 97.7% | |
| phishing | 1507 | 224 | 1283 | 0 | 0 | 85.1% | |

## Ham false blocks by pool (what an operator would see)

- ham_personal: 157/7692 = 2.04%
- ham_a2p: 935/7484 = 12.49%
  blocked A2P by category: {'legit': 562, 'notice_pair_legit': 287, 'legit_notice': 22, 'legit_delivery': 14, 'legit_security_notice': 13, 'legit_reminder': 10, 'legit_fraud_alert': 8, 'legit_billing': 7}

## Phishing by obfuscation

- plain      blocked 1219/1438 = 84.8%
- fullwidth  blocked 14/14 = 100.0%
- lookalike  blocked 16/19 = 84.2%
- zero_width blocked 15/17 = 88.2%
- spaced     blocked 19/19 = 100.0%

## Long messages via message_payload

- 5463 sent, outcomes {'rejected': 1934, 'forwarded': 3529}; attacks blocked 1434/1493

## Latency per hour (submit_sm round trip, ms)

| hour (UTC) | n | p50 | p95 | p99 | errors+timeouts |
|---|---|---|---|---|---|
| 2026-09-18T20 | 10422 | 144 | 249 | 357 | 0 |
| 2026-09-18T21 | 8060 | 145 | 283 | 465 | 0 |

## Errors and timeouts: 0


## Resources (first hour vs last hour)

- container memory: 1527 MB -> 1535 MB
- proxy RSS: 33 MB -> 30 MB
- health check latency: p50 46 ms, max 58 ms
- unhealthy samples: 0 of 72
- container restarts: 0
- audit dir inside container: 9108 KB; proxy log: 8196 KB
- API queue depth: max 1

## Proxy log counters

- classification errors (fail-open): 0
- API timeouts: 0
- below threshold (acted on): 230
- skipped unclassifiable: 0
- upstream reconnects: 0
- no upstream available: 0

## Sample of blocked legitimate messages (review these)

  [ham_a2p      curated_additions] id=7
  [ham_a2p      synthetic ] id=30
  [ham_a2p      synthetic ] id=33
  [ham_a2p      synthetic ] id=48
  [ham_a2p      curated_additions] id=74
  [ham_a2p      synthetic ] id=91
  [ham_a2p      synthetic ] id=96
  [ham_a2p      synthetic ] id=130
  [ham_a2p      synthetic ] id=131
  [ham_a2p      curated_additions] id=141
  [ham_personal uci       ] id=163
  [ham_a2p      curated_additions] id=172
  [ham_a2p      curated_additions] id=192
  [ham_a2p      curated_additions] id=214
  [ham_a2p      curated_additions] id=216
