# Daily probe

Invents realistic SMS with a cheap model, sends them to the live API, has a
stronger model grade the answers against `docs/LABELING_GUIDE.md`, and logs what
the next training round should fix. It replaces "watch real traffic for a week"
until there is real traffic, and keeps running after that as a regression watch.

| Step | What | Model |
|---|---|---|
| Generate | 150 messages a day: ham, spam and phishing across rotating sectors, languages and styles, each attack paired with a benign twin. De-duplicated against every earlier day. | Claude Haiku 4.5 |
| Classify | `POST /predict/` on `https://ots.telecomsxchange.com` (or `--api`). | OTS |
| Grade | Correct label under the guide, error type, severity, and a one-line note on what OTS failed to recognise. Unrealistic messages are marked unusable and ignored. | Claude Opus 5 |

Outputs, all under `logs/`:

- `<day>.jsonl`: every message with the OTS answer and the grader verdict.
- `improvements.log`: one line per miss, appended daily. Read this first.
- `training_additions.csv`: every graded miss with its gold label, ready to
  register as a source in `evals/distill_labels.py` for the next fine-tune.
- `../REPORT.md`: overall and per-day accuracy, block accuracy, false-block and
  missed-threat rates, the worst categories and languages, and the grader notes.

Cost is a few cents a day: about 4 Haiku calls and 5 Opus calls with a cached
system prompt.

## Run it

```bash
export ANTHROPIC_API_KEY=...
ots/bin/python evals/probe/probe.py run            # one day, 150 messages
ots/bin/python evals/probe/probe.py run --n 200    # bigger day
ots/bin/python evals/probe/probe.py report         # re-render REPORT.md from the logs
ots/bin/python evals/probe/probe.py run --dry-run  # no LLM calls: bench100 through the plumbing
```

## Daily on the public host

The host runs it at 06:10 UTC from a systemd timer. Install once:

```bash
ssh -i ~/.ssh/ots-deploy-key.pem ubuntu@3.99.29.158
sudo mkdir -p /opt/ots-probe && sudo chown ubuntu /opt/ots-probe
python3 -m venv /opt/ots-probe/venv && /opt/ots-probe/venv/bin/pip install anthropic pydantic
# copy probe.py and docs/LABELING_GUIDE.md into /opt/ots-probe (see sync.sh --push)
sudo cp ots-probe.service ots-probe.timer /etc/systemd/system/
echo 'ANTHROPIC_API_KEY=sk-ant-...' | sudo tee /etc/ots-probe.env && sudo chmod 600 /etc/ots-probe.env
sudo systemctl daemon-reload && sudo systemctl enable --now ots-probe.timer
sudo systemctl start ots-probe.service && journalctl -u ots-probe -f
```

Pull the logs and report back into the repo with `evals/probe/sync.sh`; push a
changed `probe.py` or guide with `evals/probe/sync.sh --push`.

## Feeding the next training round

`logs/training_additions.csv` has `text,label` in the corpus format. Add it as a
source in `evals/distill_labels.py`, run `ask` so TypeSafe confirms the grader's
labels, then `build` and `evals/train_distill_seeds.sh`. Rows the two graders
disagree on are dropped automatically.
