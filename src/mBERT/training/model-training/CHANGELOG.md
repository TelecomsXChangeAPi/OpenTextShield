# Changelog

All notable changes to the OpenTextShield mBERT model will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.12.0] - 2026-11-13

Built entirely in-house: no external labeling service is used to label, train
or serve. Fine-tuned from the 2.9 weights on `dataset/curated/train_v2.12.csv`
(162,297 rows), 2 epochs, seed 42.

### Changed
- Labels come from the guide's rules (`evals/label_audit.py`) and 12,672
  reviewed template clusters (`dataset/curated/label_review_2.12.csv`) instead
  of an external reviewing model.

### Added
- 9,547 generated rows in 27 languages, each graded blind by a second pass and
  deduplicated against the training and eval sets: shortener-link
  impersonation, no-link callback lures, wrong-number openers, prize callbacks,
  conversational ham and real notices with links. See
  `dataset/curated/SYNTHETIC_V2.12.md`.

### Improved (2.9 -> 2.12, real held-out sets)
- IMC25 smishing: phishing F1 81.0 -> 82.6 (recall 69.6% -> 71.9%), spam F1
  34.4 -> 35.5. Confirmed on a second, disjoint 18,268-row IMC25 sample.
- UCI: spam F1 65.8 -> 76.1 (recall 49.9% -> 61.9%), ham F1 99.0 -> 99.2.
- Mishra: ham F1 98.6 -> 98.9, spam F1 75.6 -> 77.1.
- Hard legitimate notices blocked: 1 of 40 -> 0 of 40.

### Known regression
- Mishra phishing F1 80.7 -> 73.5 (recall 73.5% -> 65.4%). Mishra labels UK
  phone-upgrade and dating ads as smishing; the labeling guide calls them
  spam. They are still blocked: Mishra block accuracy is 97.7% -> 98.2%.

## [2.9.0] - 2026-09-21

### Changed
- Trained on the whole 137k corpus relabeled under `docs/LABELING_GUIDE.md` by an
  external reviewing model (since removed from the project), with soft targets,
  instead of the 9.4k-row subset model 2.7 learned from.
- Prize, job and delivery lures are now phishing rather than spam, as the guide says.

### Added
- 13,430 generated legitimate branded notices in 14 languages and 9,358 lure
  twins, so a brand name plus a link no longer means phishing on its own.
- 3,263 conversational archetypes: new-number family scams, code-relay asks,
  wrong-number openers, fake orders with a callback number, and benign twins.

### Improved
- bench100 accuracy 83% -> 96%, false blocks on hard_legit 32.5% -> 0%,
  Mishra accuracy 89% -> 95%.
- Errors are no longer made at 0.99 confidence, so the SMPP threshold can act on them.

## [2.1.0] - 2024-05-08

### Added
- Multilingual support for Sri Lanka's official languages (Tamil and Sinhala)
- Enhanced cross-lingual capability for South Asian languages
- Improved model architecture for better language detection

### Changed
- Updated training dataset to include Tamil and Sinhala SMS samples
- Optimized model parameters for multilingual performance
- Enhanced tokenization for complex scripts

### Improved
- Better accuracy for non-Latin script languages
- Reduced false positives in multilingual contexts

## [1.9.0] - 2024-04-03

### Added
- German language dataset integration
- Italian language dataset integration
- Extended multilingual support to 5 languages total

### Changed
- Training pipeline now supports English, Arabic, Indonesian, German, and Italian
- Updated model weights for improved cross-lingual transfer learning

### Improved
- Enhanced spam detection accuracy for European languages
- Better handling of language-specific phishing patterns

## [1.8.0] - 2024-04-03

### Added
- German language dataset (initial implementation)
- Expanded multilingual capabilities to 4 languages

### Changed
- Training now supports English, Arabic, Indonesian, and German
- Updated data preprocessing for Germanic language features

### Fixed
- Improved tokenization for German compound words
- Enhanced character encoding handling for non-ASCII languages

## [1.7.6] - 2024-04-03

### Added
- Enhanced URL-based phishing detection
- Training on shortened URL patterns
- Specific phishing/ham URL classification examples

### Changed
- Improved model's ability to detect URL-based threats
- Updated training methodology for URL pattern recognition

### Security
- Better detection of malicious shortened URLs
- Enhanced phishing URL pattern recognition

## [1.7.5] - 2024-04-03

### Added
- Complete Indonesian language dataset
- Full localization support for Indonesian SMS patterns

### Changed
- Comprehensive Indonesian spam/ham training data
- Improved model performance for Indonesian language

### Improved
- Better accuracy for Indonesian spam detection
- Enhanced understanding of Indonesian linguistic patterns

## [1.7.0] - 2024-04-03

### Added
- Arabic language dataset (full integration)
- Indonesian spam/ham dataset (initial small dataset)
- Multilingual SMS classification capabilities

### Changed
- Extended model to support Arabic and Indonesian languages
- Updated training pipeline for right-to-left script support

### Improved
- Enhanced multilingual tokenization
- Better handling of Arabic script and diacritics

## [1.6.0] - 2024-04-03

### Added
- Initial multilingual support framework
- Enhanced model architecture for cross-lingual transfer

### Changed
- Updated base model architecture
- Improved training methodology

### Improved
- Better foundation for multilingual expansion
- Enhanced model stability and performance

## [1.5.0] - 2024-04-03

### Added
- Initial release of OpenTextShield mBERT model
- Basic English SMS spam/phishing detection
- Core model training infrastructure

### Features
- BERT-based transformer architecture
- SMS text classification (ham/spam/phishing)
- Apple Silicon MLX optimization support

## v2.4.1 — Dataset cleaning & leakage fix

### Fixed
- Removed 7,718 duplicate rows causing train/test leakage
  (7.4% → 0% under config.py split, seed=42).
- Removed 80 machine-translation artifacts and 10 junk-text rows.
- Extended the artifact marker list (Indonesian, Spanish, Italian, French, German,
  Russian, Arabic assistant replies): 335 more rows removed, now 137,668 rows.

### Added
- `clean_ots_dataset.py` — reproducible dataset cleaning utility.
- `test_clean_dataset.py` — tests for duplicates, labels, empty text.
- `dataset/sms_spam_phishing_dataset_v2.4.1_dedup.csv` — cleaned dataset.

---

## Release Notes

### Language Support Timeline
- **v1.5.0**: English (baseline)
- **v1.7.0**: + Arabic, Indonesian (partial)
- **v1.7.5**: + Indonesian (complete)
- **v1.8.0**: + German
- **v1.9.0**: + Italian
- **v2.1.0**: + Tamil, Sinhala

### Model Performance
Each version includes improvements to accuracy, reduced false positives, and enhanced multilingual capabilities. The model maintains backwards compatibility while expanding language support.

### Security Updates
All versions include security enhancements for better phishing detection and reduced attack surface for adversarial inputs.