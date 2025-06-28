# Changelog

All notable changes to the OpenTextShield mBERT model will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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