# OpenTextShield mBERT Model Training

Professional training suite for OpenTextShield mBERT models, fully optimized for Apple Silicon (M1, M2, M3, etc.) with MLX framework integration. This implementation provides both legacy compatibility and modern, maintainable code structure.

## 🚀 Key Features

- **Apple Silicon Optimization**: Native MLX framework support for maximum performance
- **Multi-device Support**: Automatic device detection (MPS, CUDA, CPU)
- **Modern Architecture**: Configurable, maintainable, and extensible codebase
- **Enhanced Training**: Validation splits, early stopping, comprehensive metrics
- **Security First**: Secure credential management and best practices
- **Dataset Management**: Comprehensive dataset organization and validation

## 📁 Project Structure

```
model-training/
├── config.py                     # Centralized configuration management
├── train_ots.py                 # Original training script (legacy)
├── train_ots_improved.py        # Enhanced training with modern features
├── translate_dataset.py         # Original translation script (legacy)
├── translate_dataset_secure.py  # Secure translation with proper error handling
├── dataset_manager.py           # Dataset organization and validation
├── load_bert.py                 # BERT to MLX conversion utility
├── main.py                      # MLX inference testing
├── clean_dataset.py             # Basic dataset cleaning
├── utils/
│   └── bert.py                  # MLX BERT implementation
├── CHANGELOG.md               # Version history and release notes
├── archive/                    # Historical models, datasets, and logs
└── dataset/                    # Training datasets
    ├── sms_spam_phishing_dataset_v2.1.csv  # Current production dataset
    └── ...                       # Historical datasets
```

## 🛠 Installation

### Prerequisites
- Python 3.8+
- Apple Silicon Mac (for MLX optimization) or any system with PyTorch support
- OpenAI API key (for dataset translation, optional)

### Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables (optional, for translation)
export OPENAI_API_KEY="your-api-key-here"
```

## 📚 Usage

### Quick Start (Recommended)
```bash
# Use the improved training script with modern features
python train_ots_improved.py
```

### Legacy Training (Original)
```bash
# Original training script (maintained for compatibility)
python train_ots.py
```

### Advanced Usage

#### Dataset Management
```bash
# List all available datasets
python dataset_manager.py list

# Validate current dataset
python dataset_manager.py validate

# Get dataset summary
python dataset_manager.py summary

# Clean up old datasets (keep latest 3 versions)
python dataset_manager.py cleanup --keep 3
```

#### Secure Dataset Translation
```bash
# Translate dataset with proper security
python translate_dataset_secure.py --language Spanish --input dataset/input.csv --output dataset/output.csv
```

#### MLX Model Conversion
```bash
# Convert BERT model to MLX format
python load_bert.py
```

#### MLX Inference Testing
```bash
# Test MLX model inference
python main.py
```

#### Inference Performance Benchmark on Apple Silicon M1 Pro

| Metric                        | Value                   |
|-------------------------------|-------------------------|
| **Inference Speed**           | 54 SMS messages/second  |
| **Tested Platform**           | Apple Silicon M1 Pro    |





#### Training Process

![6AZuNzub7YUb3aTsnzpsiK](https://github.com/TelecomsXChangeAPi/OpenTextShield/assets/19316784/bbce8f96-b3b3-4beb-9e78-417b47a09e15)


## 📋 Version History

See [CHANGELOG.md](./CHANGELOG.md) for detailed release notes and version history. The changelog follows [Keep a Changelog](https://keepachangelog.com/) format and includes:

- **Language Support Timeline**: Progressive multilingual capabilities from English to 7+ languages
- **Model Improvements**: Performance enhancements and accuracy improvements
- **Security Updates**: Enhanced phishing detection and threat mitigation
- **Breaking Changes**: API and compatibility notes

## Contact and Acknowledgements

We appreciate your interest in MLX Bert for OTS and welcome any questions, feedback, or contributions. Please feel free to reach out to us via the following channels:

### For OTS inquiries:
- **LinkedIn**: [Ameed Jamous](https://www.linkedin.com/in/ameedjamous/)
- **Email**: [a.jamous@telecomsxchange.com](mailto:a.jamous@telecomsxchange.com)
- **GitHub**: [TelecomsxchangeAPI/Open-Text-Shield](https://github.com/TelecomsxchangeAPI/Open-Text-Shield)

### For MLX-BERT inquiries:
- **LinkedIn**: [Tim Cvetko](https://www.linkedin.com/in/tim-cvetko-32842a1a6/)
- **Gmail**: [cvetko.tim@gmail.com](mailto:cvetko.tim@gmail.com)
