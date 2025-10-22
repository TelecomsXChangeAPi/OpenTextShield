# External Model Test Prompts

This directory contains test prompts for comparing OpenTextShield with external models (GPT-OSS).

## Files

### Test Prompts
- **`test_ham.json`** - Ham (legitimate) message test prompt for GPT-OSS
- **`test_spam.json`** - Spam message test prompt for GPT-OSS
- **`test_prompt.json`** - General test prompt for GPT-OSS

## Format

These files follow the OpenAI Chat API format:
```json
{
  "model": "openai/gpt-oss-20b",
  "messages": [
    {
      "role": "system",
      "content": "System prompt..."
    },
    {
      "role": "user",
      "content": "Test message..."
    }
  ],
  "temperature": 0.3,
  "max_tokens": 500,
  "stream": false
}
```

## Purpose

Used for comparative benchmarking between:
- OpenTextShield mBERT model
- External GPT-OSS models

## Usage

These prompts are used by comparison scripts to evaluate model performance across different platforms.

**Note**: These are NOT test fixtures for the OpenTextShield API. For OTS API test data, see `tests/data/`.
