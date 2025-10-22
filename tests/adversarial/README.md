# Adversarial Testing

This directory contains adversarial testing scripts for the OpenTextShield platform.

## Tests

### `advanced_adversarial_test.py`
Advanced adversarial evaluation testing for SMS classification models.

**Purpose**: Tests model robustness against adversarial inputs and edge cases

**Features**:
- Character substitution attacks
- Obfuscation techniques
- Edge case testing
- Comprehensive evaluation metrics

## Running Adversarial Tests

```bash
# Run advanced adversarial evaluation
python tests/adversarial/advanced_adversarial_test.py
```

## Test Reports

Results are generated in `docs/reports/`:
- `advanced_adversarial_evaluation_report.md` - Detailed evaluation results

## Adding New Tests

1. Create new test file following naming convention: `test_[attack_type].py`
2. Import base testing utilities
3. Document attack methodology
4. Generate reports with results
