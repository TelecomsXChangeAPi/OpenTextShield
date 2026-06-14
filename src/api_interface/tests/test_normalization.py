"""
Tests for the adversarial-input normalisation in the live batching path.

This is the security-critical fix that closes the obfuscation gap: zero-width /
invisible characters and homoglyphs are normalised *before* tokenisation so an
attacker can't split a keyword (e.g. "V​e​rify") into a wall of
[UNK]s that the model never learned. These tests document the regression being
prevented and exercise the defensive no-op fallback.

Run:
    pytest src/api_interface/tests/test_normalization.py -v
"""

import sys
from pathlib import Path

import pytest

# Make ``src`` importable when the tests are run from the repo root.
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.api_interface.services import batching_service
from src.api_interface.services.batching_service import _normalize_text
from src.api_interface.services.enhanced_preprocessing import EnhancedPreprocessor


ZWSP = "​"  # zero-width space — the canonical obfuscation character


@pytest.fixture
def pre():
    return EnhancedPreprocessor()


# --------------------------------------------------------------------------- #
# normalize_unicode: invisible-char stripping
# --------------------------------------------------------------------------- #
def test_zero_width_space_stripped(pre):
    """U+200B inserted between letters collapses back to the real word."""
    obfuscated = f"V{ZWSP}e{ZWSP}r{ZWSP}i{ZWSP}f{ZWSP}y"
    assert pre.normalize_unicode(obfuscated) == "Verify"


def test_every_invisible_char_is_removed(pre):
    """Each character in INVISIBLE_CHARS is stripped by normalize_unicode.

    Guards against the set drifting out of sync with the stripping logic.
    """
    for ch in EnhancedPreprocessor.INVISIBLE_CHARS:
        sample = f"a{ch}b"
        result = pre.normalize_unicode(sample)
        assert ch not in result, f"U+{ord(ch):04X} survived normalisation"
        assert result == "ab", f"U+{ord(ch):04X} not stripped cleanly: {result!r}"


def test_invisible_chars_set_covers_key_categories():
    """The set must include the common obfuscation characters."""
    chars = EnhancedPreprocessor.INVISIBLE_CHARS
    for codepoint in (0x200B, 0x200C, 0x200D, 0x2060, 0xFEFF, 0x00AD):
        # zero-width space, ZWNJ, ZWJ, word joiner, BOM, soft hyphen
        assert chr(codepoint) in chars, f"missing U+{codepoint:04X}"


# --------------------------------------------------------------------------- #
# normalize_unicode: homoglyph folding
# --------------------------------------------------------------------------- #
def test_cyrillic_homoglyph_folded(pre):
    """Cyrillic look-alikes fold to their Latin forms (all chars are in the map)."""
    # U+0440 р, U+0430 а, U+043E о, U+0445 х -> Latin p, a, o, x
    spoofed = "раох"
    folded = pre.normalize_unicode(spoofed)
    assert folded == "paox"
    assert folded.isascii(), f"homoglyphs not fully folded: {folded!r}"


def test_homoglyph_preserves_uppercase(pre):
    """Uppercase Cyrillic homoglyph folds to uppercase Latin."""
    assert pre.normalize_unicode("А") == "A"  # Cyrillic 'А' (U+0410) -> Latin 'A'


def test_plain_ascii_is_unchanged(pre):
    """Normalisation must not mangle ordinary messages."""
    msg = "Your OTP is 449128. Do not share it."
    assert pre.normalize_unicode(msg) == msg


# --------------------------------------------------------------------------- #
# Batching path: _normalize_text wiring + fallback
# --------------------------------------------------------------------------- #
def test_batching_path_normalises_zero_width():
    """The batching entrypoint applies normalisation (the production fix)."""
    obfuscated = f"cl{ZWSP}ick here"
    assert _normalize_text(obfuscated) == "click here"


def test_ascii_fast_path_skips_preprocessor(monkeypatch):
    """Pure-ASCII text returns unchanged without invoking the preprocessor.

    Locks in the perf fast-path: a preprocessor that raises on call proves the
    ASCII branch never reaches it.
    """
    class _Boom:
        def normalize_unicode(self, text):
            raise AssertionError("preprocessor must not be called for ASCII input")

    monkeypatch.setattr(batching_service, "_preprocessor", _Boom())
    assert _normalize_text("Your OTP is 449128") == "Your OTP is 449128"


def test_fallback_is_noop_when_preprocessor_unavailable(monkeypatch):
    """If the preprocessor failed to construct, _normalize_text passes text through.

    Documents that normalisation degrades gracefully rather than breaking a
    batch when EnhancedPreprocessor is unavailable.
    """
    monkeypatch.setattr(batching_service, "_preprocessor", None)
    obfuscated = f"cl{ZWSP}ick here"
    assert _normalize_text(obfuscated) == obfuscated  # unchanged, no exception


def test_normalize_never_raises_on_bad_input(monkeypatch):
    """A preprocessor that raises must not propagate into the batch path."""
    class _Boom:
        def normalize_unicode(self, text):
            raise RuntimeError("boom")

    monkeypatch.setattr(batching_service, "_preprocessor", _Boom())
    # Non-ASCII so it gets past the ASCII fast-path and actually hits the
    # preprocessor (which raises) — exercising the swallow.
    raising_input = f"caf{ZWSP}é"
    assert _normalize_text(raising_input) == raising_input  # swallowed -> original text
