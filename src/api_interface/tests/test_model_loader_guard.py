"""The model loader must refuse a Git LFS pointer instead of failing inside torch.load.

A clone or CI checkout without `git lfs pull` writes a ~130 byte text pointer where the
weights should be. It passes Path.exists(), so the service used to start and only fail
later with an opaque torch error, or serve a model that never loaded.
"""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

POINTER = (
    b"version https://git-lfs.github.com/spec/v1\n"
    b"oid sha256:9f2c1a1b0d5e4f3a8c7b6d5e4f3a2b1c0d9e8f7a6b5c4d3e2f1a0b9c8d7e6f5a\n"
    b"size 711504880\n"
)


@pytest.fixture
def pointer_file(tmp_path):
    path = tmp_path / "mbert_ots_model_2.7.pth"
    path.write_bytes(POINTER)
    return path


def test_pointer_is_detected_by_signature(pointer_file):
    """The guard's two conditions both hold for a real LFS pointer."""
    assert pointer_file.stat().st_size < 1_000_000
    with open(pointer_file, "rb") as handle:
        assert handle.read(64).startswith(b"version https://git-lfs")


def test_real_checkpoint_is_not_flagged(tmp_path):
    """A file over the size floor is never inspected as a pointer."""
    path = tmp_path / "mbert_ots_model_2.7.pth"
    path.write_bytes(b"PK\x03\x04" + b"\0" * 1_000_001)
    assert path.stat().st_size >= 1_000_000


def test_loader_skips_pointer_without_calling_torch(monkeypatch, pointer_file):
    """load_mbert_models() must skip the pointer and never reach torch.load."""
    from src.api_interface.services import model_loader as ml

    monkeypatch.setattr(ml.settings, "mbert_model_configs", {
        "multilingual": {"path": str(pointer_file), "tokenizer": "bert-base-multilingual-cased",
                         "num_labels": "3", "version": "2.7"},
    })
    monkeypatch.setattr(ml.settings, "models_base_path", pointer_file.parent)

    def explode(*args, **kwargs):  # pragma: no cover - fails the test if reached
        raise AssertionError("torch.load must not be called on an LFS pointer")

    monkeypatch.setattr(ml.torch, "load", explode)
    manager = ml.ModelManager()
    manager.load_mbert_models()
    assert manager.mbert_models == {}, "a pointer file must not register a loaded model"
