"""--device: auto follows the installed torch build; a named backend must match it."""
import pytest

torch = pytest.importorskip("torch")

from r2v import device  # noqa: E402


def _fake(monkeypatch, hip, cuda, available):
    monkeypatch.setattr(torch.version, "hip", hip, raising=False)
    monkeypatch.setattr(torch.version, "cuda", cuda)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: available)


def test_auto_follows_the_torch_build(monkeypatch):
    _fake(monkeypatch, None, "12.8", True)
    assert device.pick("auto") == "cuda"
    _fake(monkeypatch, "7.2", None, True)
    assert device.pick("auto") == "rocm"


def test_named_backend_must_match_the_torch_build(monkeypatch):
    _fake(monkeypatch, "7.2", None, True)
    assert device.pick("rocm") == "rocm"
    with pytest.raises(SystemExit, match="--extra gpu"):
        device.pick("cuda")
    _fake(monkeypatch, None, "12.8", True)
    with pytest.raises(SystemExit, match="--extra amd"):
        device.pick("rocm")


def test_no_gpu_is_an_error(monkeypatch):
    _fake(monkeypatch, None, "12.8", False)
    with pytest.raises(SystemExit, match="no usable GPU"):
        device.pick("auto")
    _fake(monkeypatch, None, None, False)
    with pytest.raises(SystemExit, match="CPU-only"):
        device.pick("auto")


def test_unknown_name_is_rejected():
    with pytest.raises(SystemExit, match="unknown device"):
        device.pick("cpu")
