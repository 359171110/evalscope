from __future__ import annotations

from src import model_loading


def test_qwen35_path_uses_qwen36_loader_family(monkeypatch) -> None:
    calls = []

    def fake_loader(model_path, device_map=None, model_family=None):
        calls.append((model_path, device_map, model_family))
        return object(), object()

    monkeypatch.setattr(model_loading, "load_qwen3_moe", fake_loader)

    model_loading.load_supported_moe(
        "/data01/datasets/Qwen3.5-35B-A3B",
        device_map={"": "cpu"},
    )

    assert calls == [
        (
            "/data01/datasets/Qwen3.5-35B-A3B",
            {"": "cpu"},
            "qwen3.6",
        )
    ]


def test_existing_qwen3_path_keeps_qwen3_family(monkeypatch) -> None:
    calls = []

    def fake_loader(model_path, device_map=None, model_family=None):
        calls.append((model_path, device_map, model_family))
        return object(), object()

    monkeypatch.setattr(model_loading, "load_qwen3_moe", fake_loader)
    model_loading.load_supported_moe(
        "/data01/datasets/Qwen3-30B-A3B-Instruct-2507"
    )

    assert calls[0][2] == "qwen3"


def test_explicit_model_family_overrides_path_detection(monkeypatch) -> None:
    calls = []

    def fake_loader(model_path, device_map=None, model_family=None):
        calls.append((model_path, device_map, model_family))
        return object(), object()

    monkeypatch.setattr(model_loading, "load_qwen3_moe", fake_loader)
    model_loading.load_supported_moe(
        "/checkpoints/model-under-test",
        device_map={"": "cpu"},
        model_family="qwen3_5_moe",
    )

    assert calls == [
        (
            "/checkpoints/model-under-test",
            {"": "cpu"},
            "qwen3.6",
        )
    ]
