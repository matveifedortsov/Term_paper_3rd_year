"""Tests that the config loader returns expected keys."""

from __future__ import annotations

from src.config import config, load_config, reload_config


REQUIRED_TOP_LEVEL_KEYS = {
    "mc", "lomn_real", "label", "features", "split", "xgb",
    "neural", "benchmarks", "significance", "sensitivity",
    "conformal", "hawkes", "lstm", "paths",
}


def test_default_config_loads():
    cfg = config()
    assert isinstance(cfg, dict)
    assert REQUIRED_TOP_LEVEL_KEYS.issubset(cfg.keys())


def test_config_has_pinned_seeds():
    cfg = config()
    assert "base_seed" in cfg["mc"]
    assert "seed" in cfg["neural"]
    assert "bootstrap_seed" in cfg["significance"]


def test_train_test_split_is_disjoint():
    cfg = config()
    train = set(cfg["split"]["train_days"])
    test = set(cfg["split"]["test_days"])
    assert train.isdisjoint(test)
    assert len(train) > 0 and len(test) > 0


def test_reload_picks_up_new_yaml(tmp_path, monkeypatch):
    p = tmp_path / "test.yaml"
    p.write_text("mc:\n  n: 99\n")
    monkeypatch.setenv("TERMPAPER_CONFIG", str(p))
    reload_config()
    cfg = config()
    assert cfg["mc"]["n"] == 99
    # restore default
    monkeypatch.delenv("TERMPAPER_CONFIG", raising=False)
    reload_config()
