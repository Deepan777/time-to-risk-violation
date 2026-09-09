"""Config resolution, run manifests, and IO contracts.

`implementation_plan.md` section 10 requires one test in particular: a run manifest must contain
every field listed in `experimental_protocol.md` section 9. That requirement is asserted against
`REQUIRED_FIELDS` so the manifest and the specification cannot drift apart silently.
"""
from __future__ import annotations

import json

import pytest

from src.utils.config import SCALES, Config, deep_merge, load_config
from src.utils.io import should_skip, write_json
from src.utils.manifest import REQUIRED_FIELDS, RunManifest, run_manifest
from src.utils.seeding import SEEDS_DEV, SEEDS_FULL, SEEDS_SMOKE, seed_everything


# ------------------------------------------------------------------ config

def test_deep_merge_does_not_mutate_inputs():
    base = {"a": {"b": 1, "c": 2}}
    over = {"a": {"c": 3}}
    out = deep_merge(base, over)
    assert out == {"a": {"b": 1, "c": 3}}
    assert base == {"a": {"b": 1, "c": 2}}, "deep_merge mutated its base argument"


def test_config_requires_a_scale(tmp_path):
    p = tmp_path / "no_scale.yaml"
    p.write_text("validity:\n  H: 20\n", encoding="utf-8")
    with pytest.raises(KeyError, match="scale"):
        load_config(p)


def test_config_rejects_an_unknown_scale(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("scale: enormous\n", encoding="utf-8")
    with pytest.raises(ValueError, match="scale must be one of"):
        load_config(p)


def test_extends_chain_resolves_and_child_wins(tmp_path):
    (tmp_path / "parent.yaml").write_text(
        "scale: smoke\nvalidity:\n  H: 20\n  L: 20\n", encoding="utf-8")
    (tmp_path / "child.yaml").write_text(
        "extends: parent.yaml\nvalidity:\n  H: 5\n", encoding="utf-8")
    cfg = load_config(tmp_path / "child.yaml")
    assert cfg["validity.H"] == 5, "child value did not override parent"
    assert cfg["validity.L"] == 20, "parent value was lost"
    assert cfg.scale == "smoke"


def test_require_raises_on_missing_key():
    cfg = Config({"scale": "smoke"})
    with pytest.raises(KeyError, match="required config key"):
        cfg.require("validity.eps")
    assert cfg.get("validity.eps", 0.2) == 0.2


def test_config_hash_is_order_independent():
    a = Config({"scale": "full", "x": 1, "y": 2})
    b = Config({"y": 2, "x": 1, "scale": "full"})
    assert a.hash() == b.hash()
    assert Config({"scale": "full", "x": 2}).hash() != a.hash()


def test_shipped_configs_all_load_and_declare_a_valid_scale():
    from src.utils.cli import REPO_ROOT
    found = list((REPO_ROOT / "configs" / "scales").glob("*.yaml"))
    assert found, "no scale configs shipped"
    for p in found:
        cfg = load_config(p)
        assert cfg.scale in SCALES


# ------------------------------------------------------------------ manifests

def test_manifest_contains_every_required_field():
    man = RunManifest(experiment="unit-test")
    assert man.missing_fields() == [], (
        f"manifest is missing fields required by experimental_protocol.md section 9: "
        f"{man.missing_fields()}"
    )
    for f in REQUIRED_FIELDS:
        assert f in man.to_dict()


def test_manifest_records_timing_and_writes_on_success(tmp_path):
    out = tmp_path / "man.json"
    with run_manifest("e2e", out_path=out, config={"scale": "smoke"}) as man:
        man.metrics["mae"] = 0.25
    written = json.loads(out.read_text(encoding="utf-8"))
    assert written["status"] == "completed"
    assert written["metrics"]["mae"] == 0.25
    assert written["scale"] == "smoke"
    assert written["duration_seconds"] >= 0.0
    assert written["started_at"] and written["ended_at"]


def test_manifest_is_written_even_when_the_run_crashes(tmp_path):
    """Integrity rule 7: failures are recorded, not hidden. A crashed run that leaves nothing
    behind is an invisible failure."""
    out = tmp_path / "failed.json"
    with pytest.raises(RuntimeError):
        with run_manifest("crash", out_path=out, config={"scale": "smoke"}):
            raise RuntimeError("boom")
    written = json.loads(out.read_text(encoding="utf-8"))
    assert written["status"] == "failed"
    assert any("boom" in e for e in written["exceptions"])


def test_manifest_captures_warnings(tmp_path):
    import warnings
    out = tmp_path / "warn.json"
    with run_manifest("warns", out_path=out, config={"scale": "smoke"}):
        warnings.warn("degenerate covariance", RuntimeWarning)
    written = json.loads(out.read_text(encoding="utf-8"))
    assert any("degenerate covariance" in w for w in written["warnings"])


def test_manifest_records_git_commit():
    man = RunManifest(experiment="x")
    assert man.git_commit and man.git_commit != ""
    assert len(man.git_commit) == 40 or man.git_commit == "NOT A GIT REPOSITORY"


# ------------------------------------------------------------------ io / seeds

def test_should_skip_honours_force(tmp_path):
    f = tmp_path / "out.json"
    assert not should_skip(f, force=False), "missing output must not be skipped"
    write_json({"a": 1}, f)
    assert should_skip(f, force=False), "existing output must be skipped without --force"
    assert not should_skip(f, force=True), "--force must override the skip"


def test_seed_sets_are_the_pre_registered_ones():
    """The protocol fixes five seeds. Selecting a seed is forbidden, so the sets live in code."""
    assert SEEDS_FULL == (0, 1, 2, 3, 4)
    assert len(SEEDS_DEV) == 2 and len(SEEDS_SMOKE) == 1


def test_seed_everything_is_reproducible():
    import numpy as np
    seed_everything(7)
    a = np.random.rand(5)
    seed_everything(7)
    assert np.allclose(a, np.random.rand(5))


def test_seed_everything_rejects_a_bool():
    with pytest.raises(TypeError):
        seed_everything(True)  # bool is an int subclass; silently seeding with 1 would be wrong
