import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from local_llm_fitter import (  # noqa: E402
    Hardware, apply_overrides, classify, estimate_gb, load_models, model_label,
)

ROOT = Path(__file__).resolve().parents[1]
MODELS = load_models(ROOT / "models.json")


def test_estimate_scales_with_quant_and_params():
    # 8B: Q4 = 8*0.5*1.10+1.0 = 5.4 ; FP16 must be ~4x the weight bytes
    assert estimate_gb(8, "Q4") == 5.4
    assert estimate_gb(8, "Q8") == 9.8
    assert estimate_gb(8, "FP16") == 18.6
    assert estimate_gb(8, "Q4") < estimate_gb(8, "Q5") < estimate_gb(8, "Q8") < estimate_gb(8, "FP16")
    assert estimate_gb(16, "Q4") > estimate_gb(8, "Q4")


def test_filtering_tiers_on_16gb_machine():
    hw = Hardware(total_ram_gb=16, available_ram_gb=16, cpu_cores=4)
    tiers = classify(MODELS, hw)
    rec = {r["name"] for r in tiers["recommended"]}
    no = {r["name"] for r in tiers["not_recommended"]}
    # 7-8B models fit at Q8 inside 85% of 16 GB (=13.6 GB); 405B never does
    assert "Meta Llama 3.1 8B" in rec
    assert "Meta Llama 3.1 405B" in no
    assert "DeepSeek V3 671B" in no
    all_names = rec | {r["name"] for r in tiers["max_fit"]} | no
    assert all_names == {m["name"] for m in MODELS}


def test_manual_override_changes_verdict():
    hw = apply_overrides(Hardware(total_ram_gb=8, available_ram_gb=8), ram=64, vram=0)
    assert hw.total_ram_gb == 64 and hw.available_ram_gb == 64
    assert hw.gpu_vram_gb is None  # explicit 0 = no GPU
    tiers = classify(MODELS, hw)
    rec = {r["name"] for r in tiers["recommended"]} | {r["name"] for r in tiers["max_fit"]}
    assert "Meta Llama 3.1 70B" in rec  # impossible on 8 GB, possible on 64 GB at Q4


def test_moe_label_shows_total_and_active_and_sizes_on_total():
    mixtral = next(m for m in MODELS if m["name"] == "Mixtral 8x7B")
    label = model_label(mixtral)
    assert "MoE" in label and "46.7B total" in label and "12.9B active" in label
    row = classify([mixtral], Hardware(total_ram_gb=64))["recommended"][0]
    # memory is sized on TOTAL params (46.7B), not active (12.9B)
    assert row["sizes"]["Q4"] == estimate_gb(46.7, "Q4")


def test_empty_model_table_is_graceful():
    tiers = classify([], Hardware(total_ram_gb=32))
    assert tiers == {"recommended": [], "max_fit": [], "not_recommended": []}
    empty = ROOT / "tests" / "empty.json"
    empty.write_text(json.dumps({"models": []}))
    try:
        assert load_models(empty) == []
    finally:
        empty.unlink()


def test_unknown_quant_rejected_and_table_is_valid():
    try:
        estimate_gb(8, "Q3")
        raise AssertionError("should have raised")
    except ValueError:
        pass
    assert len(MODELS) >= 12
    for m in MODELS:
        assert m["params_b"] > 0 and m["active_params_b"] <= m["params_b"]
