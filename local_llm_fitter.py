#!/usr/bin/env python3
"""local-llm-fitter — will this local LLM fit on this machine?

Pure standard library. No model weights are downloaded, ever.

Size formula (see README for the full discussion):
    weights_gb = params_b * bytes_per_param[quant]
    estimate_gb = weights_gb * 1.10 + 1.0

  * 1.10  = 10% allowance for GGUF metadata / tensor overhead
  * +1.0 GB = flat runtime + KV-cache allowance for a ~4k context.
              Longer contexts need a larger KV cache; treat this as a
              floor, not a guarantee.

Quant bytes-per-parameter (idealised, before the 10% overhead):
    Q4 = 0.5, Q5 = 0.625, Q8 = 1.0, FP16 = 2.0 bytes/param

These are ESTIMATES. Real GGUF files (Q4_K_M, Q5_K_M, ...) differ by a
few percent. Always check the actual file size on the download page.

Memory sizing always uses TOTAL parameters. For MoE models every expert
must be resident, even though only a subset is active per token.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

QUANT_BPP = {"Q4": 0.5, "Q5": 0.625, "Q8": 1.0, "FP16": 2.0}
OVERHEAD_FACTOR = 1.10
KV_ALLOWANCE_GB = 1.0
HEADROOM = 0.85  # never plan to use more than 85% of total RAM

DEFAULT_MODELS = Path(__file__).with_name("models.json")


# ---------------------------------------------------------------- hardware
@dataclass
class Hardware:
    total_ram_gb: float | None = None
    available_ram_gb: float | None = None
    cpu_cores: int | None = None
    gpu_vram_gb: float | None = None  # None = unknown / not detected
    gpu_name: str | None = None

    def describe(self) -> str:
        ram = f"{self.total_ram_gb:.1f} GB" if self.total_ram_gb else "unknown"
        avail = f"{self.available_ram_gb:.1f} GB" if self.available_ram_gb else "unknown"
        gpu = (
            f"{self.gpu_name or 'GPU'} {self.gpu_vram_gb:.1f} GB VRAM"
            if self.gpu_vram_gb
            else "unknown / none detected"
        )
        return f"RAM total: {ram} | RAM available: {avail} | CPU cores: {self.cpu_cores or 'unknown'} | GPU: {gpu}"


def _read_meminfo(path: str = "/proc/meminfo") -> dict[str, float]:
    out: dict[str, float] = {}
    try:
        for line in Path(path).read_text().splitlines():
            parts = line.split()
            if len(parts) >= 2 and parts[0] in ("MemTotal:", "MemAvailable:"):
                out[parts[0].rstrip(":")] = int(parts[1]) / 1024 / 1024  # kB -> GB
    except OSError:
        pass
    return out


def _detect_gpu() -> tuple[float | None, str | None]:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None, None
    try:
        r = subprocess.run(
            [exe, "--query-gpu=memory.total,name", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        first = r.stdout.strip().splitlines()[0]
        mib, name = first.rsplit(",", 1)
        return int(mib.strip()) / 1024, name.strip()
    except Exception:
        return None, None


def detect_hardware() -> Hardware:
    mem = _read_meminfo()
    vram, gpu_name = _detect_gpu()
    return Hardware(
        total_ram_gb=mem.get("MemTotal"),
        available_ram_gb=mem.get("MemAvailable"),
        cpu_cores=os.cpu_count(),
        gpu_vram_gb=vram,
        gpu_name=gpu_name,
    )


def apply_overrides(hw: Hardware, ram: float | None, vram: float | None) -> Hardware:
    """Manual overrides: --ram sets both total and available (a what-if
    machine), --vram sets VRAM (0 explicitly means 'no GPU')."""
    if ram is not None:
        hw.total_ram_gb = ram
        hw.available_ram_gb = ram
    if vram is not None:
        hw.gpu_vram_gb = vram if vram > 0 else None
        if vram == 0:
            hw.gpu_name = None
    return hw


# ------------------------------------------------------------------- sizing
def estimate_gb(params_b: float, quant: str) -> float:
    if quant not in QUANT_BPP:
        raise ValueError(f"unknown quant {quant!r}, expected one of {sorted(QUANT_BPP)}")
    return round(params_b * QUANT_BPP[quant] * OVERHEAD_FACTOR + KV_ALLOWANCE_GB, 2)


def model_label(model: dict) -> str:
    if model.get("architecture") == "MoE":
        return (f"{model['name']} [MoE: {model['params_b']:g}B total / "
                f"{model['active_params_b']:g}B active per token]")
    return f"{model['name']} [{model['params_b']:g}B dense]"


def classify(models: list[dict], hw: Hardware) -> dict[str, list[dict]]:
    """Bucket models into recommended / max-fit / not-recommended.

    Budget = 85% of total RAM (OS + apps need headroom).
      recommended     : even Q8 fits the budget — run it comfortably.
      max-fit         : Q4 fits the budget, Q8 does not — it fits, barely.
      not-recommended : Q4 alone exceeds the budget.
    Empty input returns three empty buckets (no crash).
    """
    tiers: dict[str, list[dict]] = {"recommended": [], "max_fit": [], "not_recommended": []}
    if not models:
        return tiers
    budget = (hw.total_ram_gb or 0) * HEADROOM
    for m in models:
        row = {
            "name": m["name"], "label": model_label(m),
            "architecture": m.get("architecture", "dense"),
            "params_b": m["params_b"], "active_params_b": m.get("active_params_b", m["params_b"]),
            "sizes": {q: estimate_gb(m["params_b"], q) for q in QUANT_BPP},
            "gpu_full_load": bool(hw.gpu_vram_gb and hw.gpu_vram_gb >= estimate_gb(m["params_b"], "Q4")),
        }
        if row["sizes"]["Q8"] <= budget:
            tiers["recommended"].append(row)
        elif row["sizes"]["Q4"] <= budget:
            tiers["max_fit"].append(row)
        else:
            tiers["not_recommended"].append(row)
    for rows in tiers.values():
        rows.sort(key=lambda r: r["params_b"])
    return tiers


def load_models(path: Path | str = DEFAULT_MODELS) -> list[dict]:
    data = json.loads(Path(path).read_text())
    if isinstance(data, dict):
        return data.get("models", [])
    return data


# ---------------------------------------------------------------------- CLI
def render(hw: Hardware, tiers: dict[str, list[dict]]) -> str:
    lines = ["local-llm-fitter", "=" * 60, "Hardware", "-" * 60, hw.describe(), ""]
    if not any(tiers.values()):
        lines.append("No models in the table — nothing to fit. Add entries to models.json.")
        return "\n".join(lines)
    headers = [
        ("recommended", "✅ RECOMMENDED — fits comfortably (even at Q8)"),
        ("max_fit", "⚠️  MAX FIT — fits at Q4, no headroom for higher quants"),
        ("not_recommended", "❌ NOT RECOMMENDED — Q4 alone exceeds 85% of RAM"),
    ]
    for key, title in headers:
        lines += [title, "-" * 60]
        if not tiers[key]:
            lines.append("  (none)")
        for r in tiers[key]:
            s = r["sizes"]
            gpu = " | GPU full-load: yes" if r["gpu_full_load"] else ""
            lines.append(f"  {r['label']}")
            lines.append(f"      Q4 {s['Q4']:.1f} GB · Q5 {s['Q5']:.1f} GB · Q8 {s['Q8']:.1f} GB · FP16 {s['FP16']:.1f} GB{gpu}")
        lines.append("")
    lines.append("Estimates only (params × bytes/param × 1.10 + 1.0 GB KV/runtime). Check the download page for real file sizes.")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="local-llm-fitter", description="Which local LLMs fit on this machine?")
    p.add_argument("--ram", type=float, default=None, help="override total RAM in GB (e.g. --ram 32)")
    p.add_argument("--vram", type=float, default=None, help="override GPU VRAM in GB (0 = no GPU)")
    p.add_argument("--models", default=str(DEFAULT_MODELS), help="path to models.json")
    p.add_argument("--json", action="store_true", help="machine-readable JSON output")
    args = p.parse_args(argv)

    hw = apply_overrides(detect_hardware(), args.ram, args.vram)
    models = load_models(args.models)
    tiers = classify(models, hw)
    if args.json:
        print(json.dumps({"hardware": hw.__dict__, "tiers": tiers}, indent=2))
    else:
        print(render(hw, tiers))
    return 0


if __name__ == "__main__":
    sys.exit(main())
