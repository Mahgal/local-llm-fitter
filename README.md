# local-llm-fitter

**Before you download 40 GB of weights, find out in 2 seconds whether the model will actually fit on your machine.**

![Python](https://img.shields.io/badge/python-3.10%2B-blue) ![Dependencies](https://img.shields.io/badge/dependencies-zero-brightgreen) ![License](https://img.shields.io/badge/license-MIT-green) ![Tests](https://img.shields.io/badge/tests-6%20passing-brightgreen)

`local-llm-fitter` detects your hardware (total RAM, available RAM, CPU cores, NVIDIA VRAM when present), then checks it against a built-in table of 13 mainstream open-weight models and tells you, for every quant level (Q4 / Q5 / Q8 / FP16), how much memory each model needs — ending in three plain verdicts: **✅ recommended**, **⚠️ max fit**, **❌ not recommended**.

No downloads, no dependencies, no telemetry. Pure Python standard library.

## Features

- **Real hardware detection (Linux)** — RAM from `/proc/meminfo`, VRAM via `nvidia-smi` when present; anything unreadable degrades gracefully to `unknown` instead of crashing
- **13 models built in** — Llama 3.1 (8B/70B/405B), Qwen2.5 (7B/32B/72B), Qwen3 235B-A22B, Mistral 7B, Mixtral 8x7B, Gemma 2 (9B/27B), Phi-4 14B, DeepSeek V3 671B
- **MoE done right** — Mixtral / Qwen3 / DeepSeek V3 are labelled with *total vs active* parameters, and memory is sized on **total** params (all experts must be resident — sizing on active params is the classic mistake)
- **All four quant levels per model** — Q4, Q5, Q8, FP16 side by side
- **What-if mode** — `--ram 64 --vram 24` answers "what could my friend's / my next machine run?" without owning it
- **`--json` output** for scripts and dotfiles

## Quick start

```bash
git clone https://github.com/Mahgal/local-llm-fitter.git
cd local-llm-fitter
python3 local_llm_fitter.py                  # your machine, as detected
python3 local_llm_fitter.py --ram 32 --vram 0   # what-if: 32 GB RAM, no GPU
python3 local_llm_fitter.py --json           # machine-readable
python3 -m pytest                            # 6 tests
```

## Real output — run on the machine that wrote this README

Environment, honestly stated: **a small Linux VM, 4 CPU cores, ~8 GB RAM, no GPU.** This is not a workstation — which is exactly the point of the tool. (`RAM available` is a live reading and will differ slightly on every run; total RAM and all model sizes are deterministic.)

```text
local-llm-fitter
============================================================
Hardware
------------------------------------------------------------
RAM total: 7.7 GB | RAM available: 1.7 GB | CPU cores: 4 | GPU: unknown / none detected

✅ RECOMMENDED — fits comfortably (even at Q8)
------------------------------------------------------------
  (none)

⚠️  MAX FIT — fits at Q4, no headroom for higher quants
------------------------------------------------------------
  Mistral 7B v0.3 [7.25B dense]
      Q4 5.0 GB · Q5 6.0 GB · Q8 9.0 GB · FP16 16.9 GB
  Qwen2.5 7B [7.62B dense]
      Q4 5.2 GB · Q5 6.2 GB · Q8 9.4 GB · FP16 17.8 GB
  Meta Llama 3.1 8B [8.03B dense]
      Q4 5.4 GB · Q5 6.5 GB · Q8 9.8 GB · FP16 18.7 GB
  Google Gemma 2 9B [9.24B dense]
      Q4 6.1 GB · Q5 7.3 GB · Q8 11.2 GB · FP16 21.3 GB

❌ NOT RECOMMENDED — Q4 alone exceeds 85% of RAM
------------------------------------------------------------
  Microsoft Phi-4 14B [14.7B dense]
      Q4 9.1 GB · Q5 11.1 GB · Q8 17.2 GB · FP16 33.3 GB
  Google Gemma 2 27B [27.2B dense]
      Q4 16.0 GB · Q5 19.7 GB · Q8 30.9 GB · FP16 60.8 GB
  Qwen2.5 32B [32.5B dense]
      Q4 18.9 GB · Q5 23.3 GB · Q8 36.8 GB · FP16 72.5 GB
  Mixtral 8x7B [MoE: 46.7B total / 12.9B active per token]
      Q4 26.7 GB · Q5 33.1 GB · Q8 52.4 GB · FP16 103.7 GB
  Meta Llama 3.1 70B [70.6B dense]
      Q4 39.8 GB · Q5 49.5 GB · Q8 78.7 GB · FP16 156.3 GB
  Qwen2.5 72B [72.7B dense]
      Q4 41.0 GB · Q5 51.0 GB · Q8 81.0 GB · FP16 160.9 GB
  Qwen3 235B-A22B [MoE: 235B total / 22B active per token]
      Q4 130.2 GB · Q5 162.6 GB · Q8 259.5 GB · FP16 518.0 GB
  Meta Llama 3.1 405B [405.9B dense]
      Q4 224.2 GB · Q5 280.1 GB · Q8 447.5 GB · FP16 894.0 GB
  DeepSeek V3 671B [MoE: 671B total / 37B active per token]
      Q4 370.1 GB · Q5 462.3 GB · Q8 739.1 GB · FP16 1477.2 GB

Estimates only (params × bytes/param × 1.10 + 1.0 GB KV/runtime). Check the download page for real file sizes.
```

On this VM the honest answer is "7–9B models at Q4, and that's it" — better to learn that here than after a 40 GB download.

## How the estimate works

```text
estimate_gb = params_b × bytes_per_param × 1.10 + 1.0 GB
```

| Quant | Bytes / param |
|---|---|
| Q4 | 0.5 |
| Q5 | 0.625 |
| Q8 | 1.0 |
| FP16 | 2.0 |

- `× 1.10` — 10% allowance for GGUF metadata and tensor overhead
- `+ 1.0 GB` — flat runtime + KV-cache allowance for a ~4k context. Longer context = larger KV cache; treat this as a floor.
- **Verdict budget = 85% of total RAM.** Planning to use 100% of RAM is planning to swap.
  - ✅ Recommended: even Q8 fits the budget
  - ⚠️ Max fit: Q4 fits, Q8 doesn't
  - ❌ Not recommended: Q4 alone exceeds the budget

> **These are estimates, not measurements.** Real GGUF variants (Q4_K_M, Q5_K_M, imatrix, …) differ by a few percent. Parameter counts are the publicly announced totals for each family. Always confirm the actual file size on the model's download page before downloading.

## Adding a model

Append one object to `models.json` (`params_b`, `active_params_b`, `architecture`: `dense` or `MoE`). That's the whole extension API.

## Need a hand picking / deploying a local model stack?

This tool answers the sizing question. If you're setting up local inference for a team — hardware shortlists, runtime choice (llama.cpp / Ollama / vLLM), or an internal model gateway — that's the kind of AI infrastructure work done at [khaan.si](https://khaan.si). No client stories or testimonials are listed here, because there are none to claim yet for this specific service.

## License

MIT — see [LICENSE](LICENSE).
