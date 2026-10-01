# Stochastic Rounding in Low-Precision Transformer Inference

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23066028.svg)](https://doi.org/10.5281/zenodo.23066028)
[![Download PDF](https://img.shields.io/badge/Paper-Download%20PDF-B31B1B?logo=adobeacrobatreader&logoColor=white)](https://big-data-lab-team.github.io/fuzzy-llm/paper.pdf)

Code, data and manuscript for *Stochastic Rounding in Low-Precision Transformer
Inference: A Variable-Precision Emulation Study of a Small GPT-2*, by Yohan
Chatelain and Pablo de Oliveira Castro.

The paper compares stochastic rounding (SR) and round-to-nearest (RN) at matched
significand precision, site by site, in DistilGPT-2. Low-precision arithmetic is
emulated in PyTorch compiled with [Verificarlo](https://github.com/verificarlo/verificarlo),
whose [PRISM](https://github.com/verificarlo/prism) backend rounds every
floating-point operation to a virtual precision set at run time.

Every release of this repository is archived on Zenodo under the DOI
[10.5281/zenodo.23066028](https://doi.org/10.5281/zenodo.23066028), which
resolves to the latest version.

## Contents

| Path                  | What it holds                                                                     |
| --------------------- | --------------------------------------------------------------------------------- |
| `paper/`              | LaTeX sources of the manuscript                                                   |
| `figures/`            | the figures, as used by the manuscript and `SUPPLEMENTARY.md`                     |
| `SUPPLEMENTARY.md`    | extended tables and diagnostics                                                   |
| `results/`            | the recorded data behind every figure and table                                   |
| `scripts/`            | scripts that rebuild the figures and tables from `results/`                       |
| `experiments/LLM/`    | the harness that produced `results/`, runnable on a Slurm cluster or one machine |
| `experiments/python/` | `fuzzy_torch`, which sets PRISM's precision and rounding mode from PyTorch        |
| `containers/`         | the recipe of the published images the experiments run in                        |

## Rebuilding

The manuscript (needs a LaTeX distribution with `latexmk`):

```bash
make
```

The figures and tables are rebuilt from the recorded data by one script each,
run with [uv](https://docs.astral.sh/uv/), which installs its dependencies. For
example, the perplexity-sweep figures:

```bash
uv run scripts/make_figures.py --strict
```

[`scripts/README.md`](scripts/README.md) maps every figure and table to the
script and data that produce it, with the commands.

## Rerunning the experiments

The experiments run in published images of PyTorch 2.2.1 instrumented with
Verificarlo v2.6.0, one per x86-64 level: `verificarlo/fuzzy:v2.6.0-pytorch2.2.1-{sse2,sse4,avx2,avx512}`.
With Apptainer, podman or Docker, each experiment is a list of single-threaded
runs that executes as a Slurm array or in parallel on one machine.
[`experiments/LLM/README.md`](experiments/LLM/README.md) describes the setup and
gives the commands for each experiment.

RN runs reproduce the recorded values bit for bit. SR runs reproduce them
statistically: the recorded runs used an earlier PRISM whose random streams
differ, so a given seed gives a different value.
