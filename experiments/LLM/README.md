# Experiments: DistilGPT-2 under variable-precision SR and RN

This directory produces the data under `results/`. The figures and tables are
rebuilt from that data by the scripts in [`scripts/`](../../scripts/README.md),
without running any of this.

All runs evaluate `distilgpt2` on the first 1024 tokens of the WikiText-2 test
split shipped here (`wikitext-2-test.txt`), in four contexts of 256 tokens
(1020 scored positions). Every operation outside the target scope runs at
$t=24$ under RN.

## Setup

Runs execute in a published image that holds PyTorch 2.2.1 compiled with
Verificarlo v2.6.0, whose compiler pass routes every floating-point operation to
PRISM v0.0.8. PRISM's kernels are compiled for one instruction set, so there is
one image per x86-64 level; use the widest the CPU supports. Their recipe is in
[`containers/`](../../containers/README.md).

| Image                                          | Target      |
| ---------------------------------------------- | ----------- |
| `verificarlo/fuzzy:v2.6.0-pytorch2.2.1-sse2`   | `x86-64`    |
| `verificarlo/fuzzy:v2.6.0-pytorch2.2.1-sse4`   | `x86-64-v2` |
| `verificarlo/fuzzy:v2.6.0-pytorch2.2.1-avx2`   | `x86-64-v3` |
| `verificarlo/fuzzy:v2.6.0-pytorch2.2.1-avx512` | `x86-64-v4` |

The harness scripts in this directory, `fuzzy_torch` (`../python/`, which sets
PRISM's precision and rounding mode from PyTorch module hooks and fails a run
whose setting does not take effect) and a DistilGPT-2 cache are bound into the
image at run time. On a Slurm cluster with Apptainer:

```bash
apptainer build ~/sif/fuzzy-v2.6.0-avx2.sif docker://verificarlo/fuzzy:v2.6.0-pytorch2.2.1-avx2
./fetch_model.sh                    # once, on a node with network access
```

On a single machine, with `ENGINE=podman` or `ENGINE=docker` exported for every
command below:

```bash
./fetch_model.sh
```

`IMAGE` selects another image (a `.sif` path for Apptainer, an image name
otherwise). Outputs go under `$FUZZY_RUNS` (default `~/fuzzy-llm-runs`); the
model cache is `$FUZZY_RUNS/hf_cache`.

## Running

Each experiment is a list of configurations, one run per line, written by a
generator in `configs/`. A list runs either on a cluster, one array task per
line, or locally, a fixed number of lines at a time. Submit from this
directory:

```bash
configs/gen_sweep_configs.sh > sweeps.txt
sbatch --array=1-$(wc -l < sweeps.txt) slurm/run_array.sbatch sweeps.txt   # cluster
./run_local.sh sweeps.txt 16                                               # or locally
```

Every run is single-threaded and takes 30 to 60 minutes at the full 1024-token
budget; allow 3 GB of memory per run, 8 GB for captures (`sbatch --mem=8G`).
A line whose log already holds a result is skipped, so a list can be
resubmitted after an interruption. A failed run keeps its output as
`<log>.failed`.

The paper used the unpublished predecessor of these images (Verificarlo 2.5.1,
PRISM `4961dd3`). RN runs reproduce the recorded values bit for bit on the
`avx2` image; for example the $k=6$ all-RN control of the mixed-rounding table
gives 96.068. SR runs reproduce them only statistically: PRISM v0.0.8 draws
different random streams, so a given seed gives a different value. Results can
also differ in the last digits between the `sse2`/`sse4` images and the others,
which have hardware FMA.

## 1. Perplexity sweeps

Figures `fig:global`, `fig:component`, `fig:sublayer`, `fig:cumulative`,
Supplementary Fig. S1 and Tables S1–S2 → `results/sweep_records.csv`.

| Harness                           | Scope                                                        |
| --------------------------------- | ------------------------------------------------------------ |
| `test_global_perplexity.py`       | the whole model                                              |
| `test_percomponent_perplexity.py` | `attention`, `mlp`, `lm_head`, `attn_qkav`                   |
| `test_fine_perplexity.py`         | one projection, in all blocks, one block, or a block prefix |

`configs/gen_sweep_configs.sh` writes the 811 runs of the paper (five SR seeds
and one RN run per point, plus the $t=24$ reference); its `rn-replication` set
adds four RN repeats of every point. Logs go under
`$FUZZY_RUNS/perplexity_logs/256/`, which the figure script reads directly:

```bash
uv run ../../scripts/make_figures.py --logs $FUZZY_RUNS/perplexity_logs/256 --out <dir>
uv run ../../scripts/make_figures.py --logs $FUZZY_RUNS/perplexity_logs/256 --distill   # refresh sweep_records.csv
```

## 2. Logit captures and the loss decomposition

`fig:drift-distribution`, `fig:head-decomposition-terms`, `tab:decomposition`,
and Supplementary Figs. S2–S4 and Table S3.

`capture_logits.py` runs one (site, $t$, mode, seed) cell and stores its
per-token logit perturbation $\Delta$ against a reference capture.
`configs/gen_capture_configs.sh` writes them in four stages, run in order:
`ref`, `null`, `head`, `sites` (see its header, which also gives the commands
that extend `lm_head` and `mlp_c_proj` to 32 seeds). The cells go under
`$CAPTURE_ROOT/256/tok1024` (default `$FUZZY_RUNS/captures`) and take hundreds
of megabytes each.

The reductions below read that root and need only Python 3 and NumPy. Set
`HEAD_DECOMP_ROOT=$CAPTURE_ROOT/256/tok1024`, or pass `--root`:

| Output under `results/`                                 | Command                                                                                             |
| ------------------------------------------------------- | --------------------------------------------------------------------------------------------------- |
| `quadratic_loss_summary.csv` (all sites, $S=8$)         | `decompose.py report --root <root> --csv quadratic_loss_summary.csv`                                |
| `distributions_s32.csv` ($t=6,7,8$)                     | `drift_study.py audit`, then `cell` per (site, $t$), then `report` (`--root <root> --output <dir>`) |
| `drift_cells_ext/<site>_t<t>_<mode>.csv` ($t=4,5,9,10$) | `distributions.py cell --root <root> --site <site> --t <t> --mode <mode> --cell-csv <file>`         |
| `logit_drift/`                                          | `logit_drift_map.py <site> <t> 32 --out <dir>`                                                      |
| `logit_fluct/`                                          | `logit_fluct_map.py <site> <t> 32 --drift-dir <logit_drift dir> --out <dir>`                        |
| `logit_full_boxes/`                                     | `full_grid_drift_boxes.py <site> <t> --seeds 32 --out <dir>`                                        |
| `token_fisher_<site>_t<t>.csv`                          | `token_fisher_analysis.py <site> <t> 32 --out <dir>`                                                |

The last four run per cell, `lm_head` and `mlp_c_proj` at $t=6,7,8$;
`slurm/reduce_cells.sbatch` runs one of them over the six cells as an array.

Checks that need no capture: `python3 decompose.py selftest`,
`python3 decompose.py e2e`, `python3 drift_study.py selftest`, and
`python3 -m unittest test_decompose_cache test_drift_study`.
`decompose.py refcheck` compares the reference against the second RN seed and
the IEEE run of the `ref` stage.

## 3. Mixed rounding at matched precision

`tab:accuracy-matched-precision` → `results/mixed_followup_20260915/`.

`test_mixed_perplexity.py` evaluates one per-site assignment of precision and
rounding mode. `configs/gen_mixed_configs.sh` writes the table's 19 runs, and
`scripts/mixed_table.py` builds the table from their logs:

```bash
configs/gen_mixed_configs.sh > mixed.txt
./run_local.sh mixed.txt 19
uv run ../../scripts/mixed_table.py --in $FUZZY_RUNS/mixed_logs
```

The recorded runs were driven by `local_comparison_driver.py`, kept with them in
`results/mixed_followup_20260915/` together with its plan, logs and per-run
outcomes; its worker evaluates the same model with the same hooks as
`test_mixed_perplexity.py`.

## 4. PRISM and MCA timing

`tab:prism-vector-perf` → `results/prism_vs_mcaquad_512/`.

`bench/prism_vs_mcaquad_bench.cpp` times elementwise add, mul, div and FMA over
65,536 values in `binary32` and `binary64`. `bench/prism_vs_mca.sh` builds it at
512-bit vector width as native, PRISM SR and MCA quad RR binaries, checks that
all eight kernels call their backend, and times one run. It needs an AVX-512
CPU. Thirty runs, each on its own exclusive node:

```bash
apptainer build ~/sif/fuzzy-v2.6.0-avx512.sif docker://verificarlo/fuzzy:v2.6.0-pytorch2.2.1-avx512
apptainer exec ~/sif/fuzzy-v2.6.0-avx512.sif bench/prism_vs_mca.sh build <dir>
sbatch --array=1-30 slurm/prism_vs_mca.sbatch <dir>
uv run ../../scripts/prism_bench_table.py --in <dir>
```

Without Slurm, run `bench/prism_vs_mca.sh run <dir> <k>` for $k=1,\dots,30$ in
the same image on an otherwise idle machine. The recorded logs come from the
same benchmark on Verificarlo 2.5.1 and PRISM builds patched with what v2.6.0
and PRISM v0.0.8 now include; a run on v2.6.0 reproduces the timings within a
few percent.
