# Rebuilding the figures and tables

Every figure and data table of the paper, and of `SUPPLEMENTARY.md`, is rebuilt
here from files checked in under `results/`. No inference is needed and nothing
has to be installed first: each script carries its dependencies inline
(PEP 723) and `uv run` resolves them into a throwaway environment.
`make_figures.py.lock` pins the versions used for the sweep figures.

How the files in `results/` were produced from the model is described in
[`experiments/LLM/README.md`](../experiments/LLM/README.md).

## Map

| Paper artifact | Script | Input under `results/` |
|---|---|---|
| Fig. `fig:validity-bounds` (`rn_vs_sr_bounds.pdf`) and the bound values quoted in `sec:analysis-mlp` | `plot_bounds.py --mode precision` | none (closed form) |
| Table `tab:prism-vector-perf` and the precision-invariance sentence after it | `prism_bench_table.py` | `prism_vs_mcaquad_512/` |
| Fig. `fig:drift-distribution` (`drift_boxplot.pdf`) | `plot_drift_3row.py` | `logit_full_boxes/`, `logit_fluct/`, `logit_drift/` |
| Fig. `fig:head-decomposition-terms` (`site_decomposition_terms.pdf`) | `plot_decomposition_terms.py` | `distributions_s32.csv`, `quadratic_loss_summary.csv` |
| Table `tab:decomposition` | read directly | `distributions_s32.csv` ($t=6,7,8$), `drift_cells_ext/` ($t=4,5,9,10$) |
| Eight-seed values in `app:drift-measurements` (null control, $\hat Q$ at $S=8$) | read directly | `quadratic_loss_summary.csv` |
| Figs. `fig:global`, `fig:component`, `fig:sublayer`, `fig:cumulative` | `make_figures.py` | `sweep_records.csv` |
| Perplexities quoted in `sec:experiments`; its excess losses are $\ln(\mathrm{PPL}/62.51)$ of these | `make_figures.py --summary` | `sweep_records.csv` |
| Table `tab:accuracy-matched-precision` | `mixed_table.py` | `mixed_followup_20260915/outcomes/` |
| Supp. Fig. S1 (`blockwise_sensitivity`) | `make_figures.py` | `sweep_records.csv` |
| Supp. Fig. S2 (`logit_drift_bins`) | `plot_logit_drift_bins.py` | `logit_drift/` |
| Supp. Fig. S3 (`fluctuation_scaling`) | `plot_drift_fluctuation.py` | `distributions_s32.csv` |
| Supp. Fig. S4 (`token_uncertainty`) | `plot_token_uncertainty.py` | `token_fisher_*.csv` |
| Supp. Tables S1–S2 | `make_figures.py --summary` | `sweep_records.csv` |
| Supp. Table S3 | read directly | `distributions_s32.csv`, `drift_cells_ext/` |

Helper modules without a figure of their own: `parse_sweep_logs.py` (log parser
behind `make_figures.py`), `composition_stats.py` (Student-$t$ tail used by
`plot_drift_fluctuation.py` and `experiments/LLM/drift_study.py`),
`plot_logit_drift_map.py` (shared constants for `plot_logit_drift_bins.py`; run
alone, it draws the fuller per-logit diagnostic).

## Commands

```bash
uv run scripts/plot_bounds.py --mode precision
uv run scripts/prism_bench_table.py            # add --latex for the table body
uv run scripts/make_figures.py --strict        # add --summary for the numbers
uv run scripts/plot_drift_3row.py
uv run scripts/plot_decomposition_terms.py
uv run scripts/plot_logit_drift_bins.py
uv run scripts/plot_drift_fluctuation.py
uv run scripts/plot_token_uncertainty.py
uv run scripts/mixed_table.py
```

Each figure script writes into `figures/` (`--out` redirects). From a fresh
clone these reproduce the committed figures pixel for pixel, except
`rn_vs_sr_bounds.pdf`, whose text is set in whichever sans-serif font the
machine provides. The two table scripts print to the terminal and accept
`--in <dir>` to read a rerun instead of the recorded data.

`mixed_table.py` reports the gap as mixed minus RN, the sign convention of the
table, and relative baselines against the recorded $t=24$ reference
($62.5118$). The `summary.json` stored with the recorded runs was written by
the as-run driver, which reports the gap as RN minus mixed; the values are
otherwise identical.

## Sweep records

`make_figures.py` reads the raw sweep logs under
`results/perplexity_logs_epoch/256/` when they are present and otherwise falls
back to `results/sweep_records.csv`, one row per completed configuration. The
two paths give pixel-identical figures. The raw logs are not in the repository:
each is ten kilobytes of which one `Result ->` line matters. Refresh the CSV
from a log tree with `uv run scripts/make_figures.py --distill`.

The CSV carries more arms than the paper draws (the script reports which arm it
selects). The paper uses the RN-background, mode-scoped arms
(`scoped_*_rndefault`, and `global_fixed_*` for the global sweep), in which
every operation outside the target runs at $t=24$ under RN.

## Drift and fluctuation summaries

`drift_boxplot.pdf` uses exact box summaries over all 51,262,140 scored
token–coordinate pairs in each (site, $t$) cell, stored in
`results/logit_full_boxes/`. Pass `--sampled` to `plot_drift_3row.py` to draw the
earlier 200,000-pair approximation from `logit_drift/` and `logit_fluct/`
instead. The bundles in `logit_drift/` hold seed-level means at several pooling
levels with their standard errors, the drift grouped into reference-probability
bins, and a 200k uniform subsample; the `[N, V]` arrays they reduce stay on the
cluster.
