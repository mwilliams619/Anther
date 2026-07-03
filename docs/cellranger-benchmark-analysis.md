---
name: cellranger-benchmark-analysis
description: >-
  Analyze Cell Ranger (cellranger multi / Flex) HPC benchmark runs from SLURM
  result CSVs. Use when given a CSV of per-run wall-clock times across CPU,
  memory, and/or read-count configurations and asked to characterize scaling,
  find the resource sweet spot, compute per-read throughput, compare cloud vs
  cluster, or produce a poster-style write-up with recommended configurations.
---

# Cell Ranger HPC Benchmark Analysis

A repeatable workflow for turning a batch of SLURM `cellranger multi` benchmark
runs into scaling curves, per-read throughput, memory/cloud verdicts, and a
size-tiered resource-recommendation write-up. Built for SAIL / IRIS-style
benchmarking of Flex (fixed-RNA-profiling) GEX datasets on Cell Ranger 10.

## When to use this skill

Trigger when the user provides a CSV (or several) of Cell Ranger benchmark runs
and wants any of:
- **CPU scaling** — how runtime falls with cores, where the knee is, recommended cores.
- **Reads / throughput** — minutes per read, whether runtime is linear in read count.
- **Memory sweep** — whether more RAM buys speed (it usually does not above 128 GB).
- **Cloud vs HPC** — processing-time comparison for large samples.
- **A write-up** — Abstract / Methods / Results / Conclusion + recommended-config table.

## Expected input schema

Benchmark CSVs typically have one row per run. Column names vary; map them to these roles:

| Role | Common column names | Notes |
|---|---|---|
| Dataset id | `sample_id`, `sample_label` | Strip pool/lane suffixes to group: `sample_id.split('_')[0]` |
| Config id | `config_id`, `description` | Encodes cpus/mem/time, e.g. `config_32cpu_128gb_48h` |
| CPU cores | `cpus` | Integer |
| Memory | `memory` | Often `128.GB` — parse with `.str.replace('.GB','')` |
| Replicate | `replicate` | 1–3; average across reps before fitting |
| Reads | `reads` | Total sequencing reads (if present) |
| File size | `size_gb` | FASTQ input size |
| Runtime | `duration_seconds` / `duration_minutes` / `duration_hours` | **Wall-clock**, not CPU-time |

**Data-integrity check (do this first, every time).** Compute
`bytes_per_read = size_gb*1e9 / reads` per dataset. Healthy Illumina GEX data
clusters at **~48–74 bytes/read**. Any dataset far outside that band has a stale
or mis-typed `reads` or `size_gb` — flag it, draw it as an open/excluded marker,
and keep it out of throughput fits until the user confirms units (read-pairs vs
total reads vs bases).

## Analysis recipe

Use pandas + scipy + matplotlib. Average replicates before fitting. All times in
wall-clock hours or minutes — never mix with CPU-hours except where explicitly
computing allocation cost.

### 1. CPU scaling — fit the amortized model per dataset

Fit `T(n) = a + b/n` where `a` is the non-parallelizable wall-clock floor and
`b/n` is the parallel part. The serial fraction `a / (a + b)` separates
well-scaling datasets (<5%) from serial/IO-bound ones (>10%, e.g. large embryo
samples that flatline above ~24–32 cores).

```python
from scipy.optimize import curve_fit
import numpy as np
def model(n, a, b): return a + b/n
def fit_cpu(g):  # g = one dataset's rows, replicate-averaged
    n, t = g['cpus'].to_numpy(float), g['dur_h'].to_numpy(float)
    (a, b), _ = curve_fit(model, n, t, p0=[t.min()*0.5, (t.max()-t.min())*n.min()], maxfev=10000)
    return dict(a=a, b=b, serial_frac=a/(a+b))
```

**Recommended cores** = smallest core count within 10% of that dataset's best
observed runtime (the knee). This is the headline output — it avoids paying for
cores past the point of diminishing returns.

```python
def recommend_cores(g):  # g sorted by cpus
    best = g['duration_minutes'].min()
    return int(g[g['duration_minutes'] <= best*1.10].sort_values('cpus').iloc[0]['cpus'])
```

Also report **CPU-hours** (`cpus * dur_h`) so the user sees allocation cost rise
with cores even when runtime is flat.

### 2. Reads — test linearity, report throughput

Use each dataset's **fastest** run (matches "fastest cluster performance per
dataset"). Fit `log(runtime) ~ log(reads)`; **slope ≈ 1.0 means linear scaling**.
Then per-read cost (`min_per_million = duration_minutes / (reads/1e6)`) should be
roughly flat — verify with Spearman of `min_per_million` vs `reads` (expect no
significant trend). Planning rule to surface: **runtime_min ≈ reads / (throughput
in reads-per-minute)**; SAIL data gave ~15 M reads/min (0.065 min/M).

### 3. Memory — distinguish signal from replicate noise

The expected finding is **no speedup above ~128 GB**. Prove it rather than
asserting it: compute the across-memory runtime spread and compare it to the
within-config **replicate CV** (`std/mean` across reps). If the spread is the
same size as replicate noise (typically 6–12%), it is noise, not a memory
effect. Report Spearman(mem, runtime); a significant *positive* slope (runtime
rising with RAM) further confirms memory is not limiting.

### 4. Cloud vs HPC

Compare **processing time only** (exclude upload) to assess the compute
environments directly, then also report end-to-end including upload. Expect cloud
to win on the **largest** inputs. Present as a stacked bar (processing +
upload-overlay) with the speedup factor in the title.

## Figure conventions

- One subplot per question; log-y for runtime, log-x for reads.
- Direct-label lines at their right end instead of a legend where it fits.
- Highlight the focal / anomalous dataset in a distinct color (e.g. a
  serial-bound or unit-suspect sample) and state why in the title or caption.
- Use `fig, ax = plt.subplots(...)` and `fig.savefig(...)` — never bare `plt.*`.
- Sentence-style titles that state the finding ("Runtime scales ~linearly with
  reads (slope 1.0)"), verified true against every plotted point before saving.

## Output: write-up structure

Mirror a benchmark-poster layout:

1. **Abstract** — context, cohort size/range, the 3–4 headline findings.
2. **Methods** — CPU sweep, memory sweep, reads analysis, cloud comparison; replicate count.
3. **Results & Discussion** — one subsection each: CPU (file-size-dependent optimum), Reads (linear law + throughput), Memory (128 GB sufficient, with the noise argument), Cloud (speedup on largest).
4. **Conclusion + Recommended Configurations** — a size-tiered table.

### Recommended-config table (template)

| Input file size | Recommended cores | Notes |
|---|---|---|
| < 40 GB | 36–44 | Cores cheap relative to short runtime |
| 40–100 GB | ~40 | Consistent knee |
| > 100 GB | 24–36 | Large files plateau at *fewer* cores (thread overhead / serial stages) |
| Very large / multi-day | 24–32 on HPC **or** cloud offload | Cloud ~2× faster processing |

Memory recommendation is constant: **request 128 GB**; more does not help.

## Gotchas

- **Wall-clock vs CPU-time:** benchmark runtime is wall-clock. CPU-hours is a
  derived cost metric, not a runtime.
- **Uneven replication:** some configs have 1 rep, others 3. Average per config;
  don't treat single-rep points as equally precise.
- **Stale read/size values:** always run the bytes/read sanity check before any
  throughput claim (see Data-integrity check above).
- **Serial-bound large samples:** a dataset whose runtime is flat across cores is
  not "broken scaling" — it is IO/serial-bound; recommend *fewer* cores plus a
  longer time limit, or cloud.
