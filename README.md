<img width="300" height="400" alt="Tanuki" src="https://github.com/user-attachments/assets/94f1d7ac-7f63-4cfd-8298-425bb9787b90" />

## Tanuki: single-cell MALDI-MSI + optical microscopy pipeline

Tanuki turns paired optical microscopy and MALDI mass spectrometry imaging (imzML) of cultured cells into a
single-cell table: one row per segmented cell, with its morphology and its mean intensity at every *m/z* of a
mass list. It then compares the cells with embeddings, clusterings, a Bonsai tree, mean spectra and
donor-level volcano plots, and opens an interactive viewer in which every point can be traced back to its cell.

It was built for iPSC-derived dopaminergic neurons, astrocytes and microglia from several donor lines (GBA
project). This is **version 5** (October 2026), packaged as `src/tanuki`.

## Pipeline

| Stage | Module | What it does |
|---|---|---|
| 1 | `stage1_preprocess`, `registration` | Registers the PRE scan (optical image before matrix) onto the POST scan (after MSI) by phase correlation plus a 6 x 6 tile similarity fit, with a pass/fail gate in micrometres: a folder whose fit is not credible is skipped, not cropped. Finds the ablated (fiducial) square and checks its size against the imzML header. Extracts every *m/z* of the mass list (+/- 10 ppm), TIC-normalises each pixel and maps the ion cube onto the square. Writes `PREMSI_stage1_ready.h5`, QC text files and audit images. Runs wells in parallel. |
| 2 | `stage2_segmentation` | Cellpose-SAM on the three optical channels (brightfield inverted, membrane, nuclei), each contrast-stretched separately. Neuron fragments are joined only end to end, never side by side (directional merge). Bright ion-image blobs that no mask covers become extra masks (`mask_source = msi_rescue`). The optical layers are aligned to the MSI grid. Writes `PREMSI_seg.npy` and audit images. |
| 3 | `stage3_features` | Per-cell morphology (area, perimeter, extent, eccentricity, axis lengths, solidity, skeleton metrics) and per-cell mean, sum and standard error for every *m/z*. Parquet per well. Runs in parallel. |
| 4 | `stage4_concat` | Concatenates the per-well tables into run-level parquet / CSV files. |
| 5 | `analysis`, `bonsai` | Area filter per cell type, zero-rate feature filter per cell type, sample labels (cell type, donor line, disease group, stimulation, slide, well) and optional batch correction (off by default). Embeddings: DensMAP (mass + morphology, mass only, morphology only), UMAP, t-SNE, PCA and NMF. Clusterings: k-means, Leiden and NMF. Also writes mean spectra per grouping, per-well QC tables and `cleaned_feature_matrix.csv`. Bonsai exports its input, optionally runs it, and plots the tree when a result exists. |
| + | `volcano`, `export` | Donor-level volcano plots (stimulated vs unstimulated within each cell type; cell type vs cell type) and an analysis-friendly raw per-cell export (CSV + parquet). |
| + | `dash_viewer` | Interactive Dash app. Colour the embeddings by any label, click a cell to see its optical crop, its mask, its ion image and its spectrum. |

Stages 1 and 2 skip wells that are already done: stage 1 when a current-format `PREMSI_stage1_ready.h5` exists, stage 2 when the masks were made with the current merge rule. Stage 3 re-measures every well, and stages 4 and 5 always run. A well that fails in stage 1 or 3 is retried once in a single-worker "safe mode".

## Installation

Tested with Python 3.13 on Windows 10 (24 cores, CPU-only PyTorch). Python 3.10 or newer is required.

```bash
git clone https://github.com/SEllis86/Tanuki.git
cd Tanuki
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

`requirements.txt` pins the versions the pipeline was run with; `pip install -e .` alone installs compatible
versions and adds the `tanuki` command. PyTorch comes with Cellpose. For a GPU build, see
[pytorch.org](https://pytorch.org/get-started/locally/) for the command that matches your CUDA version.

Without installing, run `python run_tanuki.py ...` from the repository folder instead of `tanuki ...`.

**Bonsai (optional).** Clone [Bonsai-data-representation](https://github.com/dhdegroot/Bonsai-data-representation)
into its own environment and set `BONSAI_REPO` (and `BONSAI_PYTHON`, that environment's Python) to run it
automatically; otherwise Tanuki only writes the Bonsai input folder and a ready-to-run config.

## Input data

```text
RUN_FOLDER/
├── BIGmatched_peaks_raw.csv          mass list: first column = m/z; an optional 'name' column labels the volcano tables
├── A1_micro_11291_S2B1/              one folder per well (or per MSI acquisition)
│   ├── ..._pre MSI_....tif           optical scan before matrix: the name contains "pre"
│   ├── ..._POST_....tif              optical scan after MSI: the name contains "post" (if no name does, the single
│   │                                 other optical TIFF is used)
│   ├── A1.imzML + A1.ibd             the MSI acquisition
├── A1_1_neuro_11450_S2B1/            two acquisitions of one well: <well>_<n>_...
├── A2_astro_11291_stim_S2B2/
└── ...
```

The folder name carries the sample labels:

| Token | Example | Meaning |
|---|---|---|
| well position | `A1`, `B4` | first token |
| cell type | `micro`, `astro`, `neuro` / `Neurons` | case-insensitive |
| donor line | `11291` | five-digit line ID. Disease group (CONTROL / GBA) is set per line in `get_gba_category` (`analysis.py`) |
| stimulation | `stim` | present = STIM, absent = NO_STIM |
| slide | `S2B1` | the physical slide |

## Running

```bash
tanuki --input_dir "D:\data\GBA run" --csv_path "D:\data\GBA run\BIGmatched_peaks_raw.csv"
```

| Option | |
|---|---|
| `--input_dir DIR` | run folder (`MASTER_DIR`; default `./data`) |
| `--csv_path CSV` | mass list (`CSV_PATH`; default `<input_dir>/BIGmatched_peaks_raw.csv`) |
| `--config FILE` | settings file, `.json`, `.toml` or `.py` (UPPER_CASE names); may be repeated |
| `--set NAME=VALUE` | one setting, value read as JSON: `--set COMBAT_ENABLED=true --set N_CLUSTERS=8` |
| `--from-stage N` | start at stage N; `--from-stage 5` re-runs only the analysis on the existing tables |
| `--no-viewer` | finish without starting the Dash viewer |
| `--port P` | first port tried for the viewer (8050) |
| `--show-config` | print every setting with its resolved value and exit |

Every setting is defined, with the reasoning behind its default, in
[`src/tanuki/config.py`](src/tanuki/config.py): registration thresholds, Cellpose input, merge rule, MSI rescue,
area bounds per cell type, feature filters, batch correction, mean spectra, Bonsai, volcano/export switches.
[`configs/example.json`](configs/example.json) shows the format. Overrides also reach the worker processes of
stages 1 and 3 (through the `TANUKI_CONFIG_JSON` environment variable).

From Python:

```python
from tanuki import config
config.configure(MASTER_DIR=r"D:\data\GBA run", BONSAI_AUTO_RUN=False)   # before importing a stage module
from tanuki.main import run_pipeline
run_pipeline(from_stage=5, viewer=False)
```

The volcano plots and the raw export also run on their own, on any finished run:

```bash
python -m tanuki.volcano "D:\data\GBA run"
python -m tanuki.export "D:\data\GBA run"
```

## Outputs

Per well: `PREMSI_stage1_ready.h5` (registered optical layers + ion cube), `PREMSI_seg.npy` (masks, merged and
unmerged), `QC_registration_accuracy.txt`, `QC_square_size.txt`, `PREMSI_stage1_audit_registration_overlap.png`,
`PREMSI_stage2_audit_segmentation.png`, and the stage 3 parquets.

Per run (in the run folder):

| File / folder | |
|---|---|
| `<run>_concatenated_mask_id_averages.parquet` (+ sums, SEMs) | every measured cell, raw TIC-normalised intensities |
| `cleaned_feature_matrix.csv` | the analysed cells: labels, morphology, every embedding coordinate and cluster |
| DensMAP / UMAP / t-SNE / PCA / NMF `.html` + `.png` | embeddings coloured by group, cluster, slide, ... |
| mean spectrum `.html` / `.csv` per grouping | with the distinctive peaks of each group |
| `bonsai/` | Bonsai input, config and (if run) the tree + plots |
| `volcano/` | PNG + HTML + CSV per comparison, `volcano_summary.csv` |
| `export/` | `cells_raw.csv` / `.parquet`, `samples_raw.csv`, `features.csv`, `README.txt` |
| `<run>_benchmarks.txt` | time per stage |

## Volcano statistics

Cells are not replicates. For the iPSC data the biological replicate is the **donor line**. Each well's cells
are averaged (repeat acquisitions of one well pooled first), then each donor's wells are averaged. The two
groups are compared paired by donor. The per-donor log2 ratios are tested with a moderated one-sample t-test
(limma-style empirical Bayes, Smyth 2004, doi:10.2202/1544-6115.1027) and adjusted for FDR with
Benjamini-Hochberg. An *m/z* is significant at FDR < 0.05 and |mean log2 ratio| >= 1. The tables also give each
donor's log2 ratio and how many donors agree in direction, plus a cell-level Mann-Whitney test for reference
only.

## Package layout

```text
src/tanuki/
├── config.py               settings (v5 USER SETTINGS) + run-time overrides
├── main.py                 command line: stages 1-5, volcano/export, benchmarks, viewer
├── utils.py                shared helpers (labels from folder names, mass list, parquet I/O, colours)
├── registration.py         PRE -> POST phase-correlation registration
├── stage1_preprocess.py    stage 1
├── stage2_segmentation.py  stage 2
├── stage3_features.py      stage 3
├── stage4_concat.py        stage 4
├── analysis.py             stage 5 (run_post_concat_analysis), batch correction, mean spectra
├── bonsai.py               Bonsai export / run / plots
├── volcano.py              donor-level volcano plots
├── export.py               raw per-cell export
└── dash_viewer.py          interactive viewer
```

The package is `tanuki2026IPSC_phasecorr_reg_v5.py`, the single-file script, split into modules. Every function
keeps its v5 code. The changes are:

- **Settings.** Paths and Bonsai locations are no longer hard-coded. They default to `./data` and come from
  the command line or a settings file.
- **Mass list.** The list is read on first use instead of at import.
- **Stage 5 and the viewer.** The analysis and the Dash viewer were inline code in the script's `__main__`
  block. They are now the functions `analysis.run_post_concat_analysis()` and
  `dash_viewer.run_dash_interactive_viewer()`.
- **Neutrophil auto-launch.** The unused `AUTO_LAUNCH_NEUTROPHIL` setting is gone.

The version history of the script is in [CHANGELOG.md](CHANGELOG.md).

## License & Commercial Use

Tanuki is open-source and freely available under the **GNU Affero General Public License v3.0 (AGPLv3)**.

This means you are free to use, modify, and distribute this software for academic, personal, or open-source
projects, provided that you also release your modified source code under the same AGPLv3 license.

**Commercial Licensing**
If you represent a commercial entity and wish to use Tanuki in a proprietary, closed-source product or service
without the copyleft restrictions of the AGPLv3, please contact the author to negotiate a commercial license.
