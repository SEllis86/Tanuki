"""Command-line entry point: stages 1-5, the volcano plots + raw export, the benchmark file and the
Dash viewer, in that order (the __main__ block of tanuki2026IPSC_phasecorr_reg_v5.py).

    tanuki --input_dir "D:\\data\\GBA run" --csv_path "D:\\data\\BIGmatched_peaks_raw.csv"
    tanuki --input_dir "D:\\data\\GBA run" --from-stage 5 --no-viewer      (re-run the analysis only)
    tanuki --help
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from . import __version__, config


def run_extra_outputs():
    """Volcano plots + raw CSV export from this run's own outputs (VOLCANO_ENABLED, RAW_EXPORT_ENABLED).
    Never fatal: a failure is reported and the run carries on to the benchmark file and the viewer."""
    from . import export, volcano
    for enabled, func in ((config.RAW_EXPORT_ENABLED, export.export), (config.VOLCANO_ENABLED, volcano.run)):
        if not enabled:
            continue
        try:
            func(config.EXTRA_OUTPUT_PRESET, folder=config.MASTER_DIR, script=f"tanuki {__version__}")
        except Exception as e:
            print(f"⚠️ {func.__module__} failed ({type(e).__name__}: {e}); the rest of the run is unaffected.")


def write_benchmarks(pipeline_benchmarks, basename):
    benchmark_file_path = config.MASTER_DIR / f"{basename}_benchmarks.txt"

    summary_lines = []
    summary_lines.append("=" * 50)
    summary_lines.append("⏱️ PIPELINE BENCHMARK SUMMARY")
    summary_lines.append(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    summary_lines.append(f"System: {config.total_cores} cores detected, {config.cores_str} threads allocated.")
    summary_lines.append("=" * 50)

    for stage, duration in pipeline_benchmarks.items():
        mins = int(duration // 60)
        secs = duration % 60
        summary_lines.append(f"{stage:<35} | {mins:02d}m {secs:05.2f}s")

    summary_lines.append("=" * 50)
    summary_text = "\n".join(summary_lines)

    print("\n" + summary_text + "\n")
    with open(benchmark_file_path, "w", encoding="utf-8") as f:
        f.write(summary_text)

    print(f"Benchmarks saved to: {benchmark_file_path}")


def run_pipeline(from_stage=1, viewer=True, port=8050):
    """Run the pipeline on config.MASTER_DIR. Returns the stage 5 results (see run_post_concat_analysis)."""
    # Imported here, after the overrides: every stage module copies its settings from config at import.
    from .stage1_preprocess import run_stage1_preprocess_parallel
    from .stage2_segmentation import run_stage2_cellpose_serial
    from .stage3_features import run_stage3_postprocess_parallel
    from .stage4_concat import run_concat_stage
    from .analysis import run_post_concat_analysis

    pipeline_benchmarks = {}
    total_pipeline_start = time.time()

    for n, label, func in ((1, "Stage 1: Preprocessing", run_stage1_preprocess_parallel),
                           (2, "Stage 2: Cellpose Segmentation", run_stage2_cellpose_serial),
                           (3, "Stage 3: Feature Extraction", run_stage3_postprocess_parallel),
                           (4, "Stage 4: Concatenation", run_concat_stage)):
        if n < from_stage:
            continue
        t_step = time.time()
        func()
        pipeline_benchmarks[label] = time.time() - t_step

    t_step = time.time()
    res = run_post_concat_analysis()
    pipeline_benchmarks["Stage 5: Projections & Analysis"] = time.time() - t_step
    # volcano plots + raw CSV export (VOLCANO_ENABLED / RAW_EXPORT_ENABLED)
    t_step = time.time()
    run_extra_outputs()
    pipeline_benchmarks["Volcano plots + raw export"] = time.time() - t_step
    pipeline_benchmarks["Total Pipeline Time"] = time.time() - total_pipeline_start
    write_benchmarks(pipeline_benchmarks, res["basename"])

    if viewer:
        from .dash_viewer import make_plot_frame, run_dash_interactive_viewer
        df_plot = make_plot_frame(res["final_df_avg"], res["feature_cols"], res["raw_intensities_df"])
        run_dash_interactive_viewer(df_plot, "Individual Sample Type", color_map=res["custom_colors"],
                                    legend_order=res["LEGEND_ORDER"], master_dir=str(config.MASTER_DIR),
                                    port=port, bonsai_segments=res["bonsai_segments"])
    return res


def _parse_value(text):
    """--set values are JSON (true, 3, 0.5, null, [1, 2], "x"), anything else is taken as text."""
    try:
        return json.loads(text)
    except ValueError:
        return text


def build_parser():
    p = argparse.ArgumentParser(
        prog="tanuki",
        description="Single-cell MALDI-MSI + optical microscopy pipeline (registration, Cellpose-SAM "
                    "segmentation, per-cell features, embeddings, clustering, Bonsai, volcano plots, viewer).")
    p.add_argument("--input_dir", metavar="DIR",
                   help="run folder with one sub-folder per well (MASTER_DIR; default ./data)")
    p.add_argument("--csv_path", metavar="CSV",
                   help="mass list (CSV_PATH; default <input_dir>/BIGmatched_peaks_raw.csv)")
    p.add_argument("--config", action="append", default=[], metavar="FILE",
                   help="settings file (.json, .toml, or .py with UPPER_CASE names); may be repeated, "
                        "later files win")
    p.add_argument("--set", action="append", default=[], metavar="NAME=VALUE", dest="overrides",
                   help="override one setting, e.g. --set COMBAT_ENABLED=true --set N_CLUSTERS=8 "
                        "(values are read as JSON, otherwise as text); wins over --config")
    p.add_argument("--from-stage", type=int, default=1, choices=[1, 2, 3, 4, 5],
                   help="start at this stage; 5 = only the post-concat analysis, on the existing "
                        "concatenated tables (default 1; stages 1-3 skip wells that are already done)")
    p.add_argument("--no-viewer", action="store_true", help="do not start the Dash viewer at the end")
    p.add_argument("--port", type=int, default=8050, help="first port tried for the Dash viewer (8050)")
    p.add_argument("--show-config", action="store_true", help="print the resolved settings and exit")
    p.add_argument("--version", action="version", version=f"tanuki {__version__}")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    values = {}
    for f in args.config:
        values.update(config.load_settings_file(f))
    for item in args.overrides:
        name, sep, val = item.partition("=")
        if not sep:
            raise SystemExit(f"--set expects NAME=VALUE, got {item!r}")
        values[name.strip()] = _parse_value(val)
    if args.input_dir:
        values["MASTER_DIR"] = args.input_dir
    if args.csv_path:
        values["CSV_PATH"] = args.csv_path
    try:
        config.apply_overrides(values)
    except KeyError as e:
        raise SystemExit(e.args[0])

    if args.show_config:
        for name, value in config.settings().items():
            print(f"{name} = {value!r}")
        return None
    if not config.MASTER_DIR.is_dir():
        raise SystemExit(f"MASTER_DIR does not exist: {config.MASTER_DIR}  (set it with --input_dir)")
    if not Path(config.CSV_PATH).is_file():
        raise SystemExit(f"mass list not found: {config.CSV_PATH}  (set it with --csv_path)")
    return run_pipeline(from_stage=args.from_stage, viewer=not args.no_viewer, port=args.port)


if __name__ == "__main__":
    main()
