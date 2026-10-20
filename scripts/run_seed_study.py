#!/usr/bin/env python3
"""Run a multi-seed comparison of MISA shared-alignment settings.

Each run trains with 4 GPUs, evaluates the best checkpoint, and appends the
test metrics to a CSV. A markdown summary with mean/std is written at the end.
Existing result files are skipped, so the script can be resumed.
"""
import argparse
import csv
import re
import statistics
import subprocess
import sys
from datetime import datetime
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_DIR / "src"
TRAIN_SCRIPT = SRC_DIR / "train.py"
EVAL_SCRIPT = SRC_DIR / "evaluate.py"
LOG_DIR = PROJECT_DIR / "logs"
RESULT_DIR = PROJECT_DIR / "results"

SETTINGS = {
    "cmd": {
        "align": "cmd",
        "train_args": ["--shared_align_loss", "cmd"],
    },
    "atp_text": {
        "align": "atp",
        "train_args": ["--shared_align_loss", "atp", "--atp_variant", "text"],
    },
    "atp_all": {
        "align": "atp",
        "train_args": ["--shared_align_loss", "atp", "--atp_variant", "all"],
    },
    "atp_sim0": {
        "align": "atp",
        "train_args": [
            "--shared_align_loss", "atp", "--atp_variant", "all",
            "--sim_weight", "0",
        ],
    },
}

METRICS = ["mae", "corr", "mult_acc", "acc_pos_neg", "acc_nonneg_neg", "eval_loss"]
METRIC_PATTERNS = {
    "mae": r"^mae:\s+([0-9.eE+-]+)$",
    "corr": r"^corr:\s+([0-9.eE+-]+)$",
    "mult_acc": r"^mult_acc:\s+([0-9.eE+-]+)$",
    "acc_pos_neg": r"^Accuracy \(pos/neg\)\s+([0-9.eE+-]+)$",
    "acc_nonneg_neg": r"^Accuracy \(non-neg/neg\)\s+([0-9.eE+-]+)$",
    "eval_loss": r"^eval_loss:\s+([0-9.eE+-]+)$",
}


def parse_result(path):
    text = path.read_text(encoding="utf-8")
    values = {}
    for key, pattern in METRIC_PATTERNS.items():
        match = re.search(pattern, text, flags=re.MULTILINE)
        if not match:
            raise ValueError(f"missing {key} in {path}")
        values[key] = float(match.group(1))
    return values


def run_command(command, log_path, env):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.run(
            command,
            cwd=PROJECT_DIR,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
    return process.returncode


def format_mean_std(rows, key):
    values = [row[key] for row in rows]
    mean = statistics.mean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return f"{mean:.4f} ± {std:.4f}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default="336,337,338",
                        help="Comma-separated seeds.")
    parser.add_argument("--settings", default=",".join(SETTINGS),
                        help="Comma-separated settings: " + ", ".join(SETTINGS))
    parser.add_argument("--data", default="mosi")
    args = parser.parse_args()

    seeds = [int(item) for item in args.seeds.split(",") if item.strip()]
    selected = [item.strip() for item in args.settings.split(",") if item.strip()]
    unknown = [item for item in selected if item not in SETTINGS]
    if unknown:
        parser.error(f"unknown settings: {unknown}")

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    env = None
    import os
    env = os.environ.copy()
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    env["PYTHONUNBUFFERED"] = "1"

    rows = []
    failures = []
    total = len(selected) * len(seeds)
    completed = 0

    for setting in selected:
        config = SETTINGS[setting]
        for seed in seeds:
            completed += 1
            name = f"{args.data}_{setting}_4x4090_s{seed}"
            result_path = RESULT_DIR / f"{name}_test.txt"
            print(f"[{datetime.now():%H:%M:%S}] ({completed}/{total}) {name}", flush=True)

            if result_path.exists():
                values = parse_result(result_path)
                print(f"  skip: found {result_path}", flush=True)
            else:
                train_log = LOG_DIR / f"{name}.log"
                command = [
                    "conda", "run", "--no-capture-output", "-n", "misa",
                    "torchrun", "--standalone", "--nproc_per_node=4",
                    str(TRAIN_SCRIPT),
                    "--data", args.data,
                    "--use_cmd_sim", "true",
                    "--name", name,
                    "--seed", str(seed),
                ] + config["train_args"]
                code = run_command(command, train_log, env)
                if code != 0:
                    print(f"  training failed (exit {code}); log: {train_log}",
                          flush=True)
                    failures.append(name)
                    continue

                eval_log = LOG_DIR / f"{name}_eval.log"
                command = [
                    "conda", "run", "--no-capture-output", "-n", "misa",
                    "python", str(EVAL_SCRIPT),
                    "--data", args.data,
                    "--name", name,
                    "--shared_align_loss", config["align"],
                ]
                code = run_command(command, eval_log, env)
                if code != 0:
                    print(f"  evaluation failed (exit {code}); log: {eval_log}",
                          flush=True)
                    failures.append(name)
                    continue

                values = parse_result(result_path)

            row = {"setting": setting, "seed": seed, "name": name, **values}
            rows.append(row)
            print(
                "  mae={mae:.4f} corr={corr:.4f} mult={mult_acc:.4f} "
                "pos/neg={acc_pos_neg:.4f} nonneg/neg={acc_nonneg_neg:.4f}".format(**values),
                flush=True,
            )

    runs_csv = RESULT_DIR / "seed_study_runs.csv"
    with runs_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["setting", "seed", "name", *METRICS])
        writer.writeheader()
        writer.writerows(rows)

    summary_md = RESULT_DIR / "seed_study_summary.md"
    lines = [
        "# MISA multi-seed summary",
        "",
        f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}",
        f"Dataset: {args.data}",
        f"Seeds: {', '.join(str(seed) for seed in seeds)}",
        "",
        "| setting | n | MAE (mean ± std) | Corr | mult_acc | pos/neg acc | non-neg/neg acc |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for setting in selected:
        setting_rows = [row for row in rows if row["setting"] == setting]
        if not setting_rows:
            lines.append(f"| {setting} | 0 | - | - | - | - | - |")
            continue
        lines.append(
            f"| {setting} | {len(setting_rows)} | "
            f"{format_mean_std(setting_rows, 'mae')} | "
            f"{format_mean_std(setting_rows, 'corr')} | "
            f"{format_mean_std(setting_rows, 'mult_acc')} | "
            f"{format_mean_std(setting_rows, 'acc_pos_neg')} | "
            f"{format_mean_std(setting_rows, 'acc_nonneg_neg')} |"
        )
    summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n" + "\n".join(lines), flush=True)
    print(f"\nRuns CSV: {runs_csv}", flush=True)
    print(f"Summary: {summary_md}", flush=True)
    if failures:
        print(f"Failures: {', '.join(failures)}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
