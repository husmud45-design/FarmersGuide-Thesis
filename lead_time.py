"""
lead_time.py — RQ5: how early in the season can a useful prediction be issued?

Retrains at successive season-truncation points and traces error against lead
time. Uses only data already on disk; no new satellite processing required.

The season window is 15 October to 15 June in ten-day composites, so
composite index t ends approximately at:
    t=6  -> mid December    t=12 -> mid February
    t=9  -> mid January     t=15 -> end March
    t=18 -> mid May         t=23 -> mid June (full season)

Usage
-----
    python3.11 lead_time.py --data data/district_balanced.npz --protocol loso
    python3.11 lead_time.py --data data/district_dataset.npz  --protocol lopo
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import date, timedelta
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SEASON_START = date(2000, 10, 15)
COMPOSITE_DAYS = 10


def composite_end_date(t: int) -> str:
    d = SEASON_START + timedelta(days=(t + 1) * COMPOSITE_DAYS)
    return d.strftime("%d %b")


def run_one(data, protocol, t, out_dir, model, epochs):
    """Invoke the existing trainer at a truncation point and read its summary."""
    tag = f"{protocol}_t{t}"
    cmd = [sys.executable, "train_district.py",
           "--data", str(data), "--protocol", protocol,
           "--model", model, "--epochs", str(epochs),
           "--early-season", str(t), "--out-dir", str(out_dir)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    summary = Path(out_dir) / f"summary_{tag}.csv"
    if not summary.exists():
        print(f"   [fail] t={t}\n{res.stdout[-600:]}\n{res.stderr[-600:]}")
        return None
    df = pd.read_csv(summary)
    df["t"] = t
    df["through"] = composite_end_date(t)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/district_balanced.npz")
    ap.add_argument("--protocol", choices=["loso", "lopo"], default="loso")
    ap.add_argument("--steps", type=int, nargs="+",
                    default=[6, 9, 12, 15, 18, 21, 24],
                    help="truncation points, in composites from season start")
    ap.add_argument("--model", default="all")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--out-dir", default="results/leadtime")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    frames = []
    for t in args.steps:
        print(f"[run] truncating at composite {t:2d} "
              f"(season through ~{composite_end_date(t)})")
        df = run_one(args.data, args.protocol, t, out, args.model, args.epochs)
        if df is not None:
            frames.append(df)
            best = df.loc[df.MAE.idxmin()]
            print(f"       best: {best.model} MAE {best.MAE:.3f}")

    if not frames:
        raise SystemExit("No runs completed.")

    res = pd.concat(frames, ignore_index=True)
    res.to_csv(out / f"leadtime_{args.protocol}.csv", index=False)

    piv = res.pivot(index="t", columns="model", values="MAE")
    print(f"\n=== Lead-time MAE, {args.protocol.upper()} ===")
    print(piv.round(3).to_string())

    # ---- figure ----
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({"font.family": "serif", "font.size": 10,
                         "axes.titleweight": "bold", "figure.dpi": 150})
    colours = {"climatology": "#9aa5b1", "peak_ndvi": "#1f3a5f",
               "gbr_phenology": "#2d6a4f", "cnn": "#c0392b"}
    labels = {"climatology": "Climatology", "peak_ndvi": "Peak NDVI",
              "gbr_phenology": "GBR phenology", "cnn": "Temporal CNN"}

    fig, ax = plt.subplots(figsize=(8, 4.4))
    for m in piv.columns:
        ax.plot(piv.index, piv[m], "-o", ms=5, lw=1.9,
                color=colours.get(m, "#555"), label=labels.get(m, m))
    if "climatology" in piv.columns:
        ax.axhline(piv["climatology"].mean(), ls="--", lw=1,
                   color="#9aa5b1", zorder=0)

    ax.set_xticks(list(piv.index))
    ax.set_xticklabels([f"{t}\n{composite_end_date(t)}" for t in piv.index],
                       fontsize=8.5)
    ax.set_xlabel("Season truncated at composite (approximate end date)")
    ax.set_ylabel("MAE (t/ha), lower is better")
    ax.set_title(f"Predictive error against lead time "
                 f"({'unseen season' if args.protocol == 'loso' else 'unseen province'})")
    ax.legend(fontsize=8.5)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig_path = Path("reports/figures") / f"fig5_10_leadtime_{args.protocol}.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_path, bbox_inches="tight")
    print(f"\nFigure: {fig_path}")

    # ---- how early does the satellite signal saturate? ----
    sat = [m for m in piv.columns if m != "climatology"]
    if sat:
        best_full = piv.loc[piv.index.max(), sat].min()
        print("\n=== Lead-time summary ===")
        for t in piv.index:
            b = piv.loc[t, sat].min()
            pct = 100 * (b - best_full) / best_full
            print(f"  through {composite_end_date(t):>7s}: best MAE {b:.3f} "
                  f"({pct:+.1f}% vs full season)")


if __name__ == "__main__":
    main()
