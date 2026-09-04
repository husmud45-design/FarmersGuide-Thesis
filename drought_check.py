"""
drought_check.py — is the 2023/24 failure visible in the MODIS signal?

This is a gate, not a formality. The MODIS rebuild is worth a day of work only
if the anomalous season is spectrally distinguishable from normal ones. If
national NDVI in 2023/24 sits within the normal range, the model cannot detect
the drought and RQ3 stays null for a worse reason than sample size.

Run after downloading the exports, before building the dataset:

    python3.11 drought_check.py --export-dir ~/fg_data/modis
"""
from __future__ import annotations

import argparse
import glob
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SEASON = {f"{y}_{str(y+1)[2:]}": f"{y}/{str(y+1)[2:]}"
          for y in range(2012, 2025)}
DROUGHT = "2023/24"


def load(export_dir):
    rows = []
    for f in sorted(glob.glob(os.path.join(export_dir, "fg_modis_*.csv"))):
        m = re.search(r"fg_modis_(\d{4}_\d{2})", os.path.basename(f))
        if not m or m.group(1) not in SEASON:
            continue
        d = pd.read_csv(f, usecols=lambda c: c in (
            "DISTRICT", "PROVINCE", "t_index", "date",
            "NDVI_mean", "EVI_mean", "LSWI_mean",
            "LST_DAY_mean", "NDVI_count"))
        d["season"] = SEASON[m.group(1)]
        rows.append(d)
    if not rows:
        raise SystemExit(f"No fg_modis_*.csv in {export_dir}")
    df = pd.concat(rows, ignore_index=True)
    df = df.drop_duplicates(subset=["DISTRICT", "season", "t_index"])
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export-dir", default="~/fg_data/modis")
    ap.add_argument("--min-pixels", type=int, default=20,
                    help="drop composites reduced over fewer cropland pixels")
    args = ap.parse_args()
    d = load(os.path.expanduser(args.export_dir))

    print(f"Loaded {len(d):,} rows | {d.season.nunique()} seasons | "
          f"{d.DISTRICT.nunique()} districts\n")

    sparse = (d.NDVI_count < args.min_pixels).mean()
    print(f"Composites with <{args.min_pixels} cropland pixels: {sparse:.1%}")
    d = d[d.NDVI_count >= args.min_pixels]

    # ---- seasonal integrated NDVI, the standard productivity proxy ----
    per_dist = (d.groupby(["season", "DISTRICT"])
                  .agg(int_ndvi=("NDVI_mean", "sum"),
                       peak_ndvi=("NDVI_mean", "max"),
                       mean_lst=("LST_DAY_mean", "mean"),
                       n=("t_index", "size")).reset_index())
    per_dist = per_dist[per_dist.n >= 20]

    seas = (per_dist.groupby("season")
            .agg(int_ndvi=("int_ndvi", "median"),
                 peak=("peak_ndvi", "median"),
                 lst=("mean_lst", "median"),
                 n=("DISTRICT", "size")))

    print("\n=== Seasonal signal ===")
    print(f"{'season':>9s}{'int NDVI':>10s}{'peak':>8s}{'LST day':>9s}{'n':>5s}")
    for s, r in seas.iterrows():
        flag = "  <-- drought" if s == DROUGHT else ""
        print(f"{s:>9s}{r.int_ndvi:>10.2f}{r.peak:>8.3f}"
              f"{r.lst:>9.1f}{int(r.n):>5d}{flag}")

    if DROUGHT not in seas.index:
        raise SystemExit(f"\n{DROUGHT} not present — download it first.")

    normal = seas.drop(index=DROUGHT)
    dz = seas.loc[DROUGHT]
    print("\n=== Is the drought distinguishable? ===")
    for col, label, lower_is_worse in [("int_ndvi", "Integrated NDVI", True),
                                       ("peak", "Peak NDVI", True),
                                       ("lst", "Mean daytime LST", False)]:
        mu, sd = normal[col].mean(), normal[col].std(ddof=1)
        z = (dz[col] - mu) / sd
        rank = (seas[col] < dz[col]).sum() + 1
        direction = "below" if dz[col] < mu else "above"
        print(f"  {label:18s} {dz[col]:7.2f}  vs normal {mu:6.2f} "
              f"(sd {sd:.2f})   z = {z:+.2f}   {direction} mean, "
              f"rank {rank}/{len(seas)}")

    z_ndvi = (dz.int_ndvi - normal.int_ndvi.mean()) / normal.int_ndvi.std(ddof=1)
    print()
    if z_ndvi < -1.0:
        print("  VERDICT: the drought is clearly visible in the MODIS signal.")
        print("  The rebuild is worth doing; RQ3 can be tested properly.")
    elif z_ndvi < -0.5:
        print("  VERDICT: the drought is weakly visible. Proceed, but expect")
        print("  the model to find it a hard season and report that honestly.")
    else:
        print("  VERDICT: the drought is NOT distinguishable from normal")
        print("  seasons in integrated NDVI. This is itself a finding and must")
        print("  be reported: a spectral index cannot see abandonment that")
        print("  occurs after green-up. Check the LST and harvest-ratio")
        print("  relationship before concluding the rebuild adds nothing.")

    # ---- figure ----
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({"font.family": "serif", "font.size": 10,
                         "axes.titleweight": "bold", "figure.dpi": 150})
    INK, ACC, GRY = "#1f3a5f", "#c0392b", "#9aa5b1"
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))

    nat = d.groupby(["season", "t_index"]).NDVI_mean.mean().unstack(0)
    for s in nat.columns:
        if s == DROUGHT:
            continue
        a1.plot(nat.index, nat[s], color=GRY, lw=1.1, alpha=0.75)
    a1.plot(nat.index, nat[DROUGHT], color=ACC, lw=2.6, label=f"{DROUGHT} drought")
    a1.plot([], [], color=GRY, lw=1.1, label="Other seasons")
    a1.set_xlabel("Eight-day composite index")
    a1.set_ylabel("National mean NDVI over cropland")
    a1.set_title("Seasonal NDVI trajectory")
    a1.legend(fontsize=8.5)
    for sp in ("top", "right"):
        a1.spines[sp].set_visible(False)

    order = seas.sort_values("int_ndvi")
    cols = [ACC if s == DROUGHT else INK for s in order.index]
    a2.barh(range(len(order)), order.int_ndvi, color=cols)
    a2.set_yticks(range(len(order)))
    a2.set_yticklabels(order.index, fontsize=8.5)
    a2.set_xlabel("Median integrated NDVI across districts")
    a2.set_title("Seasonal productivity proxy, ranked")
    for sp in ("top", "right"):
        a2.spines[sp].set_visible(False)

    plt.tight_layout()
    out = "reports/figures/fig5_11_drought_signal.png"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out, bbox_inches="tight")
    print(f"\nFigure: {out}")


if __name__ == "__main__":
    main()
