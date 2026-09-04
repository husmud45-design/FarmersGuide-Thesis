"""
district_dataset.py — assemble supervised examples from the GEE district export.

Input
-----
  fg_district_<season>.csv   one row per district x 10-day composite,
                             columns <BAND>_<stat> from reduceRegions
  cfs_panel_joined_to_boundaries.csv   labels, already name-harmonised

Output
------
  X : (N, T, F)  float32   T=24 composites, F = bands x stats
  y : (N,)       float32   yield per hectare PLANTED (see thesis 3.3.4)
  meta : DataFrame with district, province, season, quality fields

Design notes
------------
Column names from Earth Engine's combined reducer are not fully predictable
(percentile naming in particular varies by API version), so the feature
schema is DISCOVERED from the file rather than hardcoded, then frozen and
persisted so train/test use an identical ordering. Run with --inspect to
print what was found before trusting anything downstream.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

# MODIS pipeline (fg_modis_*.csv): 8-day composites, 14 bands incl. thermal.
BANDS_MODIS = ["RED", "NIR", "BLUE", "GREEN", "SWIR1", "SWIR2", "SWIR3",
               "NDVI", "EVI", "NDWI", "GCVI", "LSWI", "LST_DAY", "LST_NIGHT"]
# Legacy Sentinel-2 pipeline (fg_district_*.csv): 10-day composites.
BANDS_S2 = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B11", "B12",
            "NDVI", "EVI", "NDRE", "GCI", "NDWI"]

TIMESTEPS_MODIS = 30
TIMESTEPS_S2 = 24

SEASON_FILE_TO_LABEL = {
    "2012_13": "2012/13", "2013_14": "2013/14", "2014_15": "2014/15",
    "2016_17": "2016/17", "2017_18": "2017/18", "2018_19": "2018/19",
    "2020_21": "2020/21", "2021_22": "2021/22", "2022_23": "2022/23",
    "2023_24": "2023/24", "2024_25": "2024/25",
}

# Set by detect_pipeline() at load time.
BANDS = BANDS_MODIS
N_TIMESTEPS = TIMESTEPS_MODIS


def detect_pipeline(export_dir: Path) -> str:
    """MODIS and Sentinel-2 exports use different prefixes and band sets."""
    global BANDS, N_TIMESTEPS
    if list(export_dir.glob("fg_modis2_*.csv")) or list(export_dir.glob("fg_modis_*.csv")):
        BANDS, N_TIMESTEPS = BANDS_MODIS, TIMESTEPS_MODIS
        return "modis"
    if list(export_dir.glob("fg_district_*.csv")):
        BANDS, N_TIMESTEPS = BANDS_S2, TIMESTEPS_S2
        return "sentinel2"
    raise SystemExit(
        f"No exports found in {export_dir}. Expected fg_modis_*.csv "
        "(MODIS) or fg_district_*.csv (Sentinel-2).")


def export_glob(pipeline: str) -> str:
    if pipeline != "modis":
        return "fg_district_*.csv"
    return "fg_modis2_*.csv"  # v2 mask; v1 exports are superseded


def strip_prefix(stem: str) -> str:
    for pre in ("fg_modis2_", "fg_modis_", "fg_district_"):
        if stem.startswith(pre):
            return stem[len(pre):]
    return stem


def normalise(name) -> str:
    return re.sub(r"[^a-z]", "", str(name).strip().lower())


# --------------------------------------------------------------------------
# Schema discovery
# --------------------------------------------------------------------------

def discover_feature_columns(df: pd.DataFrame) -> list[str]:
    """Find <band>_<stat> columns, ordered band-major then stat.

    Frozen ordering matters: the model's channel axis must mean the same
    thing at train and inference time. Sorting is explicit, never reliant
    on dict or file column order.
    """
    feats: list[str] = []
    for band in BANDS:
        matches = [c for c in df.columns
                   if c == band or c.startswith(f"{band}_")]
        # exclude other bands that share a prefix (B1 vs B11/B12)
        matches = [c for c in matches
                   if c == band or re.fullmatch(rf"{band}_[A-Za-z0-9]+", c)]
        feats.extend(sorted(matches))
    return feats


def inspect(export_dir: Path) -> None:
    pipeline = detect_pipeline(export_dir)
    files = sorted(export_dir.glob(export_glob(pipeline)))
    print(f"Pipeline        : {pipeline}")
    df = pd.read_csv(files[0])
    print(f"File            : {files[0].name}")
    print(f"Rows            : {len(df)}")
    print(f"Columns         : {len(df.columns)}")
    print(f"Districts       : {df['DISTRICT'].nunique()}")
    print(f"Timesteps       : {sorted(df['t_index'].unique())[:30]}")
    feats = discover_feature_columns(df)
    print(f"Feature columns : {len(feats)}")
    for band in BANDS:
        got = [f for f in feats if f == band or f.startswith(f"{band}_")]
        print(f"   {band:6s} {len(got):2d}  {got[:8]}")
    missing = [b for b in BANDS
               if not any(f == b or f.startswith(f"{b}_") for f in feats)]
    if missing:
        print(f"\n!! BANDS WITH NO COLUMNS: {missing}")
    print(f"\nn_scenes  min={df['n_scenes'].min()} "
          f"median={df['n_scenes'].median()} max={df['n_scenes'].max()}")
    empty = (df['n_scenes'] == 0).mean()
    print(f"composites with zero scenes: {empty:.1%}")


# --------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------

def season_key_from_path(path: Path) -> str | None:
    """Extract the season key from a filename.

    Handles both whole-country exports (fg_district_2024_25.csv) and
    per-province splits (fg_district_2024_25_Lusaka.csv, and the
    hyphen-stripped fg_district_2024_25_NorthWestern.csv).
    """
    m = re.match(r"^(\d{4}_\d{2})", strip_prefix(path.stem))
    return m.group(1) if m else None


def load_export(export_dir: Path) -> tuple[pd.DataFrame, list[str]]:
    frames, feats = [], None
    pipeline = detect_pipeline(export_dir)
    print(f"Pipeline detected: {pipeline}  "
          f"({len(BANDS)} bands, {N_TIMESTEPS} timesteps)\n")
    paths = sorted(export_dir.glob(export_glob(pipeline)))
    if not paths:
        raise SystemExit(f"No fg_district_*.csv found in {export_dir}")

    for path in paths:
        season_key = season_key_from_path(path)
        if season_key not in SEASON_FILE_TO_LABEL:
            print(f"[skip] {path.name}: unrecognised season")
            continue
        df = pd.read_csv(path)
        df["season"] = SEASON_FILE_TO_LABEL[season_key]
        if feats is None:
            feats = discover_feature_columns(df)
        else:
            got = discover_feature_columns(df)
            if got != feats:
                raise SystemExit(
                    f"{path.name} has a different feature schema.\n"
                    f"  expected {len(feats)} cols, got {len(got)}\n"
                    f"  only in expected: {sorted(set(feats) - set(got))}\n"
                    f"  only in this file: {sorted(set(got) - set(feats))}"
                )
        frames.append(df)
        print(f"[ok] {path.name}: {len(df)} rows")

    if not frames:
        raise SystemExit(f"No usable exports in {export_dir}")

    export = pd.concat(frames, ignore_index=True)

    # Province splits must not overlap; a duplicated district-season-timestep
    # would silently double-weight that district in training.
    dup = export.duplicated(subset=["DISTRICT", "season", "t_index"]).sum()
    if dup:
        print(f"[warn] {dup} duplicate district-season-timestep rows; keeping first")
        export = export.drop_duplicates(subset=["DISTRICT", "season", "t_index"],
                                        keep="first")

    n_by_season = export.groupby("season")["DISTRICT"].nunique()
    print("\nDistricts per season after merge:")
    for s, k in n_by_season.items():
        flag = "" if k >= 70 else "   <-- LOW, check for missing province files"
        print(f"  {s}  {k}{flag}")

    return export, feats


def build_sequences(export: pd.DataFrame, feats: list[str]):
    """Pivot long export into (N, T, F) cubes keyed by district-season."""
    export = export.copy()
    export["district_key"] = export["DISTRICT"].map(normalise)
    export["t_index"] = export["t_index"].astype(int)

    keys = (export[["district_key", "DISTRICT", "PROVINCE", "season"]]
            .drop_duplicates()
            .sort_values(["season", "PROVINCE", "DISTRICT"])
            .reset_index(drop=True))

    n, t, f = len(keys), N_TIMESTEPS, len(feats)
    X = np.full((n, t, f), np.nan, dtype=np.float32)
    scenes = np.zeros((n, t), dtype=np.float32)

    index = {(r.district_key, r.season): i for i, r in keys.iterrows()}
    for row in export.itertuples(index=False):
        i = index.get((row.district_key, row.season))
        ti = row.t_index
        if i is None or not (0 <= ti < t):
            continue
        X[i, ti, :] = [getattr(row, c, np.nan) for c in feats]
        scenes[i, ti] = getattr(row, "n_scenes", 0) or 0

    return X, scenes, keys


def interpolate_gaps(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Linear interpolation along time; report fraction reconstructed.

    Gaps are cloud-driven, so they are informative rather than random.
    The interpolated fraction is retained per record so Chapter 5 can check
    whether error concentrates in heavily-reconstructed districts.
    """
    X = X.copy()
    n, t, f = X.shape
    frac = np.zeros(n, dtype=np.float32)
    grid = np.arange(t)
    for i in range(n):
        miss = np.isnan(X[i]).any(axis=1)
        frac[i] = miss.mean()
        if miss.all():
            X[i] = 0.0
            continue
        for j in range(f):
            col = X[i, :, j]
            ok = ~np.isnan(col)
            if ok.sum() == 0:
                X[i, :, j] = 0.0
            elif ok.sum() < t:
                X[i, :, j] = np.interp(grid, grid[ok], col[ok])
    return X, frac


def attach_labels(keys: pd.DataFrame, panel_csv: Path,
                  target: str = "yield_planted_t_ha"):
    panel = pd.read_csv(panel_csv)
    col = "district_canon" if "district_canon" in panel.columns else "district"
    panel["district_key"] = panel[col].map(normalise)
    cols = ["district_key", "season", target, "area_planted_ha",
            "area_harvested_ha", "production_mt"]
    cols = [c for c in cols if c in panel.columns]
    merged = keys.merge(panel[cols], on=["district_key", "season"], how="left")
    return merged


def build(export_dir: Path, panel_csv: Path, target: str):
    export, feats = load_export(export_dir)
    X, scenes, keys = build_sequences(export, feats)
    X, interp_frac = interpolate_gaps(X)
    meta = attach_labels(keys, panel_csv, target)
    meta["interp_frac"] = interp_frac
    meta["mean_scenes"] = scenes.mean(axis=1)

    keep = meta[target].notna().values
    dropped = (~keep).sum()
    if dropped:
        print(f"[warn] {dropped} district-seasons have no {target} label; dropped")
        for _, r in meta[~keep].head(10).iterrows():
            print(f"       {r['DISTRICT']:20s} {r['season']}")
    X, meta = X[keep], meta[keep].reset_index(drop=True)
    y = meta[target].values.astype(np.float32)

    print(f"\nDataset: X={X.shape}  y={y.shape}")
    print(f"  seasons   {sorted(meta.season.unique())}")
    print(f"  provinces {meta.PROVINCE.nunique()}")
    print(f"  y  mean={y.mean():.2f}  sd={y.std():.2f}  "
          f"min={y.min():.2f}  max={y.max():.2f}")
    print(f"  interpolated fraction: mean={meta.interp_frac.mean():.1%}  "
          f"max={meta.interp_frac.max():.1%}")
    return X, y, meta, feats


# --------------------------------------------------------------------------
# Cross-validation splits
# --------------------------------------------------------------------------

def coverage_report(meta: pd.DataFrame) -> pd.DataFrame:
    """Cross-tabulate districts by province and season.

    With a partially-completed export, provinces differ between seasons.
    That confounds leave-one-season-out: an apparent season effect may be
    the changing provincial composition of the panel. This prints the
    structure so the trade-off is visible rather than assumed away.
    """
    tab = (meta.groupby(["PROVINCE", "season"])["DISTRICT"]
           .nunique().unstack(fill_value=0))
    print("\n=== Coverage: districts per province per season ===")
    print(tab.to_string())

    seasons = list(tab.columns)
    complete = tab.index[(tab > 0).all(axis=1)].tolist()
    partial = tab.index[~(tab > 0).all(axis=1)].tolist()

    print(f"\nProvinces present in ALL {len(seasons)} seasons: {complete}")
    if partial:
        print(f"Provinces with gaps: {partial}")
    bal = meta[meta.PROVINCE.isin(complete)]
    print(f"\n  full panel      {len(meta):4d} records, "
          f"{meta.PROVINCE.nunique()} provinces")
    print(f"  balanced subset {len(bal):4d} records, "
          f"{len(complete)} provinces")
    print("\nUse --balanced for leave-one-season-out; the unbalanced panel "
          "confounds season with provincial composition.")
    return tab


def restrict_to_balanced(X, meta):
    tab = (meta.groupby(["PROVINCE", "season"])["DISTRICT"]
           .nunique().unstack(fill_value=0))
    complete = tab.index[(tab > 0).all(axis=1)].tolist()
    keep = meta.PROVINCE.isin(complete).values
    return X[keep], meta[keep].reset_index(drop=True), complete


def leave_one_season_out(meta: pd.DataFrame):
    for season in sorted(meta.season.unique()):
        te = (meta.season == season).values
        yield season, ~te, te


def leave_one_province_out(meta: pd.DataFrame):
    for prov in sorted(meta.PROVINCE.unique()):
        te = (meta.PROVINCE == prov).values
        yield prov, ~te, te


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export-dir", type=Path, default=Path("data/gee_exports"))
    ap.add_argument("--panel", type=Path,
                    default=Path("data/cfs_panel_joined_to_boundaries.csv"))
    ap.add_argument("--target", default="yield_planted_t_ha")
    ap.add_argument("--inspect", action="store_true")
    ap.add_argument("--balanced", action="store_true",
                    help="keep only provinces present in every season")
    ap.add_argument("--out", type=Path, default=Path("data/district_dataset.npz"))
    args = ap.parse_args()

    if args.inspect:
        inspect(args.export_dir)
        return

    X, y, meta, feats = build(args.export_dir, args.panel, args.target)
    coverage_report(meta)
    if args.balanced:
        X, meta, provs = restrict_to_balanced(X, meta)
        y = meta[args.target].values.astype("float32")
        print(f"\nRestricted to balanced subset: {len(meta)} records, "
              f"provinces {provs}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, X=X, y=y)
    meta.to_csv(args.out.with_suffix(".meta.csv"), index=False)
    args.out.with_suffix(".features.json").write_text(json.dumps(feats, indent=1))
    print(f"\nSaved {args.out}")

    print("\nLOSO folds:")
    for name, tr, te in leave_one_season_out(meta):
        print(f"  {name}  train={tr.sum():4d}  test={te.sum():4d}")
    print("LOPO folds:")
    for name, tr, te in leave_one_province_out(meta):
        print(f"  {name:16s} train={tr.sum():4d}  test={te.sum():4d}")


if __name__ == "__main__":
    main()
