"""
district_join.py — reconcile CFS district names to the ZNSDI/GRID3
"ZMB - Operational Districts" boundary vintage (116 districts, Adm 2).

Two distinct problems are handled here:

1. CFS-internal spelling drift. The same district is spelled differently in
   different seasons' workbooks (Chienge / Chiengi, Chifumunabuli /
   Chifunabuli, Senga / Senga Hill). Left alone, the panel treats each
   variant as a separate district and splits that district's time series,
   which is silent and destroys the seasonal history the model trains on.

2. CFS-to-boundary orthographic mismatch. Where the survey and the Surveyor
   General disagree on spelling (Chikankata / Chikankanta, Milenge /
   Milengi, Mushindamo / Mushindano), the boundary spelling is treated as
   canonical, since it is the spatial key.

Every mapping below was verified by fuzzy match constrained to the same
province, not matched across the national name list. Do not add entries
without that constraint: Senga fuzzy-matches Senanga (Western) more closely
than Senga Hill (Northern) nationally.
"""
from __future__ import annotations

import re
import pandas as pd

# CFS spelling -> ZNSDI canonical spelling. Keys are normalised form.
CROSSWALK = {
    "chienge":       "Chiengi",        # CFS uses both spellings
    "chifumunabuli": "Chifunabuli",    # CFS uses both spellings
    "senga":         "Senga Hill",     # CFS uses both spellings
    "chikankata":    "Chikankanta",    # CFS vs Surveyor General
    "milenge":       "Milengi",        # CFS vs Surveyor General
    "mushindamo":    "Mushindano",     # CFS vs Surveyor General
}

PROVINCE_FIXES = {
    "n/western": "North-Western",
    "north western": "North-Western",
    "northwestern": "North-Western",
}


def normalise(name: str) -> str:
    """Lowercase, strip everything that is not a letter."""
    return re.sub(r"[^a-z]", "", str(name).strip().lower())


def canonical_district(name: str) -> str:
    key = normalise(name)
    return CROSSWALK.get(key, str(name).strip().title())


def attach_canonical(df: pd.DataFrame,
                     district_col: str = "district",
                     province_col: str = "province") -> pd.DataFrame:
    out = df.copy()
    out["district_canon"] = out[district_col].map(canonical_district)
    out["district_key"] = out["district_canon"].map(normalise)
    if province_col in out.columns:
        out[province_col] = (out[province_col].astype(str).str.strip()
                             .str.lower().replace(PROVINCE_FIXES)
                             .str.title()
                             .replace({"North-Western": "North-Western"}))
    return out


def load_boundaries(path: str) -> pd.DataFrame:
    b = pd.read_excel(path)
    b = b.rename(columns={"DISTRICT": "district_canon", "PROVINCE": "province",
                          "DIST_CODE": "dist_code", "PROV_CODE": "prov_code"})
    b["district_key"] = b["district_canon"].map(normalise)
    return b[["district_canon", "province", "dist_code", "prov_code",
              "district_key", "Area_km"]]


def validate_join(panel: pd.DataFrame, boundaries: pd.DataFrame) -> dict:
    """Report both directions of the join. Raises nothing; caller decides."""
    p = set(panel["district_key"])
    b = set(boundaries["district_key"])
    report = {
        "panel_districts": len(p),
        "boundary_districts": len(b),
        "matched": len(p & b),
        "panel_only": sorted(p - b),
        "boundary_only": sorted(b - p),
    }
    report["coverage_pct"] = round(100 * len(p & b) / max(len(p), 1), 1)
    return report


if __name__ == "__main__":
    import sys

    BOUNDARY_XLSX = ("/mnt/user-data/uploads/"
                     "Zambia_Administrative_Boundaries_Districts_2020"
                     "_4991381288370377520.xlsx")
    PANEL_CSV = "/mnt/user-data/outputs/cfs_maize_district_panel_2012-2025.csv"
    # MODIS covers every season in the CFS record, so the joined panel is no
    # longer restricted to the Sentinel-2 window.
    MODEL_SEASONS = None

    bnd = load_boundaries(BOUNDARY_XLSX)
    panel = pd.read_csv(PANEL_CSV)
    panel = attach_canonical(panel)

    model = panel if MODEL_SEASONS is None else panel[panel.season.isin(MODEL_SEASONS)]
    rep = validate_join(model, bnd)

    print("=== Join validation: full CFS panel ===")
    for k in ("panel_districts", "boundary_districts", "matched", "coverage_pct"):
        print(f"  {k:22s} {rep[k]}")
    if rep["panel_only"]:
        print(f"  UNMATCHED in panel     {rep['panel_only']}")
    if rep["boundary_only"]:
        print(f"  no CFS record          {rep['boundary_only']}")

    merged = model.merge(bnd, on="district_key", how="inner",
                         suffixes=("", "_bnd"))
    print(f"\n  merged records         {len(merged)} of {len(model)}")

    # Districts whose series were previously split by spelling drift
    print("\n=== Districts recovered from CFS spelling drift ===")
    for key, canon in CROSSWALK.items():
        rows = panel[panel.district_key == normalise(canon)]
        variants = sorted(rows.district.unique())
        if len(variants) > 1:
            print(f"  {canon:14s} was split across {variants} "
                  f"-> {len(rows)} records reunited")

    out = merged.sort_values(["season", "province", "district_canon"])
    out.to_csv("/mnt/user-data/outputs/cfs_panel_joined_to_boundaries.csv",
               index=False)
    print(f"\nWrote joined panel: {len(out)} records")
    sys.exit(0 if not rep["panel_only"] else 1)
