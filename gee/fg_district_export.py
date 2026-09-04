import argparse
import ee

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

EE_PROJECT = "farmers-guide-491313"
DRIVE_FOLDER = "farmers_guide_district"

# District boundaries. Upload the GRID3 / ZNSDI "ZMB - Operational Districts"
# shapefile (116 districts, Adm 2, Office of the Surveyor General) as an EE
# table asset and put its id here. Verified field names: DISTRICT, PROVINCE.
# GAUL is only a fallback: its district vintage is older than the CFS 2022+
# tables and will not match Chilanga / Rufunsa.
DISTRICTS_ASSET = (
    "projects/farmers-guide-491313/assets/"
    "Zambia_Administrative_Boundaries_Districts_2020_8900812281402884208"
)
FALLBACK_ASSET = "FAO/GAUL/2015/level2"

# Attribute fields carried through to the export, so the output CSV joins to
# the CFS panel on name without a second lookup. DIST_CODE is retained as a
# stable numeric key in case of further spelling drift in future seasons.
KEEP_PROPERTIES = ["DISTRICT", "PROVINCE", "DIST_CODE", "PROV_CODE"]

# Season windows. Zambian maize: planted with the first rains (Nov-Dec),
# harvested Apr-Jun. The window is deliberately wider than the crop cycle so
# the model sees bare soil before emergence and after senescence.
SEASONS = {
    "2016_17": ("2016-10-15", "2017-06-15"),
    "2017_18": ("2017-10-15", "2018-06-15"),
    "2018_19": ("2018-10-15", "2019-06-15"),
    "2020_21": ("2020-10-15", "2021-06-15"),
    "2021_22": ("2021-10-15", "2022-06-15"),
    "2022_23": ("2022-10-15", "2023-06-15"),
    "2023_24": ("2023-10-15", "2024-06-15"),
    "2024_25": ("2024-10-15", "2025-06-15"),
}

COMPOSITE_DAYS = 10          # -> 24 composites per season (override with --composite-days)
CS_THRESHOLD = 0.60          # Cloud Score+ 'cs' band cutoff
SCALE = 60                   # 60 m. A district mean covers thousands of km2, so finer
                             # sampling buys nothing and costs quota quadratically:
                             # 60 m is 4x cheaper than 30 m, 9x cheaper than 20 m.
                             # Override with --scale.

OPTICAL = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B11", "B12"]
INDICES = ["NDVI", "EVI", "NDRE", "GCI", "NDWI"]
BANDS = OPTICAL + INDICES


# --------------------------------------------------------------------------
# Image preparation
# --------------------------------------------------------------------------

def add_indices(img):
    """Vegetation and water indices. All computed from scaled reflectance."""
    ndvi = img.normalizedDifference(["B8", "B4"]).rename("NDVI")
    ndre = img.normalizedDifference(["B8", "B5"]).rename("NDRE")
    ndwi = img.normalizedDifference(["B8", "B11"]).rename("NDWI")
    gci = img.expression(
        "(NIR / RE) - 1",
        {"NIR": img.select("B8"), "RE": img.select("B5")}
    ).rename("GCI")
    evi = img.expression(
        "2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))",
        {"NIR": img.select("B8"), "RED": img.select("B4"), "BLUE": img.select("B2")}
    ).rename("EVI")
    return img.addBands([ndvi, evi, ndre, gci, ndwi])


def mask_clouds(img):
    """Cloud Score+ mask. Linked collection, so join on system:index."""
    cs = ee.Image(img.get("cs_img")).select("cs")
    return img.updateMask(cs.gte(CS_THRESHOLD))


def season_collection(start, end, region):
    s2 = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
          .filterDate(start, end)
          .filterBounds(region))
    csp = (ee.ImageCollection("GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED")
           .filterDate(start, end)
           .filterBounds(region))
    joined = ee.ImageCollection(
        ee.Join.saveFirst("cs_img").apply(
            primary=s2, secondary=csp,
            condition=ee.Filter.equals(leftField="system:index",
                                       rightField="system:index"))
    )
    return (joined
            .map(mask_clouds)
            .map(lambda i: i.divide(10000).copyProperties(i, ["system:time_start"]))
            .map(add_indices)
            .select(BANDS))


def cropland_mask():
    """
    ESA WorldCover 10 m v200, class 40 = cropland.

    Without this, a district mean is dominated by miombo woodland and the
    maize signal is buried. WorldCover is a static 2021 product, so it cannot
    track year-to-year rotation -- state this as a limitation. It is still far
    better than reducing over the whole district polygon.
    """
    return ee.ImageCollection("ESA/WorldCover/v200").first().eq(40)


# --------------------------------------------------------------------------
# Reduction
# --------------------------------------------------------------------------

def build_reducer():
    """Construct the combined reducer.

    Must be called AFTER ee.Initialize(). Every ee.Reducer.* call hits the
    API to look up its signature, so building this at module import time
    fails with 'client library not initialized' before main() ever runs.
    """
    return (ee.Reducer.mean()
            .combine(ee.Reducer.stdDev(), sharedInputs=True)
            .combine(ee.Reducer.percentile([10, 50, 90]), sharedInputs=True)
            .combine(ee.Reducer.count(), sharedInputs=True))


def composite_features(collection, districts, start_ms, t_index, crop, reducer):
    """One 10-day median composite reduced over every district."""
    start = ee.Date(start_ms)
    end = start.advance(COMPOSITE_DAYS, "day")
    window = collection.filterDate(start, end)

    composite = ee.Image(ee.Algorithms.If(
        window.size().gt(0),
        window.median(),
        ee.Image.constant([0] * len(BANDS)).rename(BANDS).selfMask()
    )).updateMask(crop)

    stats = composite.reduceRegions(
        collection=districts,
        reducer=reducer,
        scale=SCALE,
        tileScale=16,
    )
    return stats.map(lambda f: f.set({
        "t_index": t_index,
        "date": start.format("YYYY-MM-dd"),
        "n_scenes": window.size(),
    }))


def build_season(season, districts, crop, reducer):
    start, end = SEASONS[season]
    region = districts.geometry()
    coll = season_collection(start, end, region)

    start_date = ee.Date(start)
    n_steps = ee.Date(end).difference(start_date, "day").divide(COMPOSITE_DAYS).floor()
    steps = ee.List.sequence(0, n_steps.subtract(1))

    def one(i):
        i = ee.Number(i)
        t0 = start_date.advance(i.multiply(COMPOSITE_DAYS), "day")
        return composite_features(coll, districts, t0.millis(), i, crop, reducer)

    return ee.FeatureCollection(steps.map(one)).flatten() \
             .map(lambda f: f.set("season", season))


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

def load_districts():
    """Load the district asset and fail loudly if it is not what we expect."""
    try:
        fc = ee.FeatureCollection(DISTRICTS_ASSET)
        n = fc.size().getInfo()
    except Exception as exc:
        print(f"[warn] NSDI asset not readable ({exc}).")
        print("       Falling back to GAUL level 2 -- district vintage will")
        print("       NOT match CFS 2022/23 onward (no Chilanga / Rufunsa).")
        return (ee.FeatureCollection(FALLBACK_ASSET)
                .filter(ee.Filter.eq("ADM0_NAME", "Zambia")))

    props = fc.first().propertyNames().getInfo()
    missing = [p for p in KEEP_PROPERTIES if p not in props]
    if missing:
        raise SystemExit(
            f"Asset is missing expected fields {missing}.\n"
            f"Fields present: {sorted(props)}\n"
            "Fix KEEP_PROPERTIES before exporting -- a wrong field name "
            "produces a complete-looking CSV that will not join."
        )
    if n != 116:
        print(f"[warn] expected 116 districts, asset has {n}. Check the "
              "vintage before trusting the join.")
    print(f"Districts: {n}  fields: {KEEP_PROPERTIES}")
    return fc.select(KEEP_PROPERTIES)


PROVINCES = ["Central", "Copperbelt", "Eastern", "Luapula", "Lusaka",
             "Muchinga", "North-Western", "Northern", "Southern", "Western"]


def main():
    global SCALE, COMPOSITE_DAYS
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", nargs="+", default=list(SEASONS))
    ap.add_argument("--split", choices=["province", "none"], default="province",
                    help="province: one task per province per season. "
                         "A whole-country task exceeds Earth Engine's memory "
                         "ceiling at 116 districts x 24 composites.")
    ap.add_argument("--monitor", action="store_true")
    ap.add_argument("--scale", type=int, default=SCALE,
                    help="reduction scale in metres; cost scales ~1/scale^2")
    ap.add_argument("--composite-days", type=int, default=COMPOSITE_DAYS,
                    help="compositing window; 15 gives 16 timesteps instead of 24")
    args = ap.parse_args()

    SCALE, COMPOSITE_DAYS = args.scale, args.composite_days

    ee.Initialize(project=EE_PROJECT)

    if args.monitor:
        for t in ee.data.listOperations():
            meta = t.get("metadata", {})
            desc = meta.get("description", "?")
            state = meta.get("state", "?")
            print(f"{desc:45s} {state}")
            if state in ("FAILED", "CANCELLED"):
                err = t.get("error", {})
                msg = err.get("message") or meta.get("error_message") or "(no message)"
                print(f"    ERROR: {msg}")
        return

    districts = load_districts()
    crop = cropland_mask()
    reducer = build_reducer()
    n_steps = 240 // COMPOSITE_DAYS
    print(f"scale={SCALE} m  composite={COMPOSITE_DAYS} d  "
          f"({n_steps} timesteps per season)")

    n = 0
    for season in args.seasons:
        if season not in SEASONS:
            print(f"[skip] unknown season {season}")
            continue

        if args.split == "none":
            targets = [(None, districts)]
        else:
            targets = [(p, districts.filter(ee.Filter.eq("PROVINCE", p)))
                       for p in PROVINCES]

        for prov, subset in targets:
            suffix = f"_{prov.replace('-', '')}" if prov else ""
            fc = build_season(season, subset, crop, reducer)
            task = ee.batch.Export.table.toDrive(
                collection=fc,
                description=f"fg_district_{season}{suffix}",
                folder=DRIVE_FOLDER,
                fileNamePrefix=f"fg_district_{season}{suffix}",
                fileFormat="CSV",
            )
            task.start()
            n += 1
            print(f"[queued] {season}{suffix}")

    print(f"\n{n} tasks queued. Run with --monitor to track progress.")


if __name__ == "__main__":
    main()
