# Farmers Guide

District-level maize yield prediction for Zambia from Sentinel-2 time series
supervised by the national Crop Forecast Survey.

MSc thesis project, Montpellier Business School. Research code.

---

## What this does

Zambia's official maize yield record is published once a year by the Crop
Forecast Survey (CFS), an enumerated field survey run by the Ministry of
Agriculture and the Zambia Statistics Agency. Between publications there is no
update. Satellite imagery is continuous, free, and covers every district.

This project asks whether the historical survey record contains enough signal
to calibrate a satellite-based estimator, and evaluates how such a model
generalises to seasons and regions it has not observed.

**Scale of prediction is the district, not the field.** The finest yield label
that is both public and national is the CFS district figure, so the model is
identified at district resolution. No claim is made about farm-level accuracy.

---

## Main result

Two cross-validation protocols disagree, and the disagreement is the finding.

| Protocol | Climatology | Best model | Difference |
|---|---|---|---|
| Leave-one-season-out (unseen season, known districts) | 0.518 t/ha | 0.497 t/ha | −4.1%, p = 0.530 |
| Leave-one-province-out (unseen districts) | 0.770 t/ha | 0.628 t/ha | **−18.3%**, p = 0.063 |

Predicting a district you already have history for adds nothing measurable over
that district's historical mean. Predicting a district you have no history for
is where satellite observation earns its place — and 96% of that advantage is
available by late December, roughly six months before harvest.

The practical reading: this substitutes for absent survey coverage rather than
improving on coverage that exists. Newly subdivided districts, districts missed
in an enumeration round, and periods between annual publication cycles.

---

## The label problem

CFS publishes yield as production ÷ area **harvested**. Area planted but
abandoned before harvest is removed from the denominator rather than entering it
as a low value.

In the 2023/24 El Niño drought, 59% of national planted area was abandoned. The
published national median yield fell only to 1.90 t/ha. Measured per hectare
**planted** it collapsed to 0.78 t/ha. In Chongwe district, where 92% of the
planted crop was abandoned, the published yield of 3.35 t/ha stands *above* that
district's long-run average.

A satellite observes the planted field all season. Abandonment is spectrally
visible as failing vegetation but invisible in the published denominator. A
model supervised on yield-per-harvested-hectare is asked to predict a quantity
whose denominator it cannot see, and the misalignment is largest in exactly the
seasons that matter for food security.

This project therefore uses **yield per hectare planted** as the supervision
target.

---

## Data

| Source | Provider | Access |
|---|---|---|
| Crop Forecast Survey crop tables | MoA / ZamStats | Published annually |
| Sentinel-2 L2A surface reflectance | ESA Copernicus | Earth Engine catalogue |
| Cloud Score+ | Google | Earth Engine catalogue |
| ESA WorldCover v200 | ESA | Earth Engine catalogue |
| Operational district boundaries | Office of the Surveyor General | [ZNSDI](https://znsdi.szi.gov.zm) / [GRID3](https://data.grid3.org) |

All public. Nothing here requires licensed or restricted data.

**Ground-truth panel:** 927 district-season records across 11 seasons
(2012/13–2024/25), parsed from eleven workbooks with mutually incompatible
schemas, harmonised to the 116-district ZNSDI boundary vintage, and validated
against the independent ZamStats provincial production series — exact agreement
in six of nine comparable seasons, with documented causes for the rest.

**Satellite panel:** 218 district-seasons across 4 seasons and 7 provinces.
Sentinel-2 L2A coverage over Zambia is unreliable before late 2018, and the
available Earth Engine compute allocation was exhausted before the full export
completed.

---

## Pipeline

```
gee/fg_district_export.py        Sentinel-2 → per-district feature tables
gee/fg_district_modis_export.py  MODIS pipeline (implemented, not executed)
gee/monitor.py                   batch export monitoring
district_join.py                 boundary harmonisation + external validation
district_dataset.py              dataset assembly, CV split construction
train_district.py                baselines, temporal CNN, evaluation
lead_time.py                     season-truncation / lead-time analysis
```

All image reduction happens server-side in Earth Engine via `reduceRegions`;
only compact feature tables leave the platform. Features are 24 ten-day
composites × 14 bands × 6 distributional statistics per district-season.

### Quick start

```bash
pip install -r requirements.txt
earthengine authenticate

python gee/fg_district_export.py --seasons 2024_25   # export features
python district_dataset.py --inspect                 # verify schema
python district_dataset.py --balanced                # build dataset
python train_district.py --protocol loso             # train and evaluate
```

---

## Method notes

**Why not a spatial CNN.** A district-level label provides one target for the
whole district. It carries no information about which sub-areas contributed
what, so it cannot supervise spatial filters over the district interior. The
spatial dimension is summarised distributionally and learning capacity directed
at the temporal axis — following the county-level lineage of
[You et al. (2017)](https://doi.org/10.1609/aaai.v31i1.11172).

**Why not random cross-validation.** Districts within a season share weather;
neighbouring districts share soils. Random splits leak exactly the information
the model is meant to infer. Both protocols here withhold entire blocks — a
season, or a province — following
[Roberts et al. (2017)](https://doi.org/10.1111/ecog.02881) and
[Ploton et al. (2020)](https://doi.org/10.1038/s41467-020-18321-y).

**Why gradient boosting beats the CNN.** At 150–218 training records the neural
model has ample capacity to fit training-set idiosyncrasy. It records negative
R² in four of seven provincial folds. Hand-constructed phenological features
with a regularised tree ensemble proved the more defensible choice at this
sample size.

---

## Scope and limitations

- **District-level only.** No claim about individual farms.
- **No anomalous season.** All four processed seasons have harvested-to-planted
  ratios of 0.81–0.91. The 2023/24 drought was not processed, so the study's
  central hypothesis — that satellite observation beats a climatological prior
  when conditions depart from normal — was not tested.
- **Low statistical power.** Four seasonal folds, seven provincial folds. The
  one improvement of practical magnitude has p = 0.063.
- **Forecast labels, not measurements.** CFS is enumerated before harvest, so
  the model predicts the official forecast rather than realised harvest.
- **Static cropland mask.** ESA WorldCover represents ~2021 conditions.
- **Compute.** The Sentinel-2 configuration here exceeds the free noncommercial
  Earth Engine allocation. The MODIS pipeline would not.

### Implemented but not validated

A ground-level image branch and a decision-level fusion module exist in
`src/farmers_guide/models/`. Neither has been trained on real data — no corpus
of ground-level Zambian maize imagery paired with yield outcomes was obtainable.
**No result in this repository or the accompanying thesis depends on them.**

---

## Reproducibility

All configuration lives in a single module; no constant is hardcoded elsewhere.
Randomness is seeded from one configured value, so CV folds are exactly
reproducible.

Two limits: Earth Engine composites are computed against a live archive subject
to reprocessing, and GPU non-determinism means retrained weights will not match
bit-for-bit. Seeded splits ensure reported metrics remain comparable.

---

## Citation

```
Mudenda, C. (2026). Farmers Guide: District-Level Maize Yield Prediction for
Zambia from Sentinel-2 Time Series and National Crop Forecast Survey Records.
MSc thesis, Montpellier Business School.
```

## Acknowledgements

Crop Forecast Survey data published by the Ministry of Agriculture and the
Zambia Statistics Agency. District boundaries from the Office of the Surveyor
General of Zambia via ZNSDI and GRID3. Sentinel-2 imagery from ESA Copernicus.
Compute provided by Google Earth Engine's noncommercial programme.

## Licence

[Choose one — MIT or Apache 2.0 are the usual choices for research code.]
