"""
train_district.py — baselines and temporal CNN for district maize yield.

Protocols (thesis 3.6):
  loso  leave-one-season-out   : extrapolation to an unseen year
  lopo  leave-one-province-out : transfer to unseen agro-ecology

Baselines are not decoration. District identity explains much of the
variance in yield, so a satellite model that cannot beat the climatological
district mean has learned nothing about the season. All models are scored on
identical folds by identical code.

Usage
-----
    python train_district.py --protocol loso
    python train_district.py --protocol lopo --model cnn
    python train_district.py --protocol loso --early-season 18
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from district_dataset import leave_one_season_out, leave_one_province_out

SEED = 42


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

def metrics(y_true, y_pred, y_train_mean) -> dict:
    """R2 is computed against the TRAINING mean, not the test-fold mean.

    Using the test-fold mean inflates R2 on small or unrepresentative folds
    and would flatter the model precisely on the hardest folds.
    """
    err = y_pred - y_true
    ss_res = float((err ** 2).sum())
    ss_tot = float(((y_true - y_train_mean) ** 2).sum())
    return {
        "n": int(len(y_true)),
        "mae": float(np.abs(err).mean()),
        "rmse": float(np.sqrt((err ** 2).mean())),
        "r2": float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        "bias": float(err.mean()),
    }


# --------------------------------------------------------------------------
# Feature engineering for the non-neural models
# --------------------------------------------------------------------------

def phenological_features(X: np.ndarray, feat_names: list[str]) -> np.ndarray:
    """Hand-built seasonal summaries: integral, peak, timing, rates."""
    idx = {n: i for i, n in enumerate(feat_names)}
    out = []
    for key in ("NDVI_mean", "EVI_mean", "NDRE_mean", "GCI_mean", "NDWI_mean"):
        if key not in idx:
            continue
        s = X[:, :, idx[key]]
        peak_t = s.argmax(axis=1)
        out.extend([
            s.sum(axis=1), s.max(axis=1), s.min(axis=1),
            s.mean(axis=1), s.std(axis=1), peak_t.astype(float),
            s.max(axis=1) - s.min(axis=1),
            np.array([s[i, :max(p, 1)].mean() for i, p in enumerate(peak_t)]),
            np.array([s[i, p:].mean() if p < s.shape[1] - 1 else s[i, -1]
                      for i, p in enumerate(peak_t)]),
        ])
    return np.column_stack(out).astype(np.float32)


def peak_ndvi(X: np.ndarray, feat_names: list[str]) -> np.ndarray:
    i = feat_names.index("NDVI_mean")
    return X[:, :, i].max(axis=1).reshape(-1, 1).astype(np.float32)


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------

def fit_climatology(meta_tr, y_tr, meta_te):
    """Per-district training mean; global mean for unseen districts."""
    d = pd.DataFrame({"k": meta_tr["district_key"].values, "y": y_tr})
    means = d.groupby("k")["y"].mean()
    g = float(y_tr.mean())
    return meta_te["district_key"].map(means).fillna(g).values.astype(np.float32)


def fit_linear(Xtr, ytr, Xte):
    from sklearn.linear_model import LinearRegression
    return LinearRegression().fit(Xtr, ytr).predict(Xte)


def fit_gbr(Xtr, ytr, Xte):
    from sklearn.ensemble import HistGradientBoostingRegressor
    m = HistGradientBoostingRegressor(
        max_iter=300, learning_rate=0.06, max_depth=4,
        l2_regularization=1.0, random_state=SEED)
    return m.fit(Xtr, ytr).predict(Xte)


def fit_cnn(Xtr, ytr, Xte, epochs=120, mc_passes=30, verbose=False):
    """1D temporal CNN with MC-dropout uncertainty.

    Capacity is deliberately small: with ~400 training records a wider net
    memorises district identity instead of learning spectral-yield structure.
    """
    import torch
    from torch import nn

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    # standardise on TRAIN ONLY
    mu = Xtr.reshape(-1, Xtr.shape[-1]).mean(0)
    sd = Xtr.reshape(-1, Xtr.shape[-1]).std(0) + 1e-6
    Xtr_s, Xte_s = (Xtr - mu) / sd, (Xte - mu) / sd
    ymu, ysd = float(ytr.mean()), float(ytr.std() + 1e-6)

    xtr = torch.tensor(Xtr_s).permute(0, 2, 1).float().to(dev)
    ytr_t = torch.tensor((ytr - ymu) / ysd).float().to(dev)
    xte = torch.tensor(Xte_s).permute(0, 2, 1).float().to(dev)

    class Net(nn.Module):
        def __init__(self, c_in):
            super().__init__()
            self.body = nn.Sequential(
                nn.Conv1d(c_in, 48, 5, padding=2), nn.BatchNorm1d(48),
                nn.ReLU(), nn.Dropout(0.20),
                nn.MaxPool1d(2),
                nn.Conv1d(48, 64, 3, padding=1), nn.BatchNorm1d(64),
                nn.ReLU(), nn.Dropout(0.20),
                nn.MaxPool1d(2),
                nn.Conv1d(64, 64, 3, padding=1), nn.BatchNorm1d(64),
                nn.ReLU(), nn.Dropout(0.20),
                nn.AdaptiveAvgPool1d(1),
            )
            self.head = nn.Sequential(
                nn.Flatten(), nn.Linear(64, 48), nn.ReLU(),
                nn.Dropout(0.30), nn.Linear(48, 1))

        def forward(self, x):
            return self.head(self.body(x)).squeeze(-1)

    net = Net(xtr.shape[1]).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    lossf = nn.SmoothL1Loss()

    n, bs = len(xtr), 32
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(n, device=dev)
        tot = 0.0
        for i in range(0, n, bs):
            b = perm[i:i + bs]
            opt.zero_grad()
            loss = lossf(net(xtr[b]), ytr_t[b])
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            tot += loss.detach().item() * len(b)
        sched.step()
        if verbose and (ep + 1) % 40 == 0:
            print(f"      epoch {ep+1:3d}  loss {tot/n:.4f}")

    # MC dropout: dropout on, batchnorm held in eval
    net.eval()
    for m in net.modules():
        if isinstance(m, torch.nn.Dropout):
            m.train()
    with torch.no_grad():
        draws = np.stack([net(xte).cpu().numpy() for _ in range(mc_passes)])
    return draws.mean(0) * ysd + ymu, draws.std(0) * ysd


# --------------------------------------------------------------------------
# Harness
# --------------------------------------------------------------------------

def run(args):
    d = np.load(args.data)
    X, y = d["X"], d["y"]
    meta = pd.read_csv(Path(args.data).with_suffix(".meta.csv"))
    feats = json.loads(Path(args.data).with_suffix(".features.json").read_text())

    if args.early_season:
        X = X[:, :args.early_season, :]
        print(f"Early-season truncation: using first {args.early_season} "
              f"of 24 composites\n")

    splitter = (leave_one_season_out if args.protocol == "loso"
                else leave_one_province_out)

    Xphen = phenological_features(X, feats)
    Xpeak = peak_ndvi(X, feats)

    rows, preds = [], []
    for name, tr, te in splitter(meta):
        ytr, yte = y[tr], y[te]
        tm = float(ytr.mean())
        fold = {"fold": name}

        p = fit_climatology(meta[tr], ytr, meta[te])
        fold["climatology"] = metrics(yte, p, tm)

        p = fit_linear(Xpeak[tr], ytr, Xpeak[te])
        fold["peak_ndvi"] = metrics(yte, p, tm)

        p = fit_gbr(Xphen[tr], ytr, Xphen[te])
        fold["gbr_phenology"] = metrics(yte, p, tm)

        if args.model in ("cnn", "all"):
            p, sd = fit_cnn(X[tr], ytr, X[te], epochs=args.epochs,
                            verbose=args.verbose)
            fold["cnn"] = metrics(yte, p, tm)
            preds.append(pd.DataFrame({
                "fold": name,
                "district": meta.loc[te, "DISTRICT"].values,
                "province": meta.loc[te, "PROVINCE"].values,
                "season": meta.loc[te, "season"].values,
                "y_true": yte, "y_pred": p, "y_sd": sd}))

        rows.append(fold)
        line = "  ".join(
            f"{k}: MAE {v['mae']:.3f} R2 {v['r2']:+.2f}"
            for k, v in fold.items() if k != "fold")
        print(f"{name:16s} {line}")

    print("\n=== Pooled across folds ===")
    models = [k for k in rows[0] if k != "fold"]
    summary = []
    for m in models:
        mae = np.mean([r[m]["mae"] for r in rows])
        rmse = np.mean([r[m]["rmse"] for r in rows])
        r2 = np.mean([r[m]["r2"] for r in rows])
        sd = np.std([r[m]["mae"] for r in rows])
        summary.append({"model": m, "MAE": mae, "MAE_sd": sd,
                        "RMSE": rmse, "R2": r2})
        print(f"  {m:16s} MAE {mae:.3f} (sd {sd:.3f})  "
              f"RMSE {rmse:.3f}  R2 {r2:+.3f}")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tag = f"{args.protocol}{'_t' + str(args.early_season) if args.early_season else ''}"
    pd.DataFrame(summary).to_csv(out / f"summary_{tag}.csv", index=False)
    with open(out / f"folds_{tag}.json", "w") as f:
        json.dump(rows, f, indent=1)
    if preds:
        pd.concat(preds).to_csv(out / f"predictions_{tag}.csv", index=False)
    print(f"\nWrote results to {out}/*_{tag}.*")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/district_dataset.npz")
    ap.add_argument("--protocol", choices=["loso", "lopo"], default="loso")
    ap.add_argument("--model", choices=["baselines", "cnn", "all"], default="all")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--early-season", type=int, default=None,
                    help="truncate to first N composites (lead-time analysis)")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("-v", "--verbose", action="store_true")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
