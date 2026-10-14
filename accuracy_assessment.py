"""
Accuracy assessment for the polarimetric flood-change detectors.

Generalises the original per-element script to run over EVERY detector output
(Dif1/2, Rat1/2, RatDif, MonU1/2, Lin1/2, RatVV/VH, Wishart, Trace, ...),
for each land-cover class (Grassland / open_flood / Combined) and both
threshold directions (Greater / Less_than).

For each (detector, cover, condition) it sweeps `dim` thresholds between two
percentiles of the detector and records, per threshold:
    Pd, Pf, threshold, accuracy, F-score, precision
written to  <out_path>/<condition>/<cover>_<condition>_<detector>_<date>.csv
(identical column order to the original pipeline).

It then builds an AUC summary table (best direction per detector, like Table II
of the paper) and a ROC overview plot (log10 Pf vs Pd, like Fig. 8).

Metrics reproduce the original exactly; only the threshold sweep is accelerated
with np.searchsorted instead of rebuilding a full-image mask per threshold.
"""

import os
import glob
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import auc

try:
    from tqdm import tqdm                 # plain tqdm works in console and notebook
except ImportError:                       # no-op fallback if tqdm is missing
    class _NullBar:
        def __init__(self, iterable=None, **k):
            self.iterable = iterable
        def __iter__(self):
            return iter(self.iterable or [])
        def update(self, n=1):
            pass
        def set_postfix_str(self, s):
            pass
        def close(self):
            pass
    def tqdm(iterable=None, **k):
        return _NullBar(iterable, **k)

warnings.filterwarnings("ignore")

CSV_COLUMNS = ["Pd", "Pf", "threshold", "accuracy", "fscore", "precision"]


# =============================================================================
# Loading detector outputs saved by change detection
# =============================================================================
def load_detectors(results_dir, date, names=None):
    """
    Load detector arrays written by `change_detection.save_detectors`.

    Parameters
    ----------
    results_dir : str or Path    folder containing `<name>_<date>.npy`
    date : str                   the tag used when saving, e.g. '20170715'
    names : list[str] or None    specific detectors to load; None loads all
                                 `*_<date>.npy` found in the folder

    Returns
    -------
    dict[str, (H, W) array]
    """
    results_dir = str(results_dir)
    out = {}
    if names is None:
        suffix = f"_{date}.npy"
        for p in sorted(glob.glob(os.path.join(results_dir, f"*{suffix}"))):
            name = os.path.basename(p)[: -len(suffix)]
            out[name] = np.load(p)
    else:
        for name in names:
            out[name] = np.load(os.path.join(results_dir, f"{name}_{date}.npy"))
    print(f"Loaded {len(out)} detectors for {date} from {results_dir}")
    return out


# =============================================================================
# Core: one detector, one cover class, one direction
# =============================================================================
def accuracy_assessment(detector, flood_mask, no_flood_mask, dim, condition,
                        save_path=None, pct=(0.1, 93.0)):
    """
    Sweep `dim` thresholds and return a DataFrame of ROC metrics.

    Parameters
    ----------
    detector : (H, W) float array        detector output
    flood_mask, no_flood_mask : (H, W) bool arrays   reference classes
    dim : int                            number of thresholds
    condition : {'Greater', 'Less_than'} flood = detector >  t  or  detector < t
    save_path : str or Path or None      if given, write the CSV (no header)
    pct : (low, high)                    percentiles bounding the threshold sweep

    Returns
    -------
    pandas.DataFrame with columns CSV_COLUMNS
    """
    detector = np.asarray(detector)
    flood_mask = np.asarray(flood_mask, dtype=bool)
    no_flood_mask = np.asarray(no_flood_mask, dtype=bool)

    # denominators: total reference pixels (NaN detector pixels still counted here,
    # exactly as the original np.sum(flood_mask))
    sum_inf = float(flood_mask.sum())
    sum_clean = float(no_flood_mask.sum())

    # detector values at the reference pixels, NaNs dropped then sorted
    fv = detector[flood_mask]
    nv = detector[no_flood_mask]
    fv = np.sort(fv[~np.isnan(fv)])
    nv = np.sort(nv[~np.isnan(nv)])

    start = np.nanpercentile(detector, pct[0])
    stop = np.nanpercentile(detector, pct[1])
    t = np.linspace(start, stop, num=dim)

    if condition == "Greater":            # flood = values strictly greater than t
        N_det = fv.size - np.searchsorted(fv, t, side="right")
        N_false = nv.size - np.searchsorted(nv, t, side="right")
    elif condition == "Less_than":        # flood = values strictly less than t
        N_det = np.searchsorted(fv, t, side="left")
        N_false = np.searchsorted(nv, t, side="left")
    else:
        raise ValueError("condition must be 'Greater' or 'Less_than'")

    with np.errstate(divide="ignore", invalid="ignore"):
        Pd = N_det / sum_inf
        Pf = N_false / sum_clean
        t_neg = 1.0 - Pf                  # true-negative rate
        f_neg = 1.0 - Pd                  # false-negative rate
        prec = Pd / (Pd + Pf)
        rec = Pd / (Pd + f_neg)           # == Pd, kept for parity with original
        fscore = 2.0 * (prec * rec) / (prec + rec)
        acc = (Pd + t_neg) / 2.0          # balanced accuracy

    df = pd.DataFrame(
        np.column_stack([Pd, Pf, t, acc, fscore, prec]), columns=CSV_COLUMNS
    )

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path, mode="w", header=False, index=False)

    return df


# =============================================================================
# Cover-class mask construction (mirrors the original cover logic)
# =============================================================================
def cover_masks(masks, cover, open_detected_mask=None):
    """
    Build (flood, no_flood) boolean masks for a land-cover class.

    `masks` is a dict with keys:
        'grass_flood', 'open_flood', 'grass_noflood', 'open_noflood'
    `open_detected_mask` (optional) is a detector-derived open-flood mask added
    to the grassland no-flood set, as in the original Grassland branch.
    """
    gf = masks["grass_flood"].astype(bool)
    of = masks["open_flood"].astype(bool)
    gnf = masks["grass_noflood"].astype(bool)
    onf = masks["open_noflood"].astype(bool)

    if cover == "Combined":
        flood = gf | of
        no_flood = gnf | onf
    elif cover == "Grassland":
        flood = gf
        no_flood = gnf | onf | of
        if open_detected_mask is not None:
            no_flood = no_flood | open_detected_mask.astype(bool)
    elif cover == "open_flood":
        flood = of
        no_flood = gnf | onf | gf
    else:
        raise ValueError(f"unknown cover type: {cover}")
    return flood, no_flood


# =============================================================================
# Run every detector x cover x condition
# =============================================================================
def assess_all_detectors(detectors, masks, out_path, date, dim=1000,
                         conditions=("Greater", "Less_than"),
                         covers=("Grassland", "open_flood", "Combined"),
                         open_detected_mask=None, transpose=False):
    """
    Loop over all detectors and write a CSV for each (detector, cover, condition).

    Parameters
    ----------
    detectors : dict[str, (H, W) array]   e.g. output of compute_change_detectors
    masks : dict                          see cover_masks()
    out_path : str or Path                output root
    date : str                            tag used in the file names
    transpose : bool                      .T each detector before assessing
                                          (match detector/mask orientation)
    """
    out_path = str(out_path)
    total = len(detectors) * len(covers) * len(conditions)
    pbar = tqdm(total=total, desc="accuracy")
    for name, det in detectors.items():
        d = np.asarray(det).T if transpose else np.asarray(det)
        for cover in covers:
            flood, no_flood = cover_masks(masks, cover, open_detected_mask)
            for condition in conditions:
                save_path = os.path.join(
                    out_path, condition, f"{cover}_{condition}_{name}_{date}.csv"
                )
                accuracy_assessment(d, flood, no_flood, dim, condition, save_path)
                pbar.set_postfix_str(f"{name}/{cover}/{condition}")
                pbar.update(1)
    pbar.close()
    print(f"Wrote {len(detectors)} detectors x {len(covers)} covers x "
          f"{len(conditions)} conditions to {os.path.abspath(out_path)}")


# =============================================================================
# AUC from a saved CSV (proper ROC ordering: sort by Pf, integrate Pd)
# =============================================================================
def auc_from_df(df):
    """Trapezoidal AUC of the ROC implied by a metrics DataFrame."""
    order = np.argsort(df["Pf"].values)
    x = df["Pf"].values[order]
    y = df["Pd"].values[order]
    keep = ~(np.isnan(x) | np.isnan(y))
    x, y = x[keep], y[keep]
    if x.size < 2:
        return np.nan
    return auc(x, y)


def auc_summary(detectors, masks, dim=1000,
                conditions=("Greater", "Less_than"),
                covers=("Grassland", "open_flood", "Combined"),
                open_detected_mask=None, transpose=False):
    """
    Build an AUC table: for each detector and cover, keep the best direction.

    Returns
    -------
    (table, best_dir) : two DataFrames indexed by detector, columns = covers.
      `table`    holds the best AUC value; `best_dir` holds 'Greater'/'Less_than'.
    """
    rows_auc, rows_dir = {}, {}
    for name, det in tqdm(detectors.items(), total=len(detectors), desc="AUC"):
        d = np.asarray(det).T if transpose else np.asarray(det)
        auc_row, dir_row = {}, {}
        for cover in covers:
            flood, no_flood = cover_masks(masks, cover, open_detected_mask)
            best_auc, best_c = -np.inf, None
            for condition in conditions:
                df = accuracy_assessment(d, flood, no_flood, dim, condition)
                a = auc_from_df(df)
                if np.isfinite(a) and a > best_auc:
                    best_auc, best_c = a, condition
            auc_row[cover] = best_auc if np.isfinite(best_auc) else np.nan
            dir_row[cover] = best_c
        rows_auc[name] = auc_row
        rows_dir[name] = dir_row
    table = pd.DataFrame(rows_auc).T[list(covers)]
    best_dir = pd.DataFrame(rows_dir).T[list(covers)]
    return table, best_dir


# =============================================================================
# Best operating point per detector (two-stage threshold selection)
# =============================================================================
def _auc_value(df, method="roc"):
    """'roc' = PD ordered by PF (correct); 'sortboth' = original script's method."""
    if method == "sortboth":
        x = np.sort(df["Pf"].values)
        y = np.sort(df["Pd"].values)
        keep = ~(np.isnan(x) | np.isnan(y))
        return auc(x[keep], y[keep]) if keep.sum() >= 2 else np.nan
    return auc_from_df(df)


DEFAULT_FLOORS = {
    "Grassland":  dict(pf=0.30, pd=0.60, acc=0.60),
    "open_flood": dict(pf=0.15, pd=0.70, acc=0.70),
    "Combined":   dict(pf=0.15, pd=0.70, acc=0.70),
}


def best_operating_points(detectors, masks, dim=1000,
                          conditions=("Greater", "Less_than"),
                          covers=("Grassland", "open_flood", "Combined"),
                          open_detected_mask=None, transpose=False,
                          auc_gate=0.6, auc_method="roc",
                          floors=None, sort_by="accuracy"):
    """
    Two-stage selection (reproduces the paper's get_best_parameters):

    1. AUC screen  -> drop detectors with AUC <= auc_gate.
    2. Operating point -> among thresholds clearing the per-cover floors
       (PF / PD / ACC), keep the one with the highest `sort_by` metric.

    Returns a tidy DataFrame: Detector, cover, condition, AUC, THR, PD, PF, ACC, F1, PREC.
    """
    floors = floors or DEFAULT_FLOORS
    combos = [(n, c, cond) for n in detectors for c in covers for cond in conditions]
    rows = []
    for name, cover, condition in tqdm(combos, desc="best-params"):
        d = np.asarray(detectors[name]).T if transpose else np.asarray(detectors[name])
        flood, no_flood = cover_masks(masks, cover, open_detected_mask)
        df = accuracy_assessment(d, flood, no_flood, dim, condition)
        a = _auc_value(df, auc_method)
        if not np.isfinite(a) or a <= auc_gate:
            continue
        fl = floors[cover]
        sel = df[(df["Pf"] < fl["pf"]) & (df["Pd"] > fl["pd"]) & (df["accuracy"] > fl["acc"])]
        if sel.empty:                       # no threshold clears the floors -> skip
            continue
        best = sel.sort_values(sort_by, ascending=False).iloc[0]
        rows.append({"Detector": name, "cover": cover, "condition": condition,
                     "AUC": round(float(a), 4), "THR": float(best["threshold"]),
                     "PD": float(best["Pd"]), "PF": float(best["Pf"]),
                     "ACC": float(best["accuracy"]), "F1": float(best["fscore"]),
                     "PREC": float(best["precision"])})
    return pd.DataFrame(rows, columns=["Detector", "cover", "condition", "AUC",
                                       "THR", "PD", "PF", "ACC", "F1", "PREC"])


# =============================================================================
# ROC overview plot (log10 Pf vs Pd) for one cover class
# =============================================================================
def plot_roc_overview(detectors, masks, cover="Combined", dim=1000,
                      conditions=("Greater", "Less_than"),
                      open_detected_mask=None, transpose=False,
                      save_path=None, title=None):
    """Plot the best-direction ROC of every detector for one cover class."""
    flood, no_flood = cover_masks(masks, cover, open_detected_mask)
    plt.figure(figsize=(10, 7))
    for name, det in detectors.items():
        d = np.asarray(det).T if transpose else np.asarray(det)
        best_auc, best_df = -np.inf, None
        for condition in conditions:
            df = accuracy_assessment(d, flood, no_flood, dim, condition)
            a = auc_from_df(df)
            if np.isfinite(a) and a > best_auc:
                best_auc, best_df = a, df
        if best_df is None:
            continue
        order = np.argsort(best_df["Pf"].values)
        x = best_df["Pf"].values[order]
        y = best_df["Pd"].values[order]
        with np.errstate(divide="ignore"):
            plt.plot(np.log10(x), y, label=f"{name} (AUC={best_auc:.3f})")
    plt.title(title or f"ROC overview - {cover}", size=16)
    plt.xlabel("log10 False Alarm Rate (Pf)", size=13)
    plt.ylabel("Detection Rate (Pd)", size=13)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"Saved ROC overview -> {save_path}")
    return plt.gcf()


# =============================================================================
# Runnable test  --  all detectors on the 100x100 crop with SYNTHETIC masks
# =============================================================================
if __name__ == "__main__":
    import change_detection as cd

    DATA_ROOT = r"D:\water-hyacinth-polsar-lake-victoria\data"
    DATE_REF, DATE_TEST = "20171022", "20181228"
    WIN, CROP = (7, 7), 100
    OUT = os.path.join(os.path.dirname(__file__), "stats_test")

    # --- detector outputs on a crop (reusing the tested change_detection.py) ---
    rb = os.path.join(DATA_ROOT, DATE_REF) + os.sep
    rt = os.path.join(DATA_ROOT, DATE_TEST) + os.sep
    C11b, C22b, C12b = cd.Load_One_Date_Dual(rb, DATE_REF)
    C11a, C22a, C12a = cd.Load_One_Date_Dual(rt, DATE_TEST)
    r0, c0 = C11a.shape[0] // 2, C11a.shape[1] // 2
    sl = (slice(r0, r0 + CROP), slice(c0, c0 + CROP))
    C11b, C22b, C12b = cd.Boxcar_Pol_filter_Dual(C11b[sl], C22b[sl], C12b[sl], WIN)
    C11a, C22a, C12a = cd.Boxcar_Pol_filter_Dual(C11a[sl], C22a[sl], C12a[sl], WIN)
    det = cd.compute_change_detectors(C11a, C22a, C12a, C11b, C22b, C12b,
                                      progress=False)

    # --- SYNTHETIC reference masks (for a functional test only!) --------------
    # Real runs load .npy masks from the validation set; here we fabricate
    # disjoint class masks from Dif1 quantiles just to exercise the pipeline.
    ref = det["Dif1"]
    q = np.nanpercentile(ref, [20, 40, 60, 80])
    masks = {
        "open_flood":    ref >= q[3],
        "grass_flood":  (ref >= q[2]) & (ref < q[3]),
        "grass_noflood": (ref >= q[1]) & (ref < q[2]),
        "open_noflood":  ref < q[0],
    }

    print("Running accuracy assessment for ALL detectors ...")
    assess_all_detectors(det, masks, OUT, date=DATE_TEST, dim=500)

    table, best_dir = auc_summary(det, masks, dim=500)
    pd.set_option("display.float_format", lambda v: f"{v:.3f}")
    print("\n--- AUC summary (best direction per detector) ---")
    print(table)
    print("\n--- best threshold direction ---")
    print(best_dir)

    plot_roc_overview(det, masks, cover="Combined", dim=500,
                      save_path=os.path.join(OUT, "roc_overview_combined.png"),
                      title="ROC overview (Combined) - synthetic test masks")
    print("\nDone.")
