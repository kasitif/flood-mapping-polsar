"""
Dual-pol polarimetric change detection for flood mapping (per-pixel version).

Input  : SNAP-exported ENVI covariance elements (.hdr / .img) per acquisition
         C11_<date>, C22_<date>, C12_real_<date>, C12_imag_<date>
Output : a dict of change detectors (Dif, Rat, Lin, MonU, RatDif, ...)

This build uses an explicit per-pixel loop with single-matrix LAPACK calls
(np.linalg.eigh / scipy.linalg.eig / inv). Batched np.linalg is intentionally
avoided because the installed BLAS crashes on stacked arrays.

Convention (kept identical to the original driver call):
    "_a" = image 2  (during flood / test acquisition)  -> C2
    "_b" = image 1  (reference    / before  acquisition) -> C1

Paper: Isundwa et al., "Analytical Optimization of Polarimetric Change
Detectors for Mapping Open and Grassland Flooding Using Sentinel-1." JSTARS 2026.
"""

import os
import json
import time
import numpy as np
from scipy import signal
import scipy.linalg as ln

try:
    from tqdm import tqdm
except ImportError:                       # tqdm is optional
    def tqdm(x, **k):
        return x


# =============================================================================
# I/O  (minimal ENVI reader -- no 'spectral' dependency)
# =============================================================================
# ENVI data type code -> numpy base dtype
_ENVI_DTYPE = {
    1: "u1", 2: "i2", 3: "i4", 4: "f4", 5: "f8",
    6: "c8", 9: "c16", 12: "u2", 13: "u4", 14: "i8", 15: "u8",
}


def _parse_envi_header(hdr_path):
    """Parse the fields we need from an ENVI .hdr text file."""
    fields = {}
    with open(hdr_path, "r") as f:
        for line in f:
            if "=" in line:
                k, v = line.split("=", 1)
                fields[k.strip().lower()] = v.strip()
    samples = int(fields["samples"])
    lines = int(fields["lines"])
    dcode = int(fields["data type"])
    border = int(fields.get("byte order", 0))
    base = _ENVI_DTYPE[dcode]
    dtype = np.dtype((">" if border == 1 else "<") + base)   # 1 = big-endian
    return samples, lines, dtype


def Open_ENVI_Image(filename):
    """Open an ENVI binary image. `filename` is the path WITHOUT extension."""
    samples, lines, dtype = _parse_envi_header(filename + ".hdr")
    with open(filename + ".img", "rb") as f:
        img = np.fromfile(f, dtype=dtype)
    # match the original pipeline: shape (samples, lines), Fortran order
    img = img.reshape((samples, lines), order="F").astype(dtype)
    return img


def Load_One_Date_Dual(path, date):
    """Load the covariance elements (C11, C22, C12 complex) for one date."""
    C11 = Open_ENVI_Image(path + "C11_" + date)
    C22 = Open_ENVI_Image(path + "C22_" + date)
    C12_real = Open_ENVI_Image(path + "C12_real_" + date)
    C12_imag = Open_ENVI_Image(path + "C12_imag_" + date)
    C12 = C12_real + 1j * C12_imag
    del C12_real, C12_imag
    return C11, C22, C12


# =============================================================================
# Speckle filtering
# =============================================================================
def Boxcar_Pol_filter_Dual(C11, C22, C12, win=(7, 7)):
    """Boxcar multilooking of the covariance elements."""
    w1, w2 = int(win[0]), int(win[1])
    kernel = np.ones((w1, w2), np.float32) / (w1 * w2)
    opt = dict(mode="same", boundary="fill", fillvalue=0)
    C11f = signal.convolve2d(C11, kernel, **opt)
    C22f = signal.convolve2d(C22, kernel, **opt)
    C12f = signal.convolve2d(C12, kernel, **opt)   # complex -> filtered complex
    return C11f, C22f, C12f


# =============================================================================
# Per-pixel detector bank
# =============================================================================
def compute_change_detectors(C11_a, C22_a, C12_a,
                             C11_b, C22_b, C12_b,
                             eps=1e-6, progress=True):
    """
    Compute all change detectors for one image pair, pixel by pixel.

    Parameters
    ----------
    C11_a, C22_a, C12_a : (H, W) arrays  covariance elements of image 2 (C2)
    C11_b, C22_b, C12_b : (H, W) arrays  covariance elements of image 1 (C1)

    Returns
    -------
    dict of float32 (H, W) arrays, one per detector.
    """
    dima, dimr = C11_a.shape

    Dif1 = np.zeros((dima, dimr), np.float32)
    Dif2 = np.zeros((dima, dimr), np.float32)
    Alpha1 = np.zeros((dima, dimr), np.float32)
    Alpha2 = np.zeros((dima, dimr), np.float32)
    Rat1 = np.zeros((dima, dimr), np.float32)
    Rat2 = np.zeros((dima, dimr), np.float32)
    RatDif = np.zeros((dima, dimr), np.float32)
    MonU1 = np.zeros((dima, dimr), np.float32)
    MonU2 = np.zeros((dima, dimr), np.float32)
    Lin1 = np.zeros((dima, dimr), np.float32)
    Lin2 = np.zeros((dima, dimr), np.float32)
    Wishart = np.zeros((dima, dimr), np.float32)
    Trace = np.zeros((dima, dimr), np.float32)

    Ta = np.zeros((2, 2), dtype=np.complex128)
    Tb = np.zeros((2, 2), dtype=np.complex128)
    I2 = np.eye(2)

    rows = tqdm(range(dima)) if progress else range(dima)
    for i in rows:
        for ii in range(dimr):
            # Build the two 2x2 Hermitian covariance matrices
            Ta[0, 0] = C11_a[i, ii] + eps
            Ta[1, 1] = C22_a[i, ii] + eps
            Ta[0, 1] = C12_a[i, ii]
            Ta[1, 0] = np.conj(Ta[0, 1])
            Tb[0, 0] = C11_b[i, ii] + eps
            Tb[1, 1] = C22_b[i, ii] + eps
            Tb[0, 1] = C12_b[i, ii]
            Tb[1, 0] = np.conj(Tb[0, 1])

            # ---- OPDiff : B = C2 - C1 (Hermitian) ----
            d2, v2 = np.linalg.eigh(Ta - Tb)        # ascending reals
            Dif1[i, ii] = d2[1]                     # max power difference
            Dif2[i, ii] = d2[0]                     # min power difference
            Alpha1[i, ii] = np.arccos(np.abs(v2[0, 1]))
            Alpha2[i, ii] = np.arccos(np.abs(v2[0, 0]))

            # ---- OPRatio : A = C1^{-1} C2 ----
            A = np.linalg.inv(Tb) @ Ta
            d1 = ln.eig(A)[0]
            d1a = np.abs(d1)
            Rat1[i, ii] = np.max(d1a)               # largest |eigenvalue|
            Rat2[i, ii] = 1.0 / np.min(d1a)         # reciprocal of smallest
            Trace[i, ii] = np.real(A[0, 0] + A[1, 1])

            # ---- Model 4 (mixed) : headline detector  Rat2^2 * Dif1 ----
            RatDif[i, ii] = Rat2[i, ii] ** 2 * Dif1[i, ii]

            # ---- Model 1 (monomial unitary) : whiten C1 with C2 ----
            wa, va = np.linalg.eigh(Ta)             # C2 Hermitian PD
            wa = np.clip(wa, 1e-10, None)
            Ta_is = (va * (1.0 / np.sqrt(wa))) @ va.conj().T   # C2^{-1/2}
            Tm_u = Ta_is @ Tb @ Ta_is
            lam_u = np.linalg.eigvalsh(Tm_u)
            m1 = lam_u * (lam_u - 1.0)
            MonU1[i, ii] = np.max(m1)
            MonU2[i, ii] = np.min(m1)

            # ---- Model 3 (linear) : faithful to original (ELEMENT-WISE sqrt) ----
            Ta_sqrt = np.sqrt(Ta)                   # element-wise, as original code
            Tm_l = Ta_sqrt @ Tb @ Ta_sqrt
            dl = ln.eig(Tm_l)[0]
            Lin1[i, ii] = np.real(dl[1])
            Lin2[i, ii] = np.real(dl[0])

            # ---- Wishart ----
            detA = Ta[0, 0] * Ta[1, 1] - Ta[0, 1] * Ta[1, 0]
            detB = Tb[0, 0] * Tb[1, 1] - Tb[0, 1] * Tb[1, 0]
            Tavg = (Ta + Tb) / 2.0
            detAvg = Tavg[0, 0] * Tavg[1, 1] - Tavg[0, 1] * Tavg[1, 0]
            Wishart[i, ii] = np.abs(detA) * np.abs(detB) / (np.abs(detAvg) ** 2 + eps)

    # ---- Simple backscatter ratios (vectorised, no LAPACK) ----
    RatVV = ((C11_a + eps) / (C11_b + eps)).astype(np.float32)
    RatVH = ((C22_a + eps) / (C22_b + eps)).astype(np.float32)

    return {
        "Dif1": Dif1, "Dif2": Dif2,
        "Alpha1": Alpha1, "Alpha2": Alpha2,
        "Rat1": Rat1, "Rat2": Rat2,
        "RatDif": RatDif,
        "MonU1": MonU1, "MonU2": MonU2,
        "Lin1": Lin1, "Lin2": Lin2,
        "RatVV": RatVV, "RatVH": RatVH,
        "Wishart": Wishart, "Trace": Trace,
    }


# =============================================================================
# Saving detector outputs  (change detection -> results dir)
# =============================================================================
def save_detectors(det, results_dir, date, transpose=False):
    """
    Save every detector array to `results_dir` as `<name>_<date>.npy`.

    The accuracy-assessment notebook reloads these with `load_detectors`.

    Parameters
    ----------
    det : dict[str, (H, W) array]   output of compute_change_detectors
    results_dir : str or Path       folder to write into (created if missing)
    date : str                      tag used in the file names, e.g. '20170715'
    transpose : bool                save each array transposed (.T) so it matches
                                    the orientation of your reference masks

    Returns
    -------
    results_dir (str)
    """
    results_dir = str(results_dir)
    os.makedirs(results_dir, exist_ok=True)
    for name, arr in det.items():
        a = np.asarray(arr)
        if transpose:
            a = a.T
        np.save(os.path.join(results_dir, f"{name}_{date}.npy"), a.astype(np.float32))
    # small manifest for sanity / reloading
    shape = list(np.asarray(next(iter(det.values()))).T.shape) if transpose \
        else list(np.asarray(next(iter(det.values()))).shape)
    manifest = {"date": date, "detectors": list(det.keys()),
                "shape": shape, "transpose": transpose}
    with open(os.path.join(results_dir, f"manifest_{date}.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved {len(det)} detectors for {date} -> {os.path.abspath(results_dir)}")
    return results_dir


# =============================================================================
# Runnable test  --  100 x 100 crop of the real ENVI data
# =============================================================================
if __name__ == "__main__":
    # Local water-hyacinth sample data: identical ENVI format, runs out of the box.
    WIN = (7, 7)
    CROP = 100               # test window size (pixels)

    # ---- Guyana workflow (uncomment and point at your drive) ----
    DATA_ROOT = r"D:\FLOOD\Guyana\4_Stack_Sorted"
    DATE_REF = "20170329"    # C1 / before
    DATE_TEST = "20170715"   # C2 / during

    ref_dir = os.path.join(DATA_ROOT, DATE_REF) + os.sep
    test_dir = os.path.join(DATA_ROOT, DATE_TEST) + os.sep

    print(f"Loading reference {DATE_REF} ...", flush=True)
    C11b, C22b, C12b = Load_One_Date_Dual(ref_dir, DATE_REF)
    print(f"Loading test      {DATE_TEST} ...", flush=True)
    C11a, C22a, C12a = Load_One_Date_Dual(test_dir, DATE_TEST)
    print(f"Full scene size: {C11a.shape}", flush=True)

    # Centre crop so the window contains real signal
    r0 = C11a.shape[0] // 2
    c0 = C11a.shape[1] // 2
    sl = (slice(r0, r0 + CROP), slice(c0, c0 + CROP))

    print(f"Boxcar filtering {CROP}x{CROP} crop ...", flush=True)
    C11b, C22b, C12b = Boxcar_Pol_filter_Dual(C11b[sl], C22b[sl], C12b[sl], WIN)
    C11a, C22a, C12a = Boxcar_Pol_filter_Dual(C11a[sl], C22a[sl], C12a[sl], WIN)

    print(f"Running per-pixel change detection on {CROP}x{CROP} ...", flush=True)
    t0 = time.time()
    det = compute_change_detectors(C11a, C22a, C12a, C11b, C22b, C12b)
    dt = time.time() - t0
    print(f"Done in {dt:.2f} s  ({CROP*CROP:,} pixels, "
          f"{1e6*dt/(CROP*CROP):.1f} us/pixel)", flush=True)

    print("\n--- detector summary (min / max / mean) ---", flush=True)
    for k, v in det.items():
        print(f"  {k:8s} {np.nanmin(v): .4g}  ..  {np.nanmax(v): .4g}  "
              f"(mean {np.nanmean(v): .4g})", flush=True)
