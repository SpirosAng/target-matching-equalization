# target_eq.py
import numpy as np
from scipy.signal import fftconvolve, firwin2
from io import StringIO
from typing import Tuple, Optional
from csmooth import psycho_model, smooth_data 

def _rfft_db(x: np.ndarray, fs: float, nfft: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    if nfft is None:
        nfft = int(2 ** np.ceil(np.log2(len(x)))) if len(x) > 0 else 512
    X = np.fft.rfft(x, nfft)
    f = np.fft.rfftfreq(nfft, 1.0 / fs)
    db = 20.0 * np.log10(np.maximum(np.abs(X), 1e-12))
    return f, db

def _smooth_db_curve(db: np.ndarray, oct_frac: float, factor: float = 0.99) -> np.ndarray:
    Fn = len(db)
    win_sizes = psycho_model(Fn, oct_frac)
    return smooth_data(db, Fn, win_sizes, factor)

def source_envelope(x, fs, oct_frac, factor: float = 0.99,
                    nfft: Optional[int] = None, smoothFactor: float = None):
    if smoothFactor is not None:  # alias
        factor = smoothFactor
    f, db = _rfft_db(x, fs, nfft=nfft)
    db_s = _smooth_db_curve(db, oct_frac, factor)
    return f, db_s

def parse_target_txt(file_bytes: bytes) -> Tuple[np.ndarray, np.ndarray]:
    txt = file_bytes.decode("utf-8", errors="ignore")
    try:
        arr = np.loadtxt(StringIO(txt), delimiter=",")
        if arr.ndim == 1:
            arr = arr[None, :]
    except Exception:
        arr = np.loadtxt(StringIO(txt))
        if arr.ndim == 1:
            arr = arr[None, :]
    if arr.shape[1] < 2:
        raise ValueError("Target TXT/CSV must have 2 columns: Hz, dB.")
    f = np.asarray(arr[:, 0], dtype=float)
    db = np.asarray(arr[:, 1], dtype=float)
    m = np.isfinite(f) & np.isfinite(db) & (f >= 0)
    f, db = f[m], db[m]
    order = np.argsort(f)
    return f[order], db[order]

def interpolate_target(freq_src: np.ndarray, db_src: np.ndarray,
                       fs: float, npoints: int = 4096) -> Tuple[np.ndarray, np.ndarray]:
    fgrid = np.linspace(0.0, fs / 2.0, npoints)
    if len(freq_src) < 2:
        db_grid = np.full_like(fgrid, db_src[0] if len(db_src) else 0.0, dtype=float)
        return fgrid, db_grid
    db_grid = np.interp(fgrid, freq_src, db_src, left=db_src[0], right=db_src[-1])
    return fgrid, db_grid

def build_target_from_wav(target_audio: np.ndarray, fs_target: float, fs_out: float,
                          oct_frac: float, factor: float = 0.99, npoints: int = 4096):
    """
    Generates a target curve (fgrid, dB) from a reference WAV file.
    - Computes the envelope at fs_target
    - Interpolates it to fs_out over npoints in the range [0 .. fs_out/2]
    """
    # χρησιμοποιούμε την υπάρχουσα source_envelope για να πάρουμε (f, dB)
    f_t, db_t = source_envelope(target_audio, fs_target, oct_frac, factor=factor)
    fgrid = np.linspace(0.0, fs_out / 2.0, npoints)
    # extrapolate άκρα ως flat
    db_grid = np.interp(fgrid, f_t, db_t, left=db_t[0], right=db_t[-1])
    return fgrid, db_grid

def design_match_fir(
    fgrid: np.ndarray,
    source_db_grid: np.ndarray,
    target_db_grid: np.ndarray,
    fs: float,
    fir_len: int = 2049,
    f_lo: float = 20.0,
    f_hi: Optional[float] = None,
    max_boost_db: float = 6.0,
    max_cut_db: float = 12.0,
) -> np.ndarray:
    """
    Simple and stable FIR design using firwin2 over the given fgrid.
    No weighting factors and no additional processing tricks.
    """
    if f_hi is None:
        f_hi = fs / 2.0
    f_hi = min(f_hi, fs / 2.0)


    delta_db = target_db_grid - source_db_grid
    mask = (fgrid >= f_lo) & (fgrid <= f_hi)
    delta_db = np.where(mask, delta_db, 0.0)
    delta_db = np.clip(delta_db, -abs(max_cut_db), abs(max_boost_db))

    gain = 10.0 ** (delta_db / 20.0)


    if fgrid[0] > 0.0:
        fgrid = np.insert(fgrid, 0, 0.0)
        gain  = np.insert(gain,  0, gain[0])
    if fgrid[-1] < fs / 2.0:
        fgrid = np.append(fgrid, fs / 2.0)
        gain  = np.append(gain,  gain[-1])


    if fir_len % 2 == 0:
        fir_len += 1

    fir = firwin2(numtaps=fir_len, freq=fgrid, gain=gain, fs=fs)
    return fir.astype(np.float64)

def apply_fir_convolution(x: np.ndarray, fir: np.ndarray, normalize: bool = True) -> np.ndarray:
    y = fftconvolve(x, fir, mode="same")
    if normalize:
        peak = np.max(np.abs(y)) if y.size else 1.0
        if peak > 1.0:
            y = y / peak
    return y
