import numpy as np
import math


def smoothing_window(win_size: int, smooth_factor: float) -> np.ndarray:
    """smoothingWindow.m"""
    if win_size <= 1:
        return np.ones(max(1, win_size), dtype=float)
    factor = (2 * math.pi) / (win_size - 1)  # 8*atan(1) == 2*pi
    n = np.arange(win_size, dtype=float)
    return smooth_factor - (1.0 - smooth_factor) * np.cos(factor * n)

def psycho_model(len_bins: int, oct_frac: float) -> np.ndarray:
    """PsychoModel.m — win_sizes = round(0.5 * larg * (1:len))"""
    larg = (math.sqrt(2.0) ** oct_frac) - (math.sqrt(0.5) ** oct_frac)
    i = np.arange(1, len_bins + 1, dtype=float)
    return np.round(0.5 * larg * i).astype(int)  

def smooth_data(datIn: np.ndarray, datLen: int,
                win_sizes: np.ndarray, smoothFactor: float) -> np.ndarray:
    """smoothData.m"""
    datIn = np.asarray(datIn)
    out = np.zeros(datLen, dtype=datIn.dtype if np.iscomplexobj(datIn) else float)

    for i in range(datLen):
        w = int(win_sizes[i]) if i < len(win_sizes) else 0
        if w > 1:
            win = smoothing_window(2 * w + 1, smoothFactor)

            startIdx = i - w
            endIdx = i + w


            w_start = 0
            w_end = len(win)
            if startIdx < 0:
                w_start = (-startIdx)
                startIdx = 0
            if endIdx >= datLen:
                overshoot = endIdx - (datLen - 1)
                w_end = len(win) - overshoot
                endIdx = datLen - 1

            segment = datIn[startIdx:endIdx + 1]
            w_seg = win[w_start:w_end]


            out[i] = np.sum(segment * w_seg) / (2 * w + 1)
        else:
            out[i] = datIn[i]
    return out

# --- main smoothing (complexSmoothing.m) ---

def complex_smoothing_core(irIn: np.ndarray,
                           smoothingMethod: str = 'spectrum',
                           smoothFactor: float = 0.99,
                           oct_frac: float = 1/3):
    """
    Επιστρέφει (irSmoothed, specOut) όπως στο MATLAB.
    """
    x = np.asarray(irIn)
    if x.ndim != 1:
        x = x.reshape(-1)

    N = int(2 ** math.ceil(math.log2(len(x)))) if len(x) > 0 else 1
    specIn = np.fft.fft(x, N)
    phaseIn = np.angle(specIn)
    specOut = np.zeros(N, dtype=complex)
    Fn = N // 2 + 1  # bins

    method = (smoothingMethod or 'none').lower()
    if method == 'none':
        return np.real(np.fft.ifft(specIn))[:len(x)], specIn

    winSizes = psycho_model(Fn, oct_frac)

    if method == 'spectrum':
        mag = np.abs(specIn)
        out = smooth_data(mag, Fn, winSizes, smoothFactor)
        specOut[:Fn] = out[:Fn] * np.exp(1j * phaseIn[:Fn])

    elif method == 'power':
        pwr = (np.abs(specIn)) ** 2
        out = smooth_data(pwr, Fn, winSizes, smoothFactor)
        specOut[:Fn] = np.sqrt(out[:Fn]) * np.exp(1j * phaseIn[:Fn])

    elif method == 'db':
        mag_db = 20.0 * np.log10(np.abs(specIn) + 1e-12)
        out_db = smooth_data(mag_db, Fn, winSizes, smoothFactor)
        out_lin = 10.0 ** (out_db[:Fn] / 20.0)
        specOut[:Fn] = out_lin * np.exp(1j * phaseIn[:Fn])

    elif method == 'phase':
        mag = np.abs(specIn)
        ph_smooth = smooth_data(phaseIn, Fn, winSizes, smoothFactor)
        specOut[:Fn] = mag[:Fn] * np.exp(1j * ph_smooth[:Fn])

    elif method == 'complex':
        Rsm = smooth_data(np.real(specIn), Fn, winSizes, smoothFactor)
        Ism = smooth_data(np.imag(specIn), Fn, winSizes, smoothFactor)
        specOut[:Fn] = Rsm[:Fn] + 1j * Ism[:Fn]

    elif method == 'mixed':
        mag = np.abs(specIn)
        mag_s = smooth_data(mag, Fn, winSizes, smoothFactor)
        Rsm = smooth_data(np.real(specIn), Fn, winSizes, smoothFactor)
        Ism = smooth_data(np.imag(specIn), Fn, winSizes, smoothFactor)
        ph = np.angle(Rsm[:Fn] + 1j * Ism[:Fn])
        specOut[:Fn] = mag_s[:Fn] * np.exp(1j * ph)

    else:
        return np.real(np.fft.ifft(specIn))[:len(x)], specIn

    # conjugate symmetry (MATLAB: flip real, -flip imag)
    if N >= 3:
        specOut[Fn:] = np.flip(np.real(specOut[1:Fn-1])) + 1j * (-1) * np.flip(np.imag(specOut[1:Fn-1]))

    irSmoothed = np.real(np.fft.ifft(specOut))[:len(x)]
    return irSmoothed, specOut



def complex_smoothing(signal: np.ndarray, fs: float, method: str, factor: float, oct_frac: float):
    y, _ = complex_smoothing_core(signal, method, factor, oct_frac)
    return y
