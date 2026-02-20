# diplomatic_app.py
import io
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import streamlit as st
from io import StringIO
from scipy.signal import freqz, fftconvolve
import pandas as pd

from csmooth import complex_smoothing
from target_eq import (
    parse_target_txt, interpolate_target, source_envelope,
    design_match_fir, apply_fir_convolution, build_target_from_wav
)

# ============================================================
# Page setup
# ============================================================
st.set_page_config(page_title="Target-Matching EQ", layout="wide")
st.title("Digital Sound Equalization App")

# ============================================================
# Layout: 3 columns (1 | 2 | 3) as in your drawing
# ============================================================
col1, col2, col3 = st.columns(3, vertical_alignment="top")

# ============================================================
def octave_smooth_db(
    freq_hz: np.ndarray,
    db_in: np.ndarray,
    frac: float = 3.0,              # 1/frac octave: 3 => 1/3
    fmin: float = 20.0,
    fmax: float | None = None,
    points_per_oct: int = 48
) -> tuple[np.ndarray, np.ndarray]:
    f = np.asarray(freq_hz, dtype=float)
    db = np.asarray(db_in, dtype=float)
    m = np.isfinite(f) & np.isfinite(db) & (f > 0)
    f = f[m]; db = db[m]
    if f.size < 4:
        return f, db

    if fmax is None:
        fmax = float(np.max(f))

    lo = np.log2(max(fmin, np.min(f)))
    hi = np.log2(min(fmax, np.max(f)))
    n_points = max(64, int(np.ceil((hi - lo) * points_per_oct)) + 1)
    grid_log = np.linspace(lo, hi, n_points)
    grid_f = 2.0 ** grid_log

    p_lin = 10.0 ** (db / 10.0)
    p_lin_g = np.interp(grid_f, f, p_lin)

    if frac is None:
        db_unsmoothed = 10.0 * np.log10(np.maximum(p_lin_g, 1e-20))
        return grid_f, db_unsmoothed

    fwhm_oct = 1.0 / float(frac)
    sigma_oct = fwhm_oct / 2.355
    half_w_pts = int(np.ceil(3.0 * sigma_oct * points_per_oct))
    offsets = np.arange(-half_w_pts, half_w_pts + 1)
    offsets_oct = offsets / points_per_oct
    kernel = np.exp(-0.5 * (offsets_oct / sigma_oct) ** 2)
    kernel /= kernel.sum()

    p_smooth = np.convolve(p_lin_g, kernel, mode="same")
    db_smooth = 10.0 * np.log10(np.maximum(p_smooth, 1e-20))
    return grid_f, db_smooth


# ----------------- helpers -----------------
def read_wav_any(uploaded_or_bytes):
    """
    Robust WAV reader for Streamlit UploadedFile or raw bytes.
    Ensures we can read multiple times without pointer issues.
    """
    if hasattr(uploaded_or_bytes, "getvalue"):
        b = uploaded_or_bytes.getvalue()
        bio = io.BytesIO(b)
        data, fs = sf.read(bio)
    elif isinstance(uploaded_or_bytes, (bytes, bytearray)):
        bio = io.BytesIO(uploaded_or_bytes)
        data, fs = sf.read(bio)
    else:
        # fallback: assume file-like
        data, fs = sf.read(uploaded_or_bytes)

    if data.ndim > 1:
        data = data[:, 0]
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64)
    return data, fs


def match_loudness(signal_to_scale, ref_signal):
    """
    Adjusts the RMS level of signal_to_scale to match the RMS of ref_signal.
    This ensures that the EQ output does not sound quieter
    and does not appear artificially "smaller" in the frequency response plot.
    """
    # (Root Mean Square)
    rms_ref = np.sqrt(np.mean(ref_signal ** 2))
    rms_sig = np.sqrt(np.mean(signal_to_scale ** 2))

    if rms_sig < 1e-12:
        return signal_to_scale

    # (Gain)
    gain = rms_ref / rms_sig
    scaled_signal = signal_to_scale * gain

    # Safety Limiter: If RMS scaling results in peaks > 1.0 (clipping),
    # scale the entire signal proportionally so that the maximum peak is limited to 0.99.
    max_peak = np.max(np.abs(scaled_signal))
    if max_peak > 0.99:
        scaled_signal = scaled_signal * (0.99 / max_peak)

    return scaled_signal

def env_curve(signal, fs, dense_oct=1/48):
    f_raw, db_raw = source_envelope(signal, fs, dense_oct, factor=0.99)
    m = np.isfinite(f_raw) & np.isfinite(db_raw) & (f_raw > 0)
    return f_raw[m], db_raw[m]


def semilog_plot(ax, f, db, label, **kw):
    m = np.isfinite(f) & np.isfinite(db)
    if m.any():
        ax.semilogx(f[m], db[m], label=label, **kw)


def kpis(values: np.ndarray):
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan, np.nan
    rms = np.sqrt(np.mean(values**2))
    mx = np.max(np.abs(values))
    return rms, mx


# ============================================================
# Dictionaries
# ============================================================
rew_fracs = {
    "None": None,
    "1/1":1.0, "1/2":2.0, "1/3":3.0, "1/6":6.0,
    "1/12":12.0, "1/24":24.0, "1/48":48.0
}
_oct_csm = {"1/3":1/3, "1/6":1/6, "1/10":0.1, "1/12":1/12}

# ============================================================
# COLUMN 1: Uploads + csmooth + smoothing + FIR sliders
# ============================================================
with col1:
    st.subheader("Inputs & FIR")

    ir_file = st.file_uploader("① Upload IR h(t)", type=["wav"])
    if not ir_file:
        st.info("Upload a  **WAV** file with the impulse response h(t).")
        st.stop()

    h_raw, fs = read_wav_any(ir_file)
    st.caption(f"IR sample rate: {fs} Hz • Length: {len(h_raw)/fs:.3f} s")
    st.audio(ir_file.getvalue(), format="audio/wav")

    st.markdown("---")
    st.write("**S(t) — Recording/Sweep **")
    prog_file = st.file_uploader("② Upload program material S(t) ", type=["wav"], key="prog")

    st.markdown("---")
    tgt_file = st.file_uploader("③ Upload TARGET (TXT/CSV: Hz,dB) or WAV ", type=["txt","csv","wav"])
    if not tgt_file:
        st.info("Upload target **TXT/CSV** (Hz,dB) or  **WAV** as reference for h(t).")
        st.stop()

    target_is_wav = (tgt_file.type == "audio/wav")

    #smoothing
    st.markdown("---")
    st.write("**Select envelope Smoothing **")
    src_sel = st.selectbox("Source smoothing (env h(t))", list(rew_fracs.keys()), index=2)
    tgt_sel = st.selectbox("Target smoothing", list(rew_fracs.keys()), index=2)

    # Pre-EQ smoothing (csmooth)
    st.markdown("---")
    st.write("**Pre-EQ smoothing on IR (csmooth)**")
    sm_method = st.selectbox(
        "csmooth method",
        ["None","spectrum","power","db","phase","complex","mixed"],
        0
    )
    sm_oct = st.selectbox("csmooth octave fraction", ["None","1/3","1/6","1/10","1/12"], 1)

    

    # FIR sliders
    st.markdown("---")
    st.subheader("Automatic Target-Matching FIR (EQ)")
    fir_len = st.slider("FIR length (taps, odd)", 257, 8191, 2049, step=2)
    max_boost = st.slider("Max boost (dB)", 0.0, 12.0, 6.0, 0.5)
    max_cut = st.slider("Max cut (dB)", 0.0, 24.0, 12.0, 0.5)
    f_lo, f_hi = st.slider("Correction band (Hz)", 5, 20000, (20, 20000))


# ============================================================
# Compute program S(t) and dirty S*h
# ============================================================
S_raw = None
S_dirty = None
if prog_file is not None:
    S_raw, fs_prog = read_wav_any(prog_file)

    # resample program to IR fs if needed
    if fs_prog != fs:
        dur = len(S_raw) / fs_prog
        n_new = int(round(dur * fs))
        t_old = np.linspace(0, dur, len(S_raw), endpoint=False)
        t_new = np.linspace(0, dur, n_new, endpoint=False)
        S_raw = np.interp(t_new, t_old, S_raw)

    S_dirty = fftconvolve(S_raw, h_raw, mode="full")
    max_abs = np.max(np.abs(S_dirty))
    if max_abs > 0:
        S_dirty = 0.99 * S_dirty / max_abs


# ============================================================
# Apply optional csmooth to IR
# ============================================================
if sm_oct == "None" or sm_method == "None":
    x_in = h_raw.copy()
else:
    x_in = complex_smoothing(h_raw, fs, sm_method, 0.99, _oct_csm[sm_oct])


# ============================================================
# Build SOURCE envelope (smoothed)
# ============================================================
f_src_raw, src_db_raw = env_curve(x_in, fs, dense_oct=1/48)
f_src, src_db = octave_smooth_db(
    f_src_raw, src_db_raw,
    frac=rew_fracs[src_sel], fmin=20.0, fmax=fs/2, points_per_oct=48
)

# ============================================================
# Build TARGET curve (smoothed)
# ============================================================
if not target_is_wav:
    tgt_bytes = tgt_file.getvalue()
    f_txt, db_txt = parse_target_txt(tgt_bytes)
    f_tgt_raw, tgt_db_raw = np.asarray(f_txt, float), np.asarray(db_txt, float)
else:
    x_tgt, fs_tgt = read_wav_any(tgt_file)
    f_tgt_raw, tgt_db_raw = env_curve(x_tgt, fs_tgt, dense_oct=1/48)

f_tgt, tgt_db = octave_smooth_db(
    f_tgt_raw, tgt_db_raw,
    frac=rew_fracs[tgt_sel], fmin=20.0, fmax=fs/2, points_per_oct=48
)

# target on source grid
tgt_on_src = np.interp(f_src, f_tgt, tgt_db, left=tgt_db[0], right=tgt_db[-1])

# ============================================================
# FIR design
# ============================================================
if fir_len % 2 == 0:
    fir_len += 1

fir = design_match_fir(
    fgrid=f_src,
    source_db_grid=src_db,
    target_db_grid=tgt_on_src,
    fs=fs,
    fir_len=fir_len,
    f_lo=float(f_lo), f_hi=float(f_hi),
    max_boost_db=float(max_boost), max_cut_db=float(max_cut),
)

# ============================================================
# Apply FIR to IR & Normalize for Graph/Audio
# ============================================================

y_raw = apply_fir_convolution(x_in, fir, normalize=False)

# 2. RMS level matching with the original impulse response (x_in)
# This corrects both the plotted response (orange curve) and the perceived loudness simultaneously.
y = match_loudness(y_raw, x_in)

# Equalized envelope
f_eq_raw, db_eq_raw = env_curve(y, fs, dense_oct=1/48)
f_eq, db_eq = octave_smooth_db(
    f_eq_raw, db_eq_raw,
    frac=rew_fracs[src_sel], fmin=20.0, fmax=fs/2, points_per_oct=48
)

required_db = tgt_on_src - src_db
# -----------------------------------

# FIR response
fir_vec = np.asarray(fir, dtype=np.float64).ravel()
try:
    w, H = freqz(fir_vec, worN=8192, fs=fs)
    w_hz = w
except TypeError:
    w, H = freqz(fir_vec, worN=8192)
    w_hz = (w * fs) / (2.0*np.pi)

H_db = 20*np.log10(np.maximum(1e-12, np.abs(H)))
H_on = np.interp(f_src, w_hz, H_db, left=H_db[0], right=H_db[-1])

# Errors / KPIs
eq_on_src = np.interp(f_src, f_eq, db_eq, left=db_eq[0], right=db_eq[-1])
err_db = eq_on_src - tgt_on_src
rms_all, max_all = kpis(err_db)


bands = [
    ("20–80 Hz", (20, 80)),
    ("80–200 Hz", (80, 200)),
    ("200–2000 Hz", (200, 2000)),
    ("2k–20k Hz", (2000, 20000)),
]
rows = []
for label, (lo, hi) in bands:
    mask = (f_src >= lo) & (f_src <= hi)
    rms, mx = kpis(err_db[mask])
    rows.append([label, lo, hi, rms, mx])
df_bands = pd.DataFrame(rows, columns=["Band", "Lo (Hz)", "Hi (Hz)", "RMS", "Max |err|"])


df_curves = pd.DataFrame({
    "frequency_Hz": f_src,
    "original_env_dB": src_db,
    "target_dB": tgt_on_src,
    "equalized_env_dB": eq_on_src,
    "required_corr_dB": required_db,
    "fir_response_dB": H_on,
    "err_eq_minus_target_dB": err_db
})

# Filter info
latency_ms = (fir_len - 1) / (2 * fs) * 1000.0
boost_used = np.max(H_on) if np.isfinite(H_on).any() else np.nan
cut_used   = np.min(H_on) if np.isfinite(H_on).any() else np.nan


# ============================================================
# COLUMN 2: Plot controls -> plot -> first live measurements table
# ============================================================
with col2:
    st.subheader("Plot controls")

    view_mode = st.radio(
        "View",
        ["Frequency (dB)", "Time (waveform)", "Phase (degrees)"],
        index=0,
        horizontal=True
    )

    if view_mode == "Frequency (dB)":
        choices = st.multiselect(
            "Select curves to display",
            ["Original (env)", "Equalized (env)", "Target (smoothed)",
             "Required correction (dB)", "FIR response |H(f)| (dB)"],
            default=["Equalized (env)", "Target (smoothed)"]
        )

        fig, ax = plt.subplots(figsize=(7.2, 3.2))
        if "Original (env)" in choices:
            semilog_plot(ax, f_src, src_db, "Original (env)", alpha=0.6)
        if "Equalized (env)" in choices:
            semilog_plot(ax, f_eq, db_eq, "Equalized (env)")
        if "Target (smoothed)" in choices:
            semilog_plot(ax, f_src, tgt_on_src, "Target (smoothed)")
        if "Required correction (dB)" in choices:
            semilog_plot(ax, f_src, required_db, "Required correction (dB)", linestyle="--")
        if "FIR response |H(f)| (dB)" in choices:
            semilog_plot(ax, f_src, H_on, "FIR response |H(f)| (dB)", linestyle="--")

        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("dB")
        ax.grid(True, which="both", ls="--")
        ax.legend()
        ax.set_xlim(left=np.min(f_src), right=np.max(f_src))
        st.pyplot(fig, use_container_width=True)

    elif view_mode == "Time (waveform)":
        curves_time = ["h(t) original", "h_eq(t)"]
        sel_time = st.multiselect("Select waveforms", curves_time, default=curves_time)

        # Compute the maximum duration among the audio files currently loaded in memory.
        max_dur_h = len(h_raw) / fs
        max_dur_y = len(y) / fs
        limit_cap = min(10.0, max(max_dur_h, max_dur_y))

        # Zoom Slider
        view_dur = st.slider(
            "🔍 Zoom Level (Time Axis)", 
            min_value=0.02, 
            max_value=float(limit_cap), 
            value=0.10,  # Default
            step=0.01,
            format="%.2f s"
        )

        fig, ax = plt.subplots(figsize=(7.2, 3.0))

        n_show = int(view_dur * fs)

        if "h(t) original" in sel_time:
            limit = min(n_show, len(h_raw))
            t_axis = np.arange(limit) / fs
            ax.plot(t_axis, h_raw[:limit], label="h(t)", alpha=0.8, linewidth=1)
            
        if "h_eq(t)" in sel_time:
            limit = min(n_show, len(y))
            t_axis = np.arange(limit) / fs
            ax.plot(t_axis, y[:limit], label="h_eq(t)", alpha=0.8, linewidth=1)

        ax.set_xlim(0, view_dur)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right")
        
        st.pyplot(fig, use_container_width=True)

    elif view_mode == "Phase (degrees)":
        ph_choices = st.multiselect(
            "Select phase curves",
            ["Original", "Equalized"] + (["Target"] if target_is_wav else []),
            default=["Equalized"]
        )

        def phase_curve(x, fs_):
            X = np.fft.rfft(x)
            f = np.fft.rfftfreq(len(x), 1.0/fs_)
            ph_deg = np.unwrap(np.angle(X)) * 180.0 / np.pi
            return f, ph_deg

        fig, ax = plt.subplots(figsize=(7.2, 3.2))

        if "Original" in ph_choices:
            f_o, ph_o = phase_curve(x_in, fs)
            ax.semilogx(f_o, ph_o, label="Original (phase)", alpha=0.8)

        if "Equalized" in ph_choices:
            f_e, ph_e = phase_curve(y, fs)
            ax.semilogx(f_e, ph_e, label="Equalized (phase)", alpha=0.9)

        if target_is_wav and "Target" in ph_choices:
            xt, fst = read_wav_any(tgt_file)
            if fst != fs:
                t_src = np.arange(len(xt)) / fst
                t_dst = np.arange(int(round(len(xt) * fs / fst))) / fs
                xt = np.interp(t_dst, t_src, xt)
            f_t, ph_t = phase_curve(xt, fs)
            ax.semilogx(f_t, ph_t, label="Target (phase)", alpha=0.8)

        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Phase (degrees)")
        ax.grid(True, which="both", ls="--")
        ax.legend()
        ax.set_xlim(left=np.min(f_src), right=np.max(f_src))
        st.pyplot(fig, use_container_width=True)

    st.subheader("📊 Key Perfomance Indicators (KPIs)")

    c1, c2 = st.columns(2)
    c1.metric("RMS error (dB)", f"{rms_all:.2f}")
    c2.metric("Max |error| (dB)", f"{max_all:.2f}")

    m1, m2, m3 = st.columns(3)
    m1.metric("Taps", f"{fir_len}")
    m2.metric("Latency (ms)", f"{latency_ms:.2f}")
    m3.metric("FIR gain range (dB)", f"{cut_used:.1f} … {boost_used:.1f}")

    st.dataframe(df_bands, use_container_width=True)


# ============================================================
# COLUMN 3:
# ============================================================
with col3:
    st.subheader("Detailed Measurements per frequency")
    st.dataframe(df_curves, use_container_width=True)

    st.markdown("---")
    st.subheader("Export EQ IR & measurements")
    # ... exports ...

    st.markdown("---")
    st.subheader("🔊 Results Comparison & Export")

    if S_raw is not None and S_dirty is not None:

        # Buffer
        buf_s = io.BytesIO()
        sf.write(buf_s, S_raw, fs, format="WAV")


        mode = st.radio(
            "Select hearing scenario:",
            ["Simulation", "Real-time / Export"],
            index=0
        )

        st.markdown("---")

        if mode == "Simulation":
            st.info("💡 With this option you can hear how the recording will sound after simulated equalization")

            # 1. Original S(t)
            st.write("**1. S(t) — Reference**")
            st.audio(buf_s.getvalue(), format="audio/wav")

            # 2. Distorted S(t)*h(t)
            st.write("**2. S(t) * h(t) — Convolution of recording with impulse response**")

            # Match loudness of distorted with original
            S_dirty_norm = match_loudness(S_dirty, S_raw)

            buf_dirty = io.BytesIO()
            sf.write(buf_dirty, S_dirty_norm, fs, format="WAV")
            st.audio(buf_dirty.getvalue(), format="audio/wav")

            # 3. Restored (S(t)*h(t)) * FIR
            st.write("**3. (S(t) * h(t)) * FIR — Simulation**")

            S_restored = fftconvolve(S_dirty, fir_vec, mode="full")

            # --- SOS: Match RMS Loudness ---
            S_restored = match_loudness(S_restored, S_raw)

            buf_res = io.BytesIO()
            sf.write(buf_res, S_restored, fs, format="WAV")
            st.audio(buf_res.getvalue(), format="audio/wav")

        else:  # Real-time / Export
            st.info("💡 With this option you can hear and download the pre-processed file.")

            # 1. Original S(t)
            st.write("**1. S(t) — Reference**")
            st.audio(buf_s.getvalue(), format="audio/wav")

            # 2. Pre-emphasized S(t) * FIR
            st.write("**2. S(t) * h⁻¹(t) — Pre-emphasized S(t) * FIR**")

            S_pre_eq = fftconvolve(S_raw, fir_vec, mode="full")

            # --- SOS: Match RMS Loudness ---
            S_pre_eq = match_loudness(S_pre_eq, S_raw)

            buf_pre = io.BytesIO()
            sf.write(buf_pre, S_pre_eq, fs, format="WAV")
            st.audio(buf_pre.getvalue(), format="audio/wav")

            st.download_button(
                "💾 Download the file for Real-time playback (filtered.wav)",
                data=buf_pre.getvalue(),
                file_name="program_filtered_for_playback.wav",
                mime="audio/wav"
            )

    else:
        st.caption("No program material S(t) was uploaded for hearing.")

