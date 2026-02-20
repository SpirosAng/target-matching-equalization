# Target-Matching FIR Equalization (Python)

This repository contains the implementation of my Diploma Thesis on digital sound equalization.

The project presents a complete Target-Matching FIR Equalization system developed in Python, integrating psychoacoustic complex smoothing, fractional-octave envelope processing, frequency-domain analysis, and objective performance evaluation.

---

## Overview

The system performs digital equalization by designing an inverse FIR filter that forces a measured impulse response to match a predefined target curve.

The workflow includes:

1. Impulse response input (h(t))
2. Frequency-domain transformation (FFT)
3. Psychoacoustic complex smoothing
4. Fractional-octave envelope smoothing
5. Target curve interpolation
6. FIR filter design (firwin2-based)
7. Fast convolution implementation
8. Objective performance evaluation (KPIs)
9. Real-time and simulation listening 

---

## Key Features

- Psychoacoustic complex smoothing methods:
  - spectrum
  - power
  - dB
  - phase
  - complex
  - mixed

- Fractional-octave envelope smoothing
- FIR target-matching filter design
- Adjustable FIR parameters:
  - Filter length (taps)
  - Max boost / max cut
  - Correction band

- Objective performance indicators:
  - RMS error
  - Maximum error
  - FIR gain range
  - Latency

- Real-time and simulation listening modes (Streamlit interface)

---

## Mathematical Concept

Given a measured impulse response \( h(n) \), the goal is to design an inverse FIR filter \( h_{inv}(n) \) such that:

h(n) * h_{inv}(n) ≈ δ(n)

Instead of directly inverting the raw response, psychoacoustic complex smoothing is applied in the frequency domain to ensure a stable and perceptually consistent correction.

---

## Project Structure
