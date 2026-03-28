"""
Frequency Analysis Module (FFT)
Detects spectral artifacts, grid patterns (checkerboard), and abnormal frequency distributions.
GANs introduce periodic upsampling artifacts visible in the Fourier domain.
Diffusion models inherit spectral characteristics from training data.
"""

import numpy as np
import cv2
import base64
from io import BytesIO
from PIL import Image
from . import utils


def analyze(image_np):
    """
    Perform FFT-based frequency analysis on an image.
    Returns score (0-100, higher = more likely AI), details dict, and visualization.
    """
    gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY).astype(np.float32)
    h, w = gray.shape

    # Apply windowing to reduce spectral leakage
    window_h = np.hanning(h)
    window_w = np.hanning(w)
    window = np.outer(window_h, window_w)
    gray_windowed = gray * window

    # 2D FFT
    f_transform = np.fft.fft2(gray_windowed)
    f_shift = np.fft.fftshift(f_transform)
    magnitude = np.abs(f_shift)
    power_spectrum = np.log1p(magnitude)

    # 1. Azimuthal averaging - radial profile of power spectrum
    cy, cx = h // 2, w // 2
    max_radius = min(cy, cx)
    radial_profile = _azimuthal_average(magnitude, cx, cy, max_radius)

    # 2. Check for 1/f decay (natural images follow ~1/f^β power law)
    radii = np.arange(1, len(radial_profile))
    log_radii = np.log(radii)
    log_profile = np.log(radial_profile[1:] + 1e-10)

    # Fit linear regression in log-log space
    valid = np.isfinite(log_radii) & np.isfinite(log_profile)
    if valid.sum() > 10:
        coeffs = np.polyfit(log_radii[valid], log_profile[valid], 1)
        beta = -coeffs[0]  # Power law exponent
        residuals = log_profile[valid] - np.polyval(coeffs, log_radii[valid])
        fit_quality = np.std(residuals)
    else:
        beta = 0
        fit_quality = 999

    # 3. Detect periodic peaks (grid artifacts from GAN upsampling)
    peak_score = _detect_periodic_peaks(magnitude, cx, cy)

    # 4. High-frequency energy ratio
    hf_ratio = _high_freq_energy_ratio(magnitude, cx, cy, max_radius)

    # 5. Spectral flatness (synthetic images tend to have flatter spectra in HF)
    spectral_flatness = _spectral_flatness(radial_profile)

    # Score calculation
    score = 0

    # Beta deviation from natural range (1.5-3.0 typical for natural images)
    if beta < 1.2 or beta > 3.5:
        score += 25
    elif beta < 1.5 or beta > 3.0:
        score += 10

    # Fit quality (higher residual variance = more anomalous)
    if fit_quality > 1.5:
        score += 20
    elif fit_quality > 0.8:
        score += 10

    # Periodic peaks
    score += min(30, peak_score * 10)

    # High frequency anomaly
    if hf_ratio > 0.15:
        score += 15
    elif hf_ratio < 0.01:
        score += 10  # Suspiciously low HF content (over-smoothed)

    # Spectral flatness
    if spectral_flatness > 0.6:
        score += 10

    score = min(100, max(0, score))

    # Generate visualization
    viz = _generate_visualization(power_spectrum)

    details = {
        'power_law_exponent': round(float(beta), 3),
        'fit_residual_std': round(float(fit_quality), 3),
        'periodic_peak_count': int(peak_score),
        'hf_energy_ratio': round(float(hf_ratio), 4),
        'spectral_flatness': round(float(spectral_flatness), 4),
        'findings': []
    }

    if beta < 1.2 or beta > 3.5:
        details['findings'].append(f'Espectro de frequência incompatível com imagem natural (β={beta:.2f}). Modelos generativos como Stable Diffusion e DALL-E produzem espectros com decaimento anômalo, diferente do padrão 1/f² de câmeras reais')
    if peak_score > 2:
        details['findings'].append(f'Detectados {peak_score} picos periódicos no espectro de Fourier — assinatura característica de redes neurais com camadas de upsampling (GAN/Diffusion), que criam artefatos de grade invisíveis ao olho humano')
    if hf_ratio > 0.15:
        details['findings'].append('Excesso de energia nas altas frequências — padrão típico de sharpening artificial aplicado por modelos generativos para simular nitidez de câmera')
    elif hf_ratio < 0.01:
        details['findings'].append('Ausência anormal de altas frequências — indica suavização excessiva característica de modelos de difusão, que eliminam micro-detalhes naturais como poros, grãos e ruído de sensor')
    if spectral_flatness > 0.6:
        details['findings'].append('Espectro plano nas altas frequências — imagens reais apresentam decaimento natural; espectro plano é assinatura de processamento por rede neural')
    if not details['findings']:
        details['findings'].append('Distribuição espectral compatível com captura por câmera real — sem artefatos de processamento neural detectados')

    # UI/Screenshot Detection Mitigation (UI has high periodic patterns naturally)
    ui_factor = utils.detect_ui_content(image_np)
    
    if ui_factor > 0.6:
        mitigation = int(score * 0.4 * ui_factor)
        score -= mitigation
        details['findings'].append(f'Interface Digital detectada ({int(ui_factor*100)}%) - picos periódicos mitigados')

    return {
        'name': 'Análise de Frequência (FFT)',
        'score': score,
        'details': details,
        'visualization': viz,
        'icon': '📊'
    }


def _azimuthal_average(magnitude, cx, cy, max_radius):
    """Compute radial average of power spectrum."""
    y, x = np.ogrid[:magnitude.shape[0], :magnitude.shape[1]]
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2).astype(int)
    r = np.clip(r, 0, max_radius - 1)

    profile = np.zeros(max_radius)
    counts = np.zeros(max_radius)
    np.add.at(profile, r.ravel(), magnitude.ravel())
    np.add.at(counts, r.ravel(), 1)
    counts[counts == 0] = 1
    return profile / counts


def _detect_periodic_peaks(magnitude, cx, cy):
    """Detect periodic peaks suggesting grid artifacts."""
    # Sample along axes
    h_line = magnitude[cy, :]
    v_line = magnitude[:, cx]

    peak_count = 0

    for line in [h_line, v_line]:
        # Normalize
        line_norm = line / (np.max(line) + 1e-10)
        # Remove DC component
        center = len(line_norm) // 2
        margin = max(5, len(line_norm) // 20)
        line_norm[center - margin:center + margin] = 0

        # Find peaks above threshold
        threshold = np.mean(line_norm) + 3 * np.std(line_norm)
        peaks = np.where(line_norm > threshold)[0]

        if len(peaks) > 2:
            # Check for periodicity in peak positions
            diffs = np.diff(peaks)
            if len(diffs) > 1:
                diff_std = np.std(diffs)
                diff_mean = np.mean(diffs)
                if diff_mean > 0 and diff_std / diff_mean < 0.3:
                    peak_count += len(peaks) // 2

    return peak_count


def _high_freq_energy_ratio(magnitude, cx, cy, max_radius):
    """Calculate ratio of high-frequency energy to total energy."""
    y, x = np.ogrid[:magnitude.shape[0], :magnitude.shape[1]]
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

    total_energy = np.sum(magnitude ** 2)
    hf_mask = r > max_radius * 0.7
    hf_energy = np.sum(magnitude[hf_mask] ** 2)

    return float(hf_energy / (total_energy + 1e-10))


def _spectral_flatness(profile):
    """Compute spectral flatness of the radial profile (high frequency portion)."""
    hf = profile[len(profile) // 2:]
    hf = hf[hf > 0]
    if len(hf) < 2:
        return 0.0
    geometric_mean = np.exp(np.mean(np.log(hf + 1e-10)))
    arithmetic_mean = np.mean(hf)
    return float(geometric_mean / (arithmetic_mean + 1e-10))


def _generate_visualization(power_spectrum):
    """Generate FFT magnitude spectrum visualization as base64 PNG."""
    # Normalize to 0-255
    ps_norm = power_spectrum - power_spectrum.min()
    ps_norm = (ps_norm / (ps_norm.max() + 1e-10) * 255).astype(np.uint8)

    # Apply colormap
    colored = cv2.applyColorMap(ps_norm, cv2.COLORMAP_JET)

    # Resize if too large
    max_dim = 512
    h, w = colored.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        colored = cv2.resize(colored, (int(w * scale), int(h * scale)))

    # Encode to base64
    _, buffer = cv2.imencode('.png', colored)
    return base64.b64encode(buffer).decode('utf-8')
