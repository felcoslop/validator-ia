"""
Wavelet Decomposition Analysis Module
Multi-level DWT to analyze sub-band energy distribution.
AI-generated images show abnormal patterns in high-frequency sub-bands.
"""

import numpy as np
import cv2
import pywt
import base64


def analyze(image_np):
    """
    Perform wavelet decomposition analysis.
    Returns score (0-100), details, and visualization.
    """
    gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY).astype(np.float64)

    # Multi-level wavelet decomposition
    wavelet = 'db4'
    max_level = min(5, pywt.dwt_max_level(min(gray.shape), wavelet))
    coeffs = pywt.wavedec2(gray, wavelet, level=max_level)

    # Analyze energy in each sub-band at each level
    energies = []
    for level in range(1, len(coeffs)):
        cH, cV, cD = coeffs[level]
        energy_h = np.mean(cH ** 2)
        energy_v = np.mean(cV ** 2)
        energy_d = np.mean(cD ** 2)
        total = energy_h + energy_v + energy_d + 1e-10
        energies.append({
            'level': level,
            'horizontal': float(energy_h),
            'vertical': float(energy_v),
            'diagonal': float(energy_d),
            'total': float(total),
            'h_ratio': float(energy_h / total),
            'v_ratio': float(energy_v / total),
            'd_ratio': float(energy_d / total),
        })

    # 1. Energy decay across levels (natural: smooth exponential decay)
    total_energies = [e['total'] for e in energies]
    if len(total_energies) >= 3:
        log_energies = np.log(np.array(total_energies) + 1e-10)
        decay_fit = np.polyfit(np.arange(len(log_energies)), log_energies, 1)
        decay_rate = -decay_fit[0]
        decay_residual = np.std(log_energies - np.polyval(decay_fit, np.arange(len(log_energies))))
    else:
        decay_rate = 0
        decay_residual = 0

    # 2. Isotropy check (H ≈ V ≈ D ratios at each level)
    anisotropy = 0.0
    for e in energies:
        ideal = 1.0 / 3.0
        dev = abs(e['h_ratio'] - ideal) + abs(e['v_ratio'] - ideal) + abs(e['d_ratio'] - ideal)
        anisotropy += dev
    anisotropy /= (len(energies) + 1e-10)

    # 3. Sub-band kurtosis (natural images have specific statistical properties)
    kurtosis_scores = []
    for level in range(1, len(coeffs)):
        for sub in coeffs[level]:
            flat = sub.ravel()
            if len(flat) > 100:
                mu = np.mean(flat)
                std = np.std(flat)
                if std > 1e-10:
                    kurt = np.mean(((flat - mu) / std) ** 4)
                    kurtosis_scores.append(kurt)

    mean_kurtosis = np.mean(kurtosis_scores) if kurtosis_scores else 3.0

    # 4. Fine-scale noise analysis (highest level detail)
    if len(coeffs) > 1:
        finest_detail = coeffs[1]
        fine_energy = sum(np.mean(s ** 2) for s in finest_detail)
        coarse_energy = sum(np.mean(s ** 2) for s in coeffs[-1]) if len(coeffs) > 2 else fine_energy
        fine_coarse_ratio = fine_energy / (coarse_energy + 1e-10)
    else:
        fine_coarse_ratio = 1.0

    # Score
    score = 0

    # Non-smooth energy decay
    if decay_residual > 1.0:
        score += 20
    elif decay_residual > 0.5:
        score += 10

    # Kurtosis deviation (natural sub-bands: kurtosis ~6-20; AI: extreme peaks)
    if mean_kurtosis > 100:
        score += 40
    elif mean_kurtosis > 50 or mean_kurtosis < 4:
        score += 25
    elif mean_kurtosis > 30 or mean_kurtosis < 5:
        score += 15
        
    # Anisotropy (strong directional bias suggests processing artifacts)
    if anisotropy > 0.6:
        score += 30
    elif anisotropy > 0.4:
        score += 20
    elif anisotropy > 0.3:
        score += 10
    elif fine_coarse_ratio > 100:
        score += 10  # Abnormal fine detail energy

    # Fine/coarse ratio
    if fine_coarse_ratio < 0.001:
        score += 15  # Suspiciously smooth fine details

    # Abnormal decay rate
    if decay_rate < 0.3 or decay_rate > 3.0:
        score += 15

    score = min(100, max(0, score))

    # Visualization
    viz = _generate_wavelet_viz(coeffs, gray.shape)

    details = {
        'levels_analyzed': len(energies),
        'decay_rate': round(float(decay_rate), 3),
        'decay_residual': round(float(decay_residual), 3),
        'anisotropy': round(float(anisotropy), 4),
        'mean_kurtosis': round(float(mean_kurtosis), 2),
        'fine_coarse_ratio': round(float(fine_coarse_ratio), 4),
        'sub_band_energies': energies,
        'findings': []
    }

    if decay_residual > 1.0:
        details['findings'].append({'key': 'finding_wavelet_hf_anomaly'})
    if anisotropy > 0.5:
        details['findings'].append({'key': 'finding_wavelet_subband_imbalance'})
    if mean_kurtosis < 4 or mean_kurtosis > 50:
        details['findings'].append({'key': 'finding_wavelet_hf_anomaly'}) # Generic wavelet anomaly
    if fine_coarse_ratio < 0.001:
        details['findings'].append({'key': 'finding_wavelet_hf_anomaly'})
    if not details['findings']:
        details['findings'].append({'key': 'finding_wavelet_natural'})

    return {
        'name': 'Análise Wavelet (DWT)',
        'score': score,
        'details': details,
        'visualization': viz,
        'icon': '〰️'
    }


def _generate_wavelet_viz(coeffs, original_shape):
    """Generate wavelet decomposition visualization."""
    # Reconstruct a visualization showing sub-bands
    arr, slices = pywt.coeffs_to_array(coeffs)

    # Normalize
    arr_norm = np.abs(arr)
    arr_norm = arr_norm / (arr_norm.max() + 1e-10) * 255
    arr_uint8 = arr_norm.astype(np.uint8)

    colored = cv2.applyColorMap(arr_uint8, cv2.COLORMAP_MAGMA)

    max_dim = 512
    h, w = colored.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        colored = cv2.resize(colored, (int(w * scale), int(h * scale)))

    _, buffer = cv2.imencode('.png', colored)
    return base64.b64encode(buffer).decode('utf-8')
