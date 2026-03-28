"""
Horizontal/Vertical Score Analysis Module
Calculates accumulated energy along horizontal and vertical axes of the frequency spectrum.
Detects anisotropic processing artifacts from separable convolutions or block-based attention.
S_h = Σ_j |F(f_h, f_j)|², S_v = Σ_i |F(f_i, f_v)|²
"""

import numpy as np
import cv2
import base64
from . import utils


def analyze(image_np):
    """
    Compute H/V energy scores from the frequency spectrum.
    Returns score (0-100), details, and visualization.
    """
    gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY).astype(np.float64)
    h, w = gray.shape

    # Apply window
    win_h = np.hanning(h)
    win_w = np.hanning(w)
    window = np.outer(win_h, win_w)
    gray_win = gray * window

    # FFT
    f_transform = np.fft.fft2(gray_win)
    f_shift = np.fft.fftshift(f_transform)
    magnitude = np.abs(f_shift) ** 2  # Power spectrum

    cy, cx = h // 2, w // 2

    # Remove DC component
    dc_radius = 5
    y, x = np.ogrid[:h, :w]
    dc_mask = ((x - cx) ** 2 + (y - cy) ** 2) > dc_radius ** 2

    magnitude_masked = magnitude * dc_mask

    # Horizontal score: sum along vertical axis (columns) for each horizontal frequency
    s_h = np.sum(magnitude_masked, axis=0)  # Shape: (w,)

    # Vertical score: sum along horizontal axis (rows) for each vertical frequency
    s_v = np.sum(magnitude_masked, axis=1)  # Shape: (h,)

    # Normalize
    s_h_norm = s_h / (np.max(s_h) + 1e-10)
    s_v_norm = s_v / (np.max(s_v) + 1e-10)

    # 1. Detect periodic peaks in H and V scores
    h_peaks = _detect_peaks(s_h_norm, cx)
    v_peaks = _detect_peaks(s_v_norm, cy)

    # 2. Smoothness of H/V profiles (natural: smooth curve; synthetic: peaks)
    h_smoothness = _profile_smoothness(s_h_norm)
    v_smoothness = _profile_smoothness(s_v_norm)

    # 3. H/V ratio (anisotropy index)
    total_h = np.sum(s_h)
    total_v = np.sum(s_v)
    hv_ratio = total_h / (total_v + 1e-10)
    anisotropy = abs(np.log(hv_ratio + 1e-10))

    # 4. Spectral peaks outside natural frequencies
    h_anomaly = _spectral_anomaly(s_h_norm, cx)
    v_anomaly = _spectral_anomaly(s_v_norm, cy)

    # Score
    score = 0

    # Periodic peaks (strong GAN indicator)
    score += min(20, (h_peaks + v_peaks) * 3)

    # Non-smooth profiles
    mean_smoothness = (h_smoothness + v_smoothness) / 2
    if mean_smoothness > 0.5:
        score += 15
    elif mean_smoothness > 0.3:
        score += 8

    # Anisotropy
    if anisotropy > 0.8:
        score += 10

    # Spectral anomalies
    anomaly_score = (h_anomaly + v_anomaly) / 2
    score += min(15, int(anomaly_score * 15))

    score = min(100, max(0, score))

    # Visualization
    viz = _generate_hv_viz(s_h_norm, s_v_norm)

    details = {
        'h_periodic_peaks': int(h_peaks),
        'v_periodic_peaks': int(v_peaks),
        'h_smoothness': round(float(h_smoothness), 4),
        'v_smoothness': round(float(v_smoothness), 4),
        'hv_ratio': round(float(hv_ratio), 4),
        'anisotropy_index': round(float(anisotropy), 4),
        'h_anomaly': round(float(h_anomaly), 4),
        'v_anomaly': round(float(v_anomaly), 4),
        'findings': []
    }

    if h_peaks + v_peaks > 2:
        details['findings'].append({'key': 'finding_temporal_low_var'}) # Reusing as periodic peak indicator
    if mean_smoothness > 0.3:
        details['findings'].append({'key': 'finding_hv_asymmetry'})
    if anisotropy > 0.5:
        details['findings'].append({'key': 'finding_hv_asymmetry'})
    if anomaly_score > 0.5:
        details['findings'].append({'key': 'finding_hv_asymmetry'})
    if not details['findings']:
        details['findings'].append({'key': 'finding_hv_natural'})

    # UI/Screenshot Detection Mitigation (UI is all H and V lines)
    ui_factor = utils.detect_ui_content(image_np)
    
    if ui_factor > 0.6:
        mitigation = int(score * 0.7 * ui_factor)
        score -= mitigation
        details['findings'].append(f'Interface Digital detectada ({int(ui_factor*100)}%) - padrões H/V mitigados')

    return {
        'name': 'Score Horizontal/Vertical',
        'score': score,
        'details': details,
        'visualization': viz,
        'icon': '📐'
    }


def _detect_peaks(profile, center):
    """Detect periodic peaks in a 1D spectral profile."""
    # Remove center region
    margin = max(5, len(profile) // 20)
    profile_masked = profile.copy()
    profile_masked[center - margin:center + margin] = 0

    # Find peaks above threshold
    threshold = np.mean(profile_masked) + 3 * np.std(profile_masked)
    peaks = np.where(profile_masked > threshold)[0]

    if len(peaks) < 3:
        return 0

    # Check for periodicity
    diffs = np.diff(peaks)
    if len(diffs) < 2:
        return 0

    mean_diff = np.mean(diffs)
    std_diff = np.std(diffs)

    if mean_diff > 0 and std_diff / (mean_diff + 1e-10) < 0.3:
        return len(peaks)

    return max(0, len(peaks) - 3)


def _profile_smoothness(profile):
    """Measure roughness of spectral profile (0 = smooth, 1 = rough)."""
    if len(profile) < 10:
        return 0.0

    # Compute second derivative
    d2 = np.diff(profile, 2)
    roughness = np.std(d2)

    return float(min(1.0, roughness * 10))


def _spectral_anomaly(profile, center):
    """Detect anomalous spectral features."""
    if len(profile) < 20:
        return 0.0

    # Fit smooth trend
    x = np.arange(len(profile))
    coeffs = np.polyfit(x, profile, 5)
    trend = np.polyval(coeffs, x)

    residual = np.abs(profile - trend)
    anomaly = np.sum(residual > 3 * np.std(residual)) / len(profile)

    return float(min(1.0, anomaly * 10))


def _generate_hv_viz(s_h, s_v):
    """Generate H/V score visualization."""
    viz_h = 300
    viz_w = 512
    viz = np.zeros((viz_h, viz_w, 3), dtype=np.uint8)
    viz[:] = (20, 20, 30)  # Dark background

    # Draw horizontal score (cyan)
    h_resampled = np.interp(
        np.linspace(0, len(s_h) - 1, viz_w),
        np.arange(len(s_h)), s_h
    )
    for i in range(1, viz_w):
        y1 = int((1 - h_resampled[i - 1]) * (viz_h - 40) + 20)
        y2 = int((1 - h_resampled[i]) * (viz_h - 40) + 20)
        cv2.line(viz, (i - 1, y1), (i, y2), (255, 200, 0), 1)

    # Draw vertical score (magenta)
    v_resampled = np.interp(
        np.linspace(0, len(s_v) - 1, viz_w),
        np.arange(len(s_v)), s_v
    )
    for i in range(1, viz_w):
        y1 = int((1 - v_resampled[i - 1]) * (viz_h - 40) + 20)
        y2 = int((1 - v_resampled[i]) * (viz_h - 40) + 20)
        cv2.line(viz, (i - 1, y1), (i, y2), (200, 0, 255), 1)

    # Labels
    cv2.putText(viz, 'H Score', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)
    cv2.putText(viz, 'V Score', (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 0, 255), 1)

    _, buffer = cv2.imencode('.png', viz)
    return base64.b64encode(buffer).decode('utf-8')
