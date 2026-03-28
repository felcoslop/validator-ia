"""
Statistical Analysis Module
Analyzes pixel distributions, entropy, Benford's Law compliance, and adjacency correlations.
"""

import numpy as np
import cv2
import base64
from . import utils


def analyze(image_np):
    """
    Perform statistical analysis on pixel distributions.
    Returns score (0-100), details, and visualization.
    """
    gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)

    # 1. Shannon entropy
    entropy = _shannon_entropy(gray)

    # 2. Per-channel histogram analysis
    hist_anomaly = _histogram_anomaly(image_np)

    # 3. Benford's Law on DCT coefficients
    benford_score = _benford_analysis(gray)

    # 4. Pixel adjacency correlation
    adj_corr = _adjacency_correlation(gray)

    # 5. Chi-square test for uniformity of LSBs
    lsb_score = _lsb_analysis(gray)

    # 6. JPEG ghost detection
    jpeg_score = _jpeg_ghost(image_np)

    # Score — tuned for modern diffusion model detection
    score = 0

    # Entropy (natural images typically 6.5-7.5)
    if entropy < 4.5 or entropy > 7.9:
        score += 15
    elif entropy < 5.0:
        score += 8

    # Histogram anomalies
    score += min(15, int(hist_anomaly * 15))

    # Benford's Law violation (stronger weight - key AI indicator)
    score += min(15, int(benford_score * 15))

    # Adjacency correlation (AI diffusion models: very high correlation from smoothing)
    if adj_corr > 0.999:
        score += 20
    elif adj_corr > 0.997:
        score += 10

    # LSB analysis
    score += min(10, int(lsb_score * 12))

    # JPEG ghost
    score += min(10, int(jpeg_score * 10))

    score = min(100, max(0, score))

    viz = _generate_histogram_viz(image_np)

    details = {
        'shannon_entropy': round(float(entropy), 3),
        'histogram_anomaly': round(float(hist_anomaly), 4),
        'benford_deviation': round(float(benford_score), 4),
        'adjacency_correlation': round(float(adj_corr), 6),
        'lsb_score': round(float(lsb_score), 4),
        'findings': []
    }

    if entropy < 5.5 or entropy > 7.8:
        details['findings'].append({'key': 'finding_stat_low_entropy'})
    if hist_anomaly > 0.4:
        details['findings'].append({'key': 'finding_stat_unnatural_dist'})
    if benford_score > 0.3:
        details['findings'].append({'key': 'finding_stat_high_kurtosis'}) # Reusing as stat anomaly
    if adj_corr > 0.99:
        details['findings'].append({'key': 'finding_stat_unnatural_dist'}) # Generic stat anomaly
    if lsb_score > 0.4:
        details['findings'].append({'key': 'finding_stat_unnatural_dist'})
    if not details['findings']:
        details['findings'].append({'key': 'finding_stat_natural'})

    # UI/Screenshot Detection Mitigation
    ui_factor = utils.detect_ui_content(image_np)
    
    if ui_factor > 0.6:
        mitigation = int(score * 0.5 * ui_factor)
        score -= mitigation
        details['findings'].append(f'Conteúdo identificado como Interface Digital ({int(ui_factor*100)}%) - scores mitigados')

    return {
        'name': 'Análise Estatística',
        'score': score,
        'details': details,
        'visualization': viz,
        'icon': '📈'
    }


def _shannon_entropy(gray):
    """Compute Shannon entropy of grayscale image."""
    hist, _ = np.histogram(gray, bins=256, range=(0, 255), density=True)
    hist = hist[hist > 0]
    return float(-np.sum(hist * np.log2(hist)))


def _histogram_anomaly(image_np):
    """Detect anomalies in color channel histograms."""
    anomaly = 0.0
    for c in range(3):
        hist, _ = np.histogram(image_np[:, :, c], bins=256, range=(0, 255))
        hist = hist.astype(np.float64)

        # Check for gaps (zero bins in middle range)
        mid_range = hist[20:236]
        total_bins = len(mid_range)
        zero_bins = np.sum(mid_range == 0)
        gap_ratio = zero_bins / total_bins

        # Check for comb pattern (alternating high/low)
        diffs = np.abs(np.diff(mid_range))
        mean_diff = np.mean(diffs)
        mean_val = np.mean(mid_range) + 1e-10
        comb_ratio = mean_diff / mean_val

        anomaly += gap_ratio * 0.5 + max(0, comb_ratio - 0.5) * 0.5

    return float(min(1.0, anomaly / 3.0))


def _benford_analysis(gray):
    """Apply Benford's Law to DCT coefficients."""
    # Apply DCT in 8x8 blocks
    h, w = gray.shape
    h8, w8 = h - h % 8, w - w % 8
    gray_crop = gray[:h8, :w8].astype(np.float64)

    first_digits = []
    for i in range(0, h8, 8):
        for j in range(0, w8, 8):
            block = gray_crop[i:i + 8, j:j + 8]
            dct_block = cv2.dct(block)
            # Skip DC coefficient
            coeffs = np.abs(dct_block.ravel()[1:])
            coeffs = coeffs[coeffs >= 1.0]
            if len(coeffs) > 0:
                digits = (coeffs / (10 ** np.floor(np.log10(coeffs)))).astype(int)
                digits = digits[(digits >= 1) & (digits <= 9)]
                first_digits.extend(digits.tolist())

    if len(first_digits) < 100:
        return 0.0

    first_digits = np.array(first_digits)

    # Expected Benford distribution
    expected = np.log10(1 + 1.0 / np.arange(1, 10))

    # Observed distribution
    observed = np.zeros(9)
    for d in range(1, 10):
        observed[d - 1] = np.sum(first_digits == d) / len(first_digits)

    # Chi-square-like deviation
    deviation = np.sum((observed - expected) ** 2 / (expected + 1e-10))

    return float(min(1.0, deviation * 5))


def _adjacency_correlation(gray):
    """Compute horizontal pixel adjacency correlation."""
    x = gray[:, :-1].ravel().astype(np.float64)
    y = gray[:, 1:].ravel().astype(np.float64)

    # Subsample for speed
    if len(x) > 500000:
        idx = np.random.choice(len(x), 500000, replace=False)
        x = x[idx]
        y = y[idx]

    corr = np.corrcoef(x, y)[0, 1]
    return float(corr)


def _lsb_analysis(gray):
    """Analyze least significant bit patterns."""
    lsb = gray & 1
    # LSB should be roughly 50/50 in natural images
    ratio = np.mean(lsb)
    # Also check for patterns
    h_diff = np.mean(np.abs(np.diff(lsb.astype(np.float64), axis=1)))

    bias = abs(ratio - 0.5) * 4  # 0 to ~1
    pattern = max(0, 1.0 - h_diff * 2)  # Low diff = patterned

    return float(min(1.0, bias + pattern * 0.5))


def _jpeg_ghost(image_np):
    """Simple JPEG ghost analysis."""
    from io import BytesIO
    from PIL import Image

    img_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)

    differences = []
    for q in [60, 70, 80, 90]:
        buf = BytesIO()
        pil_img.save(buf, format='JPEG', quality=q)
        buf.seek(0)
        recomp = np.array(Image.open(buf)).astype(np.float64)
        diff = np.mean(np.abs(img_rgb.astype(np.float64) - recomp))
        differences.append(diff)

    # If there's a quality level with very low difference, the image was saved at that quality
    min_diff = min(differences)
    if min_diff < 1.0:
        return 0.0  # Consistent with a specific JPEG quality

    # Large variance in differences can indicate manipulation
    variance = np.std(differences)
    return float(min(1.0, variance / 5.0))


def _generate_histogram_viz(image_np):
    """Generate color histogram visualization."""
    viz = np.zeros((200, 256, 3), dtype=np.uint8)
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # BGR

    for c, color in enumerate(colors):
        hist = cv2.calcHist([image_np], [c], None, [256], [0, 256])
        hist = hist / (hist.max() + 1e-10) * 180
        for x in range(256):
            h_val = int(hist[x, 0])
            cv2.line(viz, (x, 200), (x, 200 - h_val), color, 1)

    _, buffer = cv2.imencode('.png', viz)
    return base64.b64encode(buffer).decode('utf-8')
