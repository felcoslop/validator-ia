"""
Noise Pattern Analysis Module (PRNU-like)
Extracts noise residual W = I - f(I) using wavelet denoising.
Real images have sensor-specific noise (PRNU); AI images have structured/synthetic noise.
"""

import numpy as np
import cv2
import base64
from skimage.restoration import denoise_wavelet
from . import utils


def analyze(image_np):
    """
    Extract and analyze noise residual patterns.
    Returns score (0-100), details, and visualization.
    """
    # Convert to float [0,1]
    img_float = image_np.astype(np.float64) / 255.0

    # Extract noise residual per channel: W = I - f(I)
    residuals = []
    for c in range(3):
        channel = img_float[:, :, c]
        denoised = denoise_wavelet(channel, method='BayesShrink',
                                   mode='soft', wavelet='db4',
                                   rescale_sigma=True)
        residual = channel - denoised
        residuals.append(residual)

    noise_residual = np.stack(residuals, axis=-1)
    noise_gray = np.mean(noise_residual, axis=2)

    h, w = noise_gray.shape

    # 1. Noise variance (natural images have specific noise levels)
    noise_var = np.var(noise_gray)

    # 2. Local variance consistency (PRNU is spatially consistent)
    block_size = max(16, min(h, w) // 16)
    local_vars = _compute_local_variance(noise_gray, block_size)
    var_of_vars = np.var(local_vars) / (np.mean(local_vars) + 1e-10)

    # 3. Autocorrelation analysis (detect periodicity in noise)
    autocorr_score = _autocorrelation_analysis(noise_gray)

    # 4. Noise histogram analysis (should be near-Gaussian for real cameras)
    gaussianity = _test_gaussianity(noise_gray)

    # 5. Structured noise detection (block patterns from neural networks)
    structure_score = _detect_structured_noise(noise_gray)

    # Score calculation — tuned for modern diffusion model detection
    score = 0

    # Noise level: AI images have artificially clean noise
    if noise_var < 5e-5:
        score += 35  # Very suspiciously clean (strong AI signal)
    elif noise_var < 5e-4:
        score += 25  # Suspiciously clean
    elif noise_var < 2e-3:
        score += 15  # Somewhat clean
    elif noise_var > 0.01:
        score += 10  # Abnormally noisy

    # Local variance consistency: AI has too-uniform noise distribution
    if var_of_vars < 0.01:
        score += 25  # Extremely uniform (very strong AI signal)
    elif var_of_vars < 0.05:
        score += 20  # Too uniform (synthetic)
    elif var_of_vars < 0.2:
        score += 10  # Somewhat uniform

    # Autocorrelation (periodic patterns from neural networks)
    score += min(15, int(autocorr_score * 20))

    # Non-Gaussian noise (real camera noise is Gaussian; AI noise is not)
    if gaussianity < 0.2:
        score += 10
    elif gaussianity < 0.4:
        score += 5

    # Structured patterns from neural network architecture
    score += min(15, int(structure_score * 15))

    score = min(100, max(0, score))

    # Generate visualization
    viz = _generate_noise_map(noise_gray)

    details = {
        'noise_variance': round(float(noise_var), 6),
        'variance_consistency': round(float(var_of_vars), 4),
        'autocorrelation_score': round(float(autocorr_score), 4),
        'gaussianity': round(float(gaussianity), 4),
        'structure_score': round(float(structure_score), 4),
        'findings': []
    }

    if noise_var < 5e-4:
        details['findings'].append({'key': 'finding_noise_incompatible'})
    elif noise_var > 0.01:
        details['findings'].append({'key': 'finding_noise_excessive'})
    if var_of_vars < 0.05:
        details['findings'].append({'key': 'finding_noise_uniform'})
    elif var_of_vars > 2.0:
        details['findings'].append({'key': 'finding_noise_irregular'})
    if autocorr_score > 0.5:
        details['findings'].append({'key': 'finding_noise_periodic'})
    if gaussianity < 0.6:
        details['findings'].append({'key': 'finding_noise_non_gaussian'})
    if structure_score > 0.4:
        details['findings'].append({'key': 'finding_noise_structured'})
    if not details['findings']:
        details['findings'].append({'key': 'finding_noise_natural'})

    # UI/Screenshot Detection Mitigation (PRNU doesn't exist for UI)
    ui_factor = utils.detect_ui_content(image_np)
    
    if ui_factor > 0.6:
        mitigation = int(score * 0.6 * ui_factor)
        score -= mitigation
        details['findings'].append({'key': 'finding_ui_detected', 'percent': int(ui_factor*100)})

    return {
        'name': 'Análise de Ruído (PRNU)',
        'score': score,
        'details': details,
        'visualization': viz,
        'icon': '🔬'
    }


def _compute_local_variance(noise, block_size):
    """Compute variance in non-overlapping blocks."""
    h, w = noise.shape
    variances = []
    for i in range(0, h - block_size, block_size):
        for j in range(0, w - block_size, block_size):
            block = noise[i:i + block_size, j:j + block_size]
            variances.append(np.var(block))
    return np.array(variances)


def _autocorrelation_analysis(noise):
    """Check for periodic patterns via autocorrelation."""
    # Sample center crop for efficiency
    h, w = noise.shape
    cs = min(256, min(h, w))
    crop = noise[h // 2 - cs // 2:h // 2 + cs // 2, w // 2 - cs // 2:w // 2 + cs // 2]

    # 2D autocorrelation via FFT
    f = np.fft.fft2(crop)
    acf = np.real(np.fft.ifft2(f * np.conj(f)))
    acf = np.fft.fftshift(acf)

    cy, cx = acf.shape[0] // 2, acf.shape[1] // 2
    acf_norm = acf / (acf[cy, cx] + 1e-10)

    # Mask center
    y, x = np.ogrid[:acf.shape[0], :acf.shape[1]]
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    mask = (r > 5) & (r < cs // 2)

    if mask.sum() > 0:
        secondary_peaks = acf_norm[mask]
        peak_val = np.max(secondary_peaks)
        return float(max(0, peak_val - 0.1))
    return 0.0


def _test_gaussianity(noise):
    """Test how Gaussian the noise distribution is (1.0 = perfectly Gaussian)."""
    flat = noise.ravel()
    # Subsample for speed
    if len(flat) > 100000:
        flat = np.random.choice(flat, 100000, replace=False)

    # Kurtosis test (Gaussian kurtosis = 3)
    mean = np.mean(flat)
    std = np.std(flat)
    if std < 1e-10:
        return 0.0

    normalized = (flat - mean) / std
    kurtosis = np.mean(normalized ** 4)
    # Skewness
    skewness = abs(np.mean(normalized ** 3))

    # Score: 1.0 for perfect Gaussian, 0.0 for very non-Gaussian
    kurt_score = max(0, 1.0 - abs(kurtosis - 3.0) / 5.0)
    skew_score = max(0, 1.0 - skewness / 2.0)

    return float((kurt_score + skew_score) / 2.0)


def _detect_structured_noise(noise):
    """Detect structured patterns (block artifacts, grid patterns)."""
    h, w = noise.shape

    # Check for block boundary artifacts (8x8, 16x16 blocks common in NN)
    score = 0.0

    for block_sz in [8, 16, 32]:
        if h < block_sz * 4 or w < block_sz * 4:
            continue

        # Horizontal block boundaries
        h_diffs = []
        for i in range(block_sz, h - block_sz, block_sz):
            diff = np.mean(np.abs(noise[i, :] - noise[i - 1, :]))
            h_diffs.append(diff)

        # Non-boundary differences
        non_boundary = []
        for i in range(1, min(h - 1, block_sz * 10)):
            if i % block_sz != 0:
                diff = np.mean(np.abs(noise[i, :] - noise[i - 1, :]))
                non_boundary.append(diff)

        if h_diffs and non_boundary:
            ratio = np.mean(h_diffs) / (np.mean(non_boundary) + 1e-10)
            if ratio > 1.3:
                score = max(score, min(1.0, (ratio - 1.0) / 2.0))

    return score


def _generate_noise_map(noise_gray):
    """Generate amplified noise map visualization."""
    # Amplify and normalize
    amplified = noise_gray * 20  # Amplify noise
    amplified = amplified - amplified.min()
    amplified = (amplified / (amplified.max() + 1e-10) * 255).astype(np.uint8)

    colored = cv2.applyColorMap(amplified, cv2.COLORMAP_INFERNO)

    max_dim = 512
    h, w = colored.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        colored = cv2.resize(colored, (int(w * scale), int(h * scale)))

    _, buffer = cv2.imencode('.png', colored)
    return base64.b64encode(buffer).decode('utf-8')
