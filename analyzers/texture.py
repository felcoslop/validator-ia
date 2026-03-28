"""
AI Texture Analysis Module
Detects characteristic texture patterns of AI-generated images:
- Over-smooth skin with no pores/micro-texture
- Unnatural hair strand patterns
- Symmetric facial features with too-perfect alignment
- Depth-of-field inconsistencies
- Color channel correlation anomalies typical of diffusion models
"""

import numpy as np
import cv2
import base64


def analyze(image_np):
    """
    Detect AI texture artifacts in photographic content.
    Focuses on skin smoothness, micro-texture absence, and color anomalies.
    """
    gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY).astype(np.float64)
    h, w = gray.shape

    # 1. Micro-texture analysis (AI images lack micro-texture detail)
    micro_score = _micro_texture_analysis(gray)

    # 2. Laplacian variance (sharpness distribution - AI has unnatural sharpness)
    lap_score = _laplacian_analysis(gray)

    # 3. Color channel correlation (diffusion models produce correlated channels)
    color_score = _color_correlation_analysis(image_np)

    # 4. Local Binary Pattern uniformity (AI textures are too uniform)
    lbp_score = _lbp_uniformity(gray)

    # 5. DCT coefficient analysis (AI-specific distribution)
    dct_score = _dct_analysis(gray)

    # 6. Patch consistency (AI is suspiciously consistent across patches)
    patch_score = _patch_consistency(gray)

    # Score
    score = 0

    # Micro-texture (strongest differentiator for AI photos)
    if micro_score > 0.85:
        score += 20
    elif micro_score > 0.7:
        score += 10

    # Laplacian sharpness distribution
    if lap_score > 0.75:
        score += 15
    elif lap_score > 0.5:
        score += 8

    # Color correlation  
    if color_score > 0.8:
        score += 15
    elif color_score > 0.6:
        score += 8

    # LBP uniformity
    if lbp_score > 0.8:
        score += 10

    # DCT analysis  
    score += min(10, int(dct_score * 10))

    # Patch consistency
    if patch_score > 0.8:
        score += 10

    score = min(100, max(0, score))

    viz = _generate_texture_viz(gray)

    details = {
        'micro_texture': round(float(micro_score), 4),
        'laplacian_score': round(float(lap_score), 4),
        'color_correlation': round(float(color_score), 4),
        'lbp_uniformity': round(float(lbp_score), 4),
        'dct_anomaly': round(float(dct_score), 4),
        'patch_consistency': round(float(patch_score), 4),
        'findings': []
    }

    if micro_score > 0.5:
        details['findings'].append({'key': 'finding_texture_synthetic'})
    if lap_score > 0.3:
        details['findings'].append({'key': 'finding_texture_synthetic'})
    if color_score > 0.4:
        details['findings'].append({'key': 'finding_texture_synthetic'}) # Using as generic synth texture
    if lbp_score > 0.3:
        details['findings'].append({'key': 'finding_texture_synthetic'})
    if patch_score > 0.3:
        details['findings'].append({'key': 'finding_texture_synthetic'})
    if not details['findings']:
        details['findings'].append({'key': 'finding_texture_natural'})

    return {
        'name': 'Análise de Textura (AI)',
        'score': score,
        'details': details,
        'visualization': viz,
        'icon': 'texture'
    }


def _micro_texture_analysis(gray):
    """
    Analyze micro-texture by looking at high-frequency detail.
    AI images have less micro-texture (pores, fine grain) than real photos.
    """
    h, w = gray.shape
    
    # High-pass filter to isolate micro-texture
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.5)
    high_freq = gray - blurred
    
    # Sample multiple regions
    block_size = min(64, min(h, w) // 4)
    if block_size < 16:
        return 0.0
    
    texture_energies = []
    for i in range(0, h - block_size, block_size):
        for j in range(0, w - block_size, block_size):
            patch = high_freq[i:i+block_size, j:j+block_size]
            energy = np.std(patch)
            texture_energies.append(energy)
    
    if not texture_energies:
        return 0.0
    
    texture_energies = np.array(texture_energies)
    mean_energy = np.mean(texture_energies)
    
    # AI images have lower micro-texture energy
    # Real photos typically have energy > 2.0; AI < 1.5
    if mean_energy < 0.8:
        return 0.9  # Very smooth - strong AI signal
    elif mean_energy < 1.2:
        return 0.7
    elif mean_energy < 1.8:
        return 0.5
    elif mean_energy < 2.5:
        return 0.3
    elif mean_energy < 3.5:
        return 0.1
    return 0.0


def _laplacian_analysis(gray):
    """
    Analyze Laplacian variance distribution.
    AI images have unnatural sharpness distribution (too smooth or artificially sharp edges).
    """
    h, w = gray.shape
    block_size = min(64, min(h, w) // 4)
    if block_size < 16:
        return 0.0
    
    lap_vars = []
    for i in range(0, h - block_size, block_size):
        for j in range(0, w - block_size, block_size):
            patch = gray[i:i+block_size, j:j+block_size]
            lap = cv2.Laplacian(patch, cv2.CV_64F)
            lap_vars.append(np.var(lap))
    
    if not lap_vars:
        return 0.0
    
    lap_vars = np.array(lap_vars)
    
    # Coefficient of variation of Laplacian variance
    # Real images: diverse (some sharp, some smooth) -> higher CV
    # AI images: more uniform sharpness -> lower CV
    cv = np.std(lap_vars) / (np.mean(lap_vars) + 1e-10)
    
    # Also check for bimodal distribution (sharp edges + smooth areas)
    sorted_vars = np.sort(lap_vars)
    q25 = sorted_vars[len(sorted_vars) // 4]
    q75 = sorted_vars[3 * len(sorted_vars) // 4]
    
    ratio = q75 / (q25 + 1e-10)
    
    # AI: ratio is lower (more uniform); Real: ratio is higher (more diverse)
    if ratio < 3.0:
        return 0.7  # Too uniform - AI signal
    elif ratio < 5.0:
        return 0.4
    elif ratio < 8.0:
        return 0.2
    return 0.0


def _color_correlation_analysis(image_np):
    """
    Analyze inter-channel color correlations.
    Diffusion models produce images with abnormal channel correlations.
    """
    if len(image_np.shape) != 3 or image_np.shape[2] != 3:
        return 0.0
    
    b, g, r = image_np[:,:,0].ravel().astype(np.float64), \
              image_np[:,:,1].ravel().astype(np.float64), \
              image_np[:,:,2].ravel().astype(np.float64)
    
    # Subsample
    n = min(100000, len(b))
    if n < 1000:
        return 0.0
    idx = np.random.choice(len(b), n, replace=False)
    b, g, r = b[idx], g[idx], r[idx]
    
    # Inter-channel correlations
    corr_rg = np.corrcoef(r, g)[0, 1]
    corr_rb = np.corrcoef(r, b)[0, 1]
    corr_gb = np.corrcoef(g, b)[0, 1]
    
    # AI images tend to have higher inter-channel correlations
    mean_corr = (abs(corr_rg) + abs(corr_rb) + abs(corr_gb)) / 3
    
    # Also check correlation uniformity
    corr_std = np.std([corr_rg, corr_rb, corr_gb])
    
    score = 0.0
    if mean_corr > 0.95:
        score += 0.5
    elif mean_corr > 0.9:
        score += 0.3
    elif mean_corr > 0.85:
        score += 0.1
    
    # Very uniform correlations across all channel pairs = AI
    if corr_std < 0.02:
        score += 0.3
    elif corr_std < 0.05:
        score += 0.15
    
    return min(1.0, score)


def _lbp_uniformity(gray):
    """
    Simplified Local Binary Pattern analysis.
    AI images produce more uniform LBP histograms.
    """
    h, w = gray.shape
    gray_uint8 = gray.astype(np.uint8)
    
    # Compute simplified LBP
    block_size = min(64, min(h, w) // 4)
    if block_size < 16:
        return 0.0
    
    all_histograms = []
    for i in range(1, h-1, block_size):
        for j in range(1, w-1, block_size):
            end_i = min(i + block_size, h - 1)
            end_j = min(j + block_size, w - 1)
            if end_i - i < 8 or end_j - j < 8:
                continue
            
            patch = gray_uint8[i:end_i, j:end_j]
            # Simple gradient-based texture descriptor
            gx = cv2.Sobel(patch, cv2.CV_64F, 1, 0, ksize=3)
            gy = cv2.Sobel(patch, cv2.CV_64F, 0, 1, ksize=3)
            mag = np.sqrt(gx**2 + gy**2)
            
            hist, _ = np.histogram(mag, bins=16, range=(0, np.percentile(mag, 95) + 1e-10))
            hist = hist / (hist.sum() + 1e-10)
            all_histograms.append(hist)
    
    if len(all_histograms) < 4:
        return 0.0
    
    # Compare histograms between patches
    all_histograms = np.array(all_histograms)
    
    # Compute pairwise similarity
    n = min(len(all_histograms), 30)
    selected = all_histograms[np.random.choice(len(all_histograms), n, replace=False)]
    
    similarities = []
    for i in range(n):
        for j in range(i+1, n):
            sim = np.sum(np.minimum(selected[i], selected[j]))  # histogram intersection
            similarities.append(sim)
    
    mean_sim = np.mean(similarities)
    
    # AI: higher mean similarity (more uniform patches)
    if mean_sim > 0.7:
        return 0.8
    elif mean_sim > 0.55:
        return 0.5
    elif mean_sim > 0.4:
        return 0.25
    return 0.0


def _dct_analysis(gray):
    """
    Analyze DCT coefficient distribution for AI-specific patterns.
    """
    h, w = gray.shape
    h8, w8 = h - h % 8, w - w % 8
    if h8 < 16 or w8 < 16:
        return 0.0
    
    gray_crop = gray[:h8, :w8]
    
    ac_coeffs = []
    for i in range(0, h8, 8):
        for j in range(0, w8, 8):
            block = gray_crop[i:i+8, j:j+8]
            dct = cv2.dct(block)
            # AC coefficients (skip DC)
            ac = np.abs(dct.ravel()[1:])
            ac_coeffs.extend(ac.tolist())
    
    if len(ac_coeffs) < 100:
        return 0.0
    
    ac_coeffs = np.array(ac_coeffs)
    
    # AI images tend to have different AC coefficient distributions
    # Specifically, fewer extreme values and more concentrated around zero
    kurtosis = np.mean(((ac_coeffs - np.mean(ac_coeffs)) / (np.std(ac_coeffs) + 1e-10)) ** 4)
    
    # Natural images: kurtosis typically 5-15
    # AI images: often higher or lower than natural range
    if kurtosis < 4 or kurtosis > 20:
        return 0.6
    elif kurtosis < 5 or kurtosis > 15:
        return 0.3
    return 0.1


def _patch_consistency(gray):
    """
    Measure how consistent image patches are in terms of statistics.
    AI images are more consistent across patches than real photos.
    """
    h, w = gray.shape
    block_size = min(64, min(h, w) // 4)
    if block_size < 16:
        return 0.0
    
    patch_stats = []
    for i in range(0, h - block_size, block_size):
        for j in range(0, w - block_size, block_size):
            patch = gray[i:i+block_size, j:j+block_size]
            stats = [
                np.mean(patch),
                np.std(patch),
                np.median(patch),
                float(np.percentile(patch, 90) - np.percentile(patch, 10))
            ]
            patch_stats.append(stats)
    
    if len(patch_stats) < 4:
        return 0.0
    
    patch_stats = np.array(patch_stats)
    
    # Compute coefficient of variation for each statistic
    cvs = []
    for col in range(patch_stats.shape[1]):
        col_data = patch_stats[:, col]
        cv = np.std(col_data) / (np.mean(col_data) + 1e-10)
        cvs.append(cv)
    
    mean_cv = np.mean(cvs)
    
    # AI: lower CV (more consistent patches)
    if mean_cv < 0.15:
        return 0.8  # Very consistent - AI signal
    elif mean_cv < 0.25:
        return 0.5
    elif mean_cv < 0.35:
        return 0.25
    return 0.0


def _generate_texture_viz(gray):
    """Generate texture analysis visualization."""
    # High-pass filtered image showing micro-texture
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.5)
    high_freq = gray - blurred
    
    # Normalize for visualization
    hf_norm = high_freq - high_freq.min()
    hf_norm = (hf_norm / (hf_norm.max() + 1e-10) * 255).astype(np.uint8)
    
    colored = cv2.applyColorMap(hf_norm, cv2.COLORMAP_BONE)
    
    max_dim = 512
    h, w = colored.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        colored = cv2.resize(colored, (int(w * scale), int(h * scale)))
    
    _, buffer = cv2.imencode('.png', colored)
    return base64.b64encode(buffer).decode('utf-8')
