"""
Video Analysis Module
Extracts frames and performs temporal consistency analysis.
AI-generated videos show flickering, temporal inconsistencies, and unstable noise patterns.
"""

import numpy as np
import cv2
import os
import tempfile


def analyze(video_path, frame_analyzers):
    """
    Analyze video for temporal consistency and per-frame forensics.
    frame_analyzers: dict of analyzer modules to apply per-frame.
    Returns score (0-100), details, and per-frame results.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {
            'name': 'Análise de Vídeo',
            'score': 50,
            'details': {'findings': ['Não foi possível abrir o vídeo']},
            'visualization': None,
            'icon': '🎬'
        }

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Sample frames (max 10 evenly spaced)
    n_samples = min(10, total_frames)
    if n_samples < 2:
        cap.release()
        return {
            'name': 'Análise de Vídeo',
            'score': 50,
            'details': {'findings': ['Vídeo muito curto para análise temporal']},
            'visualization': None,
            'icon': '🎬'
        }

    frame_indices = np.linspace(0, total_frames - 1, n_samples, dtype=int)
    frames = []

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)

    cap.release()

    if len(frames) < 2:
        return {
            'name': 'Análise de Vídeo',
            'score': 50,
            'details': {'findings': ['Poucos frames válidos para análise']},
            'visualization': None,
            'icon': '🎬'
        }

    # 1. Temporal noise consistency
    noise_consistency = _temporal_noise_consistency(frames)

    # 2. Inter-frame spectral stability
    spectral_stability = _spectral_stability(frames)

    # 3. Optical flow smoothness
    flow_score = _optical_flow_analysis(frames)

    # 4. Per-frame analysis (lightweight - only frequency and statistical)
    from analyzers import frequency, statistical

    frame_scores = []
    for frame in frames[:5]:  # Analyze up to 5 frames
        freq_result = frequency.analyze(frame)
        stat_result = statistical.analyze(frame)
        avg_score = (freq_result['score'] + stat_result['score']) / 2
        frame_scores.append(avg_score)

    mean_frame_score = np.mean(frame_scores)
    frame_score_var = np.var(frame_scores)

    # 5. Flickering detection (brightness/color variation between consecutive frames)
    flicker_score = _detect_flickering(frames)

    # Score
    score = 0

    # Temporal noise inconsistency
    if noise_consistency > 0.5:
        score += 25
    elif noise_consistency > 0.2:
        score += 15
    elif noise_consistency > 0.1:
        score += 8

    # Spectral instability
    if spectral_stability > 0.5:
        score += 25
    elif spectral_stability > 0.2:
        score += 15
    elif spectral_stability > 0.1:
        score += 8

    # Optical flow anomaly
    score += min(15, int(flow_score * 15))

    # Per-frame forensics (boosted: individual AI frames inherently score higher)
    score += min(30, int(mean_frame_score * 0.45))

    # Frame score variance (inconsistent = suspicious)
    if frame_score_var > 100:
        score += 10

    # Flickering
    score += min(15, int(flicker_score * 15))

    score = min(100, max(0, score))

    details = {
        'total_frames': total_frames,
        'fps': round(float(fps), 1),
        'resolution': f'{width}x{height}',
        'frames_analyzed': len(frames),
        'noise_consistency': round(float(noise_consistency), 4),
        'spectral_stability': round(float(spectral_stability), 4),
        'flow_anomaly': round(float(flow_score), 4),
        'mean_frame_score': round(float(mean_frame_score), 1),
        'frame_score_variance': round(float(frame_score_var), 1),
        'flicker_score': round(float(flicker_score), 4),
        'findings': []
    }

    if noise_consistency > 0.5:
        details['findings'].append('Inconsistência temporal significativa no padrão de ruído entre frames')
    if spectral_stability > 0.5:
        details['findings'].append('Instabilidade espectral entre frames - possível artefato de geração por IA')
    if flow_score > 0.5:
        details['findings'].append('Anomalias no fluxo óptico - movimentos geometricamente inconsistentes')
    if flicker_score > 0.5:
        details['findings'].append('Flickering detectado - variação temporal não-natural')
    if mean_frame_score > 50:
        details['findings'].append(f'Análise forense dos frames individuais sugere conteúdo sintético (score médio: {mean_frame_score:.0f})')
    if not details['findings']:
        details['findings'].append('Consistência temporal e análise de frames dentro dos parâmetros normais')

    return {
        'name': 'Análise de Vídeo',
        'score': score,
        'details': details,
        'visualization': None,
        'icon': '🎬'
    }


def _temporal_noise_consistency(frames):
    """Check if noise patterns are consistent across frames (real cameras have consistent PRNU)."""
    from skimage.restoration import denoise_wavelet

    noise_patterns = []
    for frame in frames[:5]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float64) / 255.0
        # Downsample for speed
        if gray.shape[0] > 256:
            scale = 256 / gray.shape[0]
            gray = cv2.resize(gray, (int(gray.shape[1] * scale), 256))

        denoised = denoise_wavelet(gray, method='BayesShrink', mode='soft', wavelet='db4', rescale_sigma=True)
        noise = gray - denoised
        noise_patterns.append(noise)

    # Compare noise patterns between frames
    if len(noise_patterns) < 2:
        return 0.0

    correlations = []
    for i in range(len(noise_patterns)):
        for j in range(i + 1, len(noise_patterns)):
            n1 = noise_patterns[i]
            n2 = noise_patterns[j]
            # Ensure same size
            min_h = min(n1.shape[0], n2.shape[0])
            min_w = min(n1.shape[1], n2.shape[1])
            n1 = n1[:min_h, :min_w].ravel()
            n2 = n2[:min_h, :min_w].ravel()

            if np.std(n1) > 1e-10 and np.std(n2) > 1e-10:
                corr = np.corrcoef(n1, n2)[0, 1]
                correlations.append(abs(corr))

    if not correlations:
        return 0.0

    mean_corr = np.mean(correlations)
    # Real cameras: noise patterns should be somewhat correlated (PRNU)
    # AI: noise patterns tend to be uncorrelated or overly identical
    if mean_corr > 0.8:
        return 0.3  # Suspiciously high (copy-paste noise)
    elif mean_corr < 0.05:
        return 0.7  # No noise pattern consistency (likely synthetic)

    return float(max(0, 0.3 - mean_corr))


def _spectral_stability(frames):
    """Check spectral profile stability across frames."""
    profiles = []
    for frame in frames[:5]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float64)
        if gray.shape[0] > 256:
            scale = 256 / gray.shape[0]
            gray = cv2.resize(gray, (int(gray.shape[1] * scale), 256))

        f = np.fft.fft2(gray)
        mag = np.abs(np.fft.fftshift(f))

        # Radial profile
        cy, cx = mag.shape[0] // 2, mag.shape[1] // 2
        max_r = min(cy, cx)
        y, x = np.ogrid[:mag.shape[0], :mag.shape[1]]
        r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2).astype(int)
        r = np.clip(r, 0, max_r - 1)

        profile = np.zeros(max_r)
        counts = np.zeros(max_r)
        np.add.at(profile, r.ravel(), mag.ravel())
        np.add.at(counts, r.ravel(), 1)
        counts[counts == 0] = 1
        profile = profile / counts
        profile = profile / (np.max(profile) + 1e-10)
        profiles.append(profile)

    if len(profiles) < 2:
        return 0.0

    # Compare profiles
    min_len = min(len(p) for p in profiles)
    profiles = [p[:min_len] for p in profiles]

    diffs = []
    for i in range(len(profiles) - 1):
        diff = np.mean(np.abs(profiles[i] - profiles[i + 1]))
        diffs.append(diff)

    return float(min(1.0, np.mean(diffs) * 5))


def _optical_flow_analysis(frames):
    """Basic optical flow consistency check."""
    if len(frames) < 2:
        return 0.0

    flow_magnitudes = []
    for i in range(min(len(frames) - 1, 4)):
        gray1 = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frames[i + 1], cv2.COLOR_BGR2GRAY)

        # Downsample
        scale = min(1.0, 256 / max(gray1.shape))
        if scale < 1.0:
            gray1 = cv2.resize(gray1, None, fx=scale, fy=scale)
            gray2 = cv2.resize(gray2, None, fx=scale, fy=scale)

        flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
        flow_magnitudes.append(np.mean(magnitude))

    if not flow_magnitudes:
        return 0.0

    # Check for sudden jumps in optical flow
    flow_var = np.var(flow_magnitudes)
    flow_mean = np.mean(flow_magnitudes)

    if flow_mean < 0.1:
        return 0.3  # Very little motion (could be still AI content)

    cv_flow = flow_var / (flow_mean ** 2 + 1e-10)
    return float(min(1.0, cv_flow))


def _detect_flickering(frames):
    """Detect brightness/color flickering between frames."""
    if len(frames) < 2:
        return 0.0

    brightness = []
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness.append(np.mean(gray))

    # Compute sequential differences
    diffs = np.abs(np.diff(brightness))
    mean_brightness = np.mean(brightness)

    if mean_brightness < 1:
        return 0.0

    # Normalize by mean brightness
    norm_diffs = diffs / mean_brightness
    flicker = np.std(norm_diffs)

    return float(min(1.0, flicker * 10))
