"""
Gradient Analysis Module
Analyzes gradient transitions between skin, texture, and background regions.
AI-generated images often show abnormal transitions (halos, over-smoothing).
"""

import numpy as np
import cv2
import base64
from . import utils


def analyze(image_np):
    """
    Perform gradient analysis to detect unnatural transitions.
    Returns score (0-100), details, and visualization.
    """
    gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY).astype(np.float64)
    h, w = gray.shape

    # 1. Compute Sobel gradients
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
    direction = np.arctan2(grad_y, grad_x)

    # 2. Local entropy analysis
    entropy_map = _local_entropy(gray, block_size=16)

    # 3. Gradient magnitude statistics
    mean_grad = np.mean(magnitude)
    std_grad = np.std(magnitude)
    grad_cv = std_grad / (mean_grad + 1e-10)

    # 4. Edge coherence - analyze gradient direction consistency
    edge_coherence = _edge_coherence(direction, magnitude)

    # 5. Transition analysis (entropy gradient at edges)
    transition_score = _analyze_transitions(entropy_map, magnitude)

    # 6. Halo detection (abnormal gradient patterns around edges)
    halo_score = _detect_halos(magnitude, gray)

    # 7. Texture regularity (AI tends to over-regularize)
    regularity = _texture_regularity(magnitude)

    # Score — tuned for modern diffusion model detection
    score = 0

    # Gradient coefficient of variation
    # AI over-smooths: grad_cv is low. Real photos have natural texture variation.
    if grad_cv < 0.3:
        score += 25  # Very over-smoothed (strong AI signal)
    elif grad_cv < 0.5:
        score += 18
    elif grad_cv < 0.8:
        score += 8
    elif grad_cv > 5.0:
        score += 5

    # Edge coherence (AI often has overly coherent edges)
    if edge_coherence > 0.6:
        score += 25
    elif edge_coherence > 0.4:
        score += 15
    elif edge_coherence > 0.25:
        score += 8

    # Transition abnormality
    score += min(20, int(transition_score * 25))

    # Halo artifacts
    score += min(15, int(halo_score * 20))

    # Texture over-regularity (very strong AI indicator)
    if regularity > 0.7:
        score += 25
    elif regularity > 0.5:
        score += 18
    elif regularity > 0.3:
        score += 10
    else:
        score += min(8, int(regularity * 20))

    score = min(100, max(0, score))

    # Visualization
    viz = _generate_gradient_viz(magnitude)

    details = {
        'mean_gradient': round(float(mean_grad), 3),
        'gradient_cv': round(float(grad_cv), 3),
        'edge_coherence': round(float(edge_coherence), 4),
        'transition_score': round(float(transition_score), 4),
        'halo_score': round(float(halo_score), 4),
        'texture_regularity': round(float(regularity), 4),
        'findings': []
    }

    if grad_cv < 0.5:
        details['findings'].append('Gradientes excessivamente suaves — modelos de difusão suavizam transições entre regiões de forma artificial, eliminando as variações ópticas naturais de uma lente real (aberração cromática, difração)')
    if edge_coherence > 0.4:
        details['findings'].append('Coerência de bordas excessiva — em imagens reais, bordas têm direções variadas; IAs generativas produzem bordas artificialmente alinhadas, característica de convoluções neurais')
    if transition_score > 0.5:
        details['findings'].append('Transições abruptas de entropia entre regiões — incompatível com a Point Spread Function (PSF) de lentes reais; indica processamento por rede neural')
    if halo_score > 0.5:
        details['findings'].append('Halos detectados ao redor de bordas — artefato de sombra sintética produzido por redes neurais que não simulam corretamente a dispersão de luz natural')
    if regularity > 0.5:
        details['findings'].append('Textura excessivamente regular sem micro-variações naturais — IAs generativas produzem superfícies "perfeitas demais", sem grãos, poros ou imperfeições que existem em toda foto real')
    if not details['findings']:
        details['findings'].append('Padrões de gradiente e transições compatíveis com captura óptica real — sem suavização artificial detectada')

    # UI/Screenshot Detection Mitigation
    ui_factor = utils.detect_ui_content(image_np)
    
    if ui_factor > 0.6:
        mitigation = int(score * 0.4 * ui_factor)
        score -= mitigation
        details['findings'].append(f'Interface Digital detectada ({int(ui_factor*100)}%) - gradientes artificiais mitigados')

    return {
        'name': 'Análise de Gradiente',
        'score': score,
        'details': details,
        'visualization': viz,
        'icon': '🌊'
    }


def _local_entropy(gray, block_size=16):
    """Compute local entropy map."""
    h, w = gray.shape
    entropy_map = np.zeros((h // block_size, w // block_size))

    for i in range(entropy_map.shape[0]):
        for j in range(entropy_map.shape[1]):
            block = gray[i * block_size:(i + 1) * block_size,
                         j * block_size:(j + 1) * block_size]
            hist, _ = np.histogram(block, bins=64, range=(0, 255), density=True)
            hist = hist[hist > 0]
            entropy_map[i, j] = -np.sum(hist * np.log2(hist + 1e-10))

    return entropy_map


def _edge_coherence(direction, magnitude):
    """Measure edge direction coherence (AI images tend to be overly coherent)."""
    # Focus on strong edges
    threshold = np.percentile(magnitude, 80)
    strong_edges = magnitude > threshold

    if strong_edges.sum() < 100:
        return 0.5

    dirs = direction[strong_edges]

    # Histogram of directions
    hist, _ = np.histogram(dirs, bins=36, range=(-np.pi, np.pi))
    hist = hist / (hist.sum() + 1e-10)

    # Compute entropy of direction distribution
    hist_nonzero = hist[hist > 0]
    entropy = -np.sum(hist_nonzero * np.log2(hist_nonzero))
    max_entropy = np.log2(36)

    return float(1.0 - entropy / max_entropy)


def _analyze_transitions(entropy_map, magnitude):
    """Analyze entropy transitions at gradient boundaries."""
    if entropy_map.size < 4:
        return 0.0

    # Compute gradient of entropy map
    if entropy_map.shape[0] < 3 or entropy_map.shape[1] < 3:
        return 0.0

    ent_grad_x = np.diff(entropy_map, axis=1)
    ent_grad_y = np.diff(entropy_map, axis=0)

    # Sharp entropy transitions = potential AI artifact
    sharp_transitions = np.sum(np.abs(ent_grad_x) > 2.0) + np.sum(np.abs(ent_grad_y) > 2.0)
    total = ent_grad_x.size + ent_grad_y.size

    return float(min(1.0, sharp_transitions / (total * 0.1 + 1e-10)))


def _detect_halos(magnitude, gray):
    """Detect halo artifacts around edges."""
    # Dilate magnitude to find edge neighborhoods
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    edges = (magnitude > np.percentile(magnitude, 85)).astype(np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=2)
    halo_zone = dilated - edges

    if halo_zone.sum() < 100:
        return 0.0

    # In halo zone, check for unusual intensity patterns
    halo_vals = gray[halo_zone > 0]
    non_halo_vals = gray[halo_zone == 0]

    if len(halo_vals) < 10 or len(non_halo_vals) < 10:
        return 0.0

    # Halo detection: check if edge neighborhood has abnormal intensity distribution
    halo_std = np.std(halo_vals)
    non_halo_std = np.std(non_halo_vals)

    ratio = halo_std / (non_halo_std + 1e-10)
    return float(max(0, min(1.0, abs(1.0 - ratio) * 2)))


def _texture_regularity(magnitude):
    """Detect over-regular textures (AI artifact)."""
    h, w = magnitude.shape
    block_size = 32

    if h < block_size * 4 or w < block_size * 4:
        return 0.0

    block_stds = []
    for i in range(0, h - block_size, block_size):
        for j in range(0, w - block_size, block_size):
            block = magnitude[i:i + block_size, j:j + block_size]
            block_stds.append(np.std(block))

    block_stds = np.array(block_stds)
    # Over-regular: many blocks with similar texture statistics
    cv = np.std(block_stds) / (np.mean(block_stds) + 1e-10)

    return float(max(0, min(1.0, 1.0 - cv)))


def _generate_gradient_viz(magnitude):
    """Generate gradient magnitude visualization."""
    mag_norm = magnitude / (magnitude.max() + 1e-10) * 255
    mag_uint8 = mag_norm.astype(np.uint8)
    colored = cv2.applyColorMap(mag_uint8, cv2.COLORMAP_VIRIDIS)

    max_dim = 512
    h, w = colored.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        colored = cv2.resize(colored, (int(w * scale), int(h * scale)))

    _, buffer = cv2.imencode('.png', colored)
    return base64.b64encode(buffer).decode('utf-8')
