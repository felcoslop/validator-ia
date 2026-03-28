"""
Error Level Analysis (ELA) Module
Re-compresses the image at a fixed JPEG quality and measures the difference.
Real images show uniform ELA; manipulated/synthetic regions show discrepancies.
"""

import numpy as np
import cv2
import base64
from io import BytesIO
from PIL import Image


def analyze(image_np):
    """
    Perform Error Level Analysis.
    Returns score (0-100), details, and ELA visualization.
    """
    # Convert to PIL for JPEG re-compression
    img_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)

    quality = 90
    # Re-compress at fixed quality
    buffer = BytesIO()
    pil_img.save(buffer, format='JPEG', quality=quality)
    buffer.seek(0)
    recompressed = np.array(Image.open(buffer)).astype(np.float64)

    original = img_rgb.astype(np.float64)

    # ELA = difference amplified
    ela = np.abs(original - recompressed)
    ela_gray = np.mean(ela, axis=2)

    # Amplify for visibility
    scale = 20.0
    ela_amplified = np.clip(ela * scale, 0, 255).astype(np.uint8)

    # Statistics
    mean_ela = np.mean(ela_gray)
    std_ela = np.std(ela_gray)
    max_ela = np.max(ela_gray)

    # Block-level analysis (divide into grid)
    block_size = 64
    h, w = ela_gray.shape
    block_means = []
    for i in range(0, h - block_size, block_size):
        for j in range(0, w - block_size, block_size):
            block = ela_gray[i:i + block_size, j:j + block_size]
            block_means.append(np.mean(block))

    block_means = np.array(block_means) if block_means else np.array([0])
    block_std = np.std(block_means)
    block_range = np.ptp(block_means)

    # Uniformity score
    cv_ela = std_ela / (mean_ela + 1e-10)  # Coefficient of variation

    # Score calculation — tuned for modern diffusion model detection
    score = 0

    # High variance in ELA suggests mixed content
    if cv_ela > 2.0:
        score += 25
    elif cv_ela > 1.0:
        score += 15
    elif cv_ela > 0.5:
        score += 5

    # Block-level inconsistency
    if block_std > 5.0:
        score += 20
    elif block_std > 2.0:
        score += 10

    # Very low mean ELA = AI-generated (diffusion models have extremely clean output)
    if mean_ela < 0.5:
        score += 40  # Very strong AI signal
    elif mean_ela < 1.0:
        score += 30  # Strong AI signal
    elif mean_ela < 2.0:
        score += 20  # Suspicious
    elif mean_ela < 3.0:
        score += 10  # Mildly suspicious
    elif mean_ela > 15.0:
        score += 5   # Over-compressed

    # Large range in block means
    if block_range > 20.0:
        score += 15
    elif block_range > 10.0:
        score += 8

    # AI-generated images have characteristic uniform ELA patterns
    uniform_ratio = np.sum(block_means < mean_ela * 0.3) / len(block_means)
    if uniform_ratio > 0.3:
        score += 25
    elif uniform_ratio > 0.15:
        score += 15
    elif uniform_ratio > 0.05:
        score += 8

    score = min(100, max(0, score))

    # Visualization
    viz = _generate_ela_viz(ela_amplified)

    details = {
        'mean_ela': round(float(mean_ela), 3),
        'std_ela': round(float(std_ela), 3),
        'max_ela': round(float(max_ela), 3),
        'coefficient_of_variation': round(float(cv_ela), 3),
        'block_inconsistency': round(float(block_std), 3),
        'quality_used': quality,
        'findings': []
    }

    if cv_ela > 2.0:
        details['findings'].append('Alta variação no nível de erro — regiões com compressão inconsistente indicam possível composição ou edição localizada')
    if block_std > 5.0:
        details['findings'].append('Blocos com níveis de erro muito diferentes — padrão de manipulação: áreas editadas reagem diferentemente à recompressão JPEG')
    if mean_ela < 1.0:
        details['findings'].append('ELA extremamente baixo e uniforme — modelos generativos como Stable Diffusion e DALL-E produzem imagens sem artefatos de compressão, gerando ELA próximo de zero em toda a imagem')
    elif mean_ela < 2.0:
        details['findings'].append('ELA suspeitamente baixo — consistente com imagem gerada digitalmente ou pesadamente processada, sem as variações naturais de uma foto capturada')
    elif mean_ela > 15.0:
        details['findings'].append('ELA elevado indica recompressão pesada — possível tentativa de mascarar edições')
    if uniform_ratio > 0.15:
        details['findings'].append('Regiões extensas com ELA próximo de zero — padrão diagnóstico de conteúdo gerado por IA, onde não há histórico de compressão anterior')
    if not details['findings']:
        details['findings'].append('Análise de nível de erro compatível com imagem fotográfica real — variações de compressão dentro do esperado')

    return {
        'name': 'Análise de Nível de Erro (ELA)',
        'score': score,
        'details': details,
        'visualization': viz,
        'icon': '🎯'
    }


def _generate_ela_viz(ela_amplified):
    """Generate ELA visualization."""
    ela_bgr = cv2.cvtColor(ela_amplified, cv2.COLOR_RGB2BGR)

    max_dim = 512
    h, w = ela_bgr.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        ela_bgr = cv2.resize(ela_bgr, (int(w * scale), int(h * scale)))

    _, buffer = cv2.imencode('.png', ela_bgr)
    return base64.b64encode(buffer).decode('utf-8')
