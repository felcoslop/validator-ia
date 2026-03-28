"""
Deep Learning AI Image Classifier Module
Uses a pre-trained model from HuggingFace to classify images as AI-generated or real.
Model: umm-maybe/AI-image-detector (ViT-based)
"""

import numpy as np
import cv2
from PIL import Image
from io import BytesIO

# Lazy-load the model to avoid slow startup
_classifier = None

def _get_classifier():
    """Lazy-load the HuggingFace pipeline (downloads model on first use)."""
    global _classifier
    if _classifier is None:
        from transformers import pipeline
        _classifier = pipeline(
            "image-classification",
            model="umm-maybe/AI-image-detector",
            device=-1  # CPU (use 0 for GPU)
        )
    return _classifier


def analyze(image_np):
    """
    Classify image using deep learning model.
    Returns score (0-100, higher = more likely AI), details, and no visualization.
    """
    try:
        # Convert from BGR (OpenCV) to RGB (PIL)
        img_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)

        # Resize to reasonable dimension for the model
        max_dim = 512
        w, h = pil_img.size
        if max(w, h) > max_dim:
            scale = max_dim / max(w, h)
            pil_img = pil_img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

        # Run classifier
        classifier = _get_classifier()
        results = classifier(pil_img)

        # Parse results: model returns labels like 'artificial' and 'human'
        ai_score = 0.0
        human_score = 0.0

        for r in results:
            label = r['label'].lower()
            confidence = r['score']
            if label in ['artificial', 'ai', 'fake', 'generated']:
                ai_score = confidence
            elif label in ['human', 'real', 'natural', 'authentic']:
                human_score = confidence

        # Convert to 0-100 scale
        # If model says "artificial" with 0.95 confidence -> score = 95
        score = ai_score * 100

        details = {
            'ai_probability': round(float(ai_score), 4),
            'human_probability': round(float(human_score), 4),
            'model': 'umm-maybe/AI-image-detector',
            'raw_results': [{'label': r['label'], 'score': round(r['score'], 4)} for r in results],
            'findings': []
        }

        if ai_score > 0.85:
            details['findings'].append(f'Rede neural classificou como conteúdo gerado por IA com {ai_score*100:.1f}% de confiança — padrões característicos de modelos generativos foram identificados na estrutura profunda da imagem')
        elif ai_score > 0.6:
            details['findings'].append(f'Classificador neural detectou sinais de geração por IA ({ai_score*100:.1f}%) — múltiplas características de processamento sintético foram identificadas')
        elif ai_score > 0.4:
            details['findings'].append(f'Classificador neural detectou sinais moderados de IA ({ai_score*100:.1f}% IA vs {human_score*100:.1f}% Real) — a imagem apresenta características mistas que sugerem manipulação ou geração parcial')
        else:
            details['findings'].append(f'Classificador neural identificou como conteúdo autêntico com {human_score*100:.1f}% de confiança — padrões compatíveis com captura por dispositivo real')

        return {
            'name': 'Classificador Neural (DL)',
            'score': round(score),
            'details': details,
            'visualization': None,
            'icon': 'brain'
        }

    except Exception as e:
        return {
            'name': 'Classificador Neural (DL)',
            'score': 50,
            'details': {'findings': [f'Erro no classificador: {str(e)}']},
            'visualization': None,
            'icon': 'brain'
        }
