"""
Common utilities for forensic analysis.
"""

import cv2
import numpy as np


def detect_ui_content(img):
    """
    Detect if the content is a digital UI/Screenshot vs natural image.
    UI content triggers many forensic false positives (zero noise, perfect edges).
    Returns a score from 0.0 (natural) to 1.0 (pure UI/Digital).
    """
    if img is None:
        return 0.0
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    
    # 1. Color palette uniqueness (UI has very few unique colors)
    # Resize for speed
    small = cv2.resize(img, (128, 128))
    # Count unique colors in a quantized space
    quantized = (small // 32) * 32
    if len(quantized.shape) == 3 and quantized.shape[2] == 3:
        # Color image
        unique_colors = len(np.unique(quantized.reshape(-1, 3), axis=0))
    else:
        # Grayscale image (2D or 3D with 1 channel)
        unique_colors = len(np.unique(quantized))
    # UI usually has < 30 quantized colors. Natural images have > 100.
    color_score = 1.0 - min(1.0, (unique_colors - 10) / 80) if unique_colors > 10 else 1.0
    
    # 2. Straight lines (Hough Transform) - Looking for perfect axis-alignment
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/2, 50, minLineLength=80, maxLineGap=2)
    line_score = 0
    if lines is not None:
        # UI has dozens of perfectly aligned lines
        line_score = min(1.0, len(lines) / 30)
        
    # 3. Flat areas (Solid color blocks)
    # Sample many blocks and check std dev
    blocks_h, blocks_w = h // 32, w // 32
    flat_blocks = 0
    sample_size_h = min(15, blocks_h)
    sample_size_w = min(15, blocks_w)
    total_samples = 0
    for i in range(sample_size_h):
        for j in range(sample_size_w):
            y, x = i * 32, j * 32
            block = gray[y:y+32, x:x+32]
            if np.std(block) < 0.5: # Extremely flat (digital solid color)
                flat_blocks += 1
            total_samples += 1
            
    flat_score = flat_blocks / max(1, total_samples)
    
    # 4. Corner density (UI has high density of sharp corners)
    corners = cv2.goodFeaturesToTrack(gray, 100, 0.01, 10)
    corner_score = 0
    if corners is not None:
        corner_score = min(1.0, len(corners) / 80)
    
    # Combined score - UI must be multiple things at once
    # If any texture exists (low flat_score), it's likely NOT UI
    ui_factor = (color_score * 0.3 + line_score * 0.3 + flat_score * 0.3 + corner_score * 0.1)
    
    # Apply a non-linear activation - must be very UI-like to trigger mitigation
    if ui_factor < 0.5:
        return 0.0
    
    return float(min(1.0, (ui_factor - 0.5) * 2))


def detect_low_end_sensor(img_np, metadata_dict=None):
    """
    Detect if the image likely originates from a low-end sensor (Webcam, USB Camera, etc.)
    which naturally produces high-noise, flat-frequency patterns similar to AI artifacts.
    """
    if img_np is None:
        return 0.0
        
    score = 0.0
    
    # 1. Metadata check (strongest signal)
    if metadata_dict:
        make = str(metadata_dict.get('Make', '')).lower()
        model = str(metadata_dict.get('Model', '')).lower()
        software = str(metadata_dict.get('Software', '')).lower()
        
        webcam_keywords = ['webcam', 'usb camera', 'integrated camera', 'front camera', 'ov5648', 'ov8858']
        if any(kw in make or kw in model or kw in software for kw in webcam_keywords):
            score += 0.8
            
    # 2. Resolution check (webcams are often exactly standard sensor sizes)
    h, w = img_np.shape[:2]
    # Standard Webcam/Integrated Sensor sizes
    common_webcam_res = [
        (640, 480), (1280, 720), (1280, 960), 
        (1600, 1200), (1920, 1080), (2048, 1536)
    ]
    if (w, h) in common_webcam_res or (h, w) in common_webcam_res:
        score += 0.4 # Sufficient to trigger calibration when paired with low-MP fallback
    
    # Low megapixels check (Webcams/Front cameras rarely exceed 3-5MP)
    # If the image is < 3MP and has no professional EXIF, it's highly likely a low-end sensor
    mp = (w * h) / 1_000_000
    if mp < 3.0:
        score += 0.2
        
    return min(1.0, score)
