"""
ForensicAI — Analysis Engine
Orchestrates modules with optimized video processing and frame capture for PDF reports.
"""

import os
import time
import uuid
import base64
import numpy as np
import cv2
from PIL import Image
from io import BytesIO

from analyzers import frequency, noise, ela, gradient, statistical, wavelet, metadata, hv_score, video, texture, dl_classifier


def imread_unicode(path):
    """Read image from path with potential unicode characters on Windows."""
    try:
        # Read as bytes first
        data = np.fromfile(path, dtype=np.uint8)
        # Decode using OpenCV
        img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
        return img
    except Exception:
        return None


WEIGHTS = {
    'dl_classifier': 0.50,
    'ai_texture': 0.10,
    'frequency': 0.06,
    'noise': 0.06,
    'ela': 0.05,
    'gradient': 0.05,
    'statistical': 0.05,
    'wavelet': 0.05,
    'metadata': 0.04,
    'hv_score': 0.04,
}

# Max dimension for analysis (optimization: reduce resolution)
ANALYSIS_MAX_DIM = 1024
# Max dimension for thumbnails saved in report
THUMB_MAX_DIM = 640


def _resize_for_analysis(img, max_dim=ANALYSIS_MAX_DIM):
    """Resize image for faster analysis without losing forensic quality."""
    h, w = img.shape[:2]
    if max(h, w) <= max_dim:
        return img
    scale = max_dim / max(h, w)
    return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LANCZOS4)


def _img_to_base64(img, max_dim=THUMB_MAX_DIM):
    """Convert cv2 image to base64 PNG for embedding in results."""
    h, w = img.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    _, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf).decode('utf-8')


def analyze_image(file_path, eval_id):
    """Run all analysis modules on an image."""
    start_time = time.time()

    image_np = imread_unicode(file_path)
    if image_np is None:
        return {'error': f'Não foi possível carregar a imagem em: {file_path}'}

    # Coerce to color if grayscale to prevent module crashes
    if len(image_np.shape) == 2:
        image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2BGR)

    # Save original thumbnail for report
    original_thumb = _img_to_base64(image_np)

    # Resize for analysis
    image_np = _resize_for_analysis(image_np)

    results = []
    analyzers = [
        ('dl_classifier', dl_classifier),
        ('frequency', frequency),
        ('noise', noise),
        ('ela', ela),
        ('gradient', gradient),
        ('statistical', statistical),
        ('wavelet', wavelet),
        ('hv_score', hv_score),
        ('texture', texture),
    ]

    for name, mod in analyzers:
        try:
            res = mod.analyze(image_np)
            res['name'] = name  # Sync with i18n keys
            results.append(res)
        except Exception as e:
            results.append(_error_result(name, str(e)))

    # Metadata uses file path
    try:
        meta_res = metadata.analyze(file_path)
        meta_res['name'] = 'metadata'
        results.append(meta_res)
    except Exception as e:
        results.append(_error_result('metadata', str(e)))

    final_score = _compute_final_score(results, image_np.shape[1], image_np.shape[0])
    verdict = _generate_verdict(final_score, results)

    return {
        'type': 'image',
        'filename': os.path.basename(file_path),
        'dimensions': f'{image_np.shape[1]}x{image_np.shape[0]}',
        'original_thumbnail': original_thumb,
        'final_score': round(final_score, 1),
        'verdict': verdict,
        'modules': results,
        'elapsed_seconds': round(time.time() - start_time, 2),
        'frame_thumbnails': [],
    }


def analyze_video(file_path, eval_id, selected_indices=None):
    """
    Video analysis with optional manual frame selection.
    If selected_indices is provided, analyzes only those frames.
    Otherwise, uses smart auto-sampling.
    """
    start_time = time.time()

    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        return {'error': 'Não foi possível carregar o vídeo'}

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps

    # --- PHASE 1: Quick scan or Manual selection ---
    if selected_indices:
        phase1_indices = [int(i) for i in selected_indices]
        phase1_frames = _extract_frames(cap, phase1_indices)
        auto_sampling = False
    else:
        phase1_indices = _get_key_frame_indices(total_frames, count=5)
        phase1_frames = _extract_frames(cap, phase1_indices)
        auto_sampling = True

    # Analyze phase 1 frames
    frame_results = []
    frame_thumbnails = []
    early_ai_detected = False

    for i, (frame, idx) in enumerate(zip(phase1_frames, phase1_indices)):
        if frame is None:
            continue
        # Save thumbnail for PDF
        thumb = _img_to_base64(frame)
        timestamp = idx / fps
        frame_thumbnails.append({
            'index': int(idx),
            'timestamp': f'{timestamp:.1f}s',
            'thumbnail': thumb,
            'phase': 'Manual' if selected_indices else 'Fase 1 (Rápida)',
        })

        # Resize and analyze
        frame_resized = _resize_for_analysis(frame, max_dim=768)
        frame_score = _quick_frame_analysis(frame_resized)
        frame_results.append(frame_score)

        # Early exit only if auto-sampling
        if auto_sampling and i >= 2 and all(s > 65 for s in frame_results[:3]):
            early_ai_detected = True
            break

    # --- PHASE 2: Only if auto-sampling and not clearly IA ---
    if auto_sampling and not early_ai_detected and total_frames > 30:
        phase2_indices = _get_extended_frame_indices(total_frames, phase1_indices, count=5)
        phase2_frames = _extract_frames(cap, phase2_indices)

        for frame, idx in zip(phase2_frames, phase2_indices):
            if frame is None:
                continue
            thumb = _img_to_base64(frame)
            timestamp = idx / fps
            frame_thumbnails.append({
                'index': int(idx),
                'timestamp': f'{timestamp:.1f}s',
                'thumbnail': thumb,
                'phase': 'Fase 2 (Detalhada)',
            })

            frame_resized = _resize_for_analysis(frame, max_dim=768)
            frame_score = _quick_frame_analysis(frame_resized)
            frame_results.append(frame_score)

    cap.release()

    # Run full analysis on representative frames
    # DL classifier runs on 3 frames (beginning, middle, end) for better coverage
    # Heuristic modules run on middle frame for efficiency
    full_module_results = []
    
    if phase1_frames:
        # Pick 3 diverse frames: first, middle, last
        frame_picks = []
        if len(phase1_frames) >= 3:
            frame_picks = [phase1_frames[0], phase1_frames[len(phase1_frames)//2], phase1_frames[-1]]
        elif len(phase1_frames) == 2:
            frame_picks = [phase1_frames[0], phase1_frames[1]]
        else:
            frame_picks = [phase1_frames[0]]
        
        # Run DL classifier on multiple frames, pick the HIGHEST AI score
        dl_results = []
        for fp in frame_picks:
            if fp is not None:
                try:
                    resized = _resize_for_analysis(fp)
                    dl_result = dl_classifier.analyze(resized)
                    dl_results.append(dl_result)
                except Exception as e:
                    dl_results.append(_error_result('dl_classifier', str(e)))
        
        if dl_results:
            # Use the frame with the highest AI score (most suspicious)
            best_dl = max(dl_results, key=lambda r: r.get('score', 0))
            best_dl['name'] = 'dl_classifier' # Ensure key
            full_module_results.append(best_dl)
        
        # Heuristic modules on middle frame
        middle_frame = phase1_frames[len(phase1_frames) // 2]
        if middle_frame is not None:
            mid_resized = _resize_for_analysis(middle_frame)
            heuristic_modules = [
                ('frequency', frequency),
                ('noise', noise),
                ('ela', ela),
                ('gradient', gradient),
                ('statistical', statistical),
                ('wavelet', wavelet),
                ('hv_score', hv_score),
                ('texture', texture),
            ]
            for name, mod in heuristic_modules:
                try:
                    res = mod.analyze(mid_resized)
                    res['name'] = name  # Force key
                    full_module_results.append(res)
                except Exception as e:
                    full_module_results.append(_error_result(name, str(e)))


    # Video temporal analysis
    temporal_score = _temporal_analysis(frame_results, early_ai_detected)
    full_module_results.append({
        'name': 'temporal',
        'score': temporal_score,
        'icon': 'video',
        'details': {
            'frames_analyzed': len(frame_results),
            'early_ai_detected': early_ai_detected,
            'mean_frame_score': round(float(np.mean(frame_results)) if frame_results else 50, 1),
            'frame_score_variance': round(float(np.var(frame_results)) if frame_results else 0, 1),
            'findings': _temporal_findings(frame_results, early_ai_detected),
        },
        'visualization': None,
    })

    final_score = _compute_final_score_video(full_module_results, width, height)
    verdict = _generate_verdict(final_score, full_module_results)

    return {
        'type': 'video',
        'filename': os.path.basename(file_path),
        'dimensions': f'{width}x{height}',
        'duration': f'{duration:.1f}s',
        'fps': round(fps, 1),
        'total_frames': total_frames,
        'final_score': round(final_score, 1),
        'verdict': verdict,
        'modules': full_module_results,
        'elapsed_seconds': round(time.time() - start_time, 2),
        'frame_thumbnails': frame_thumbnails,
        'original_thumbnail': frame_thumbnails[0]['thumbnail'] if frame_thumbnails else '',
    }


def extract_video_storyboard(file_path, count=24):
    """
    Extract a grid of frames for user selection.
    Distributes frames across 3 segments (beginning, middle, end) to ensure
    temporal diversity. Within each segment, picks best content frames.
    """
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    
    if total_frames < 3:
        cap.release()
        return None
    
    # Split video into 3 segments: beginning, middle, end
    segment_size = total_frames // 3
    segments = [
        (0, segment_size),                              # Beginning
        (segment_size, segment_size * 2),               # Middle
        (segment_size * 2, total_frames),               # End
    ]
    
    # Allocate frames per segment (equal distribution)
    frames_per_segment = max(count // 3, 1)
    # Give remaining frames to middle segment
    extra = count - (frames_per_segment * 3)
    seg_counts = [frames_per_segment, frames_per_segment + max(extra, 0), frames_per_segment]
    
    # Sample candidates from each segment
    all_selected = []
    for seg_idx, (start, end) in enumerate(segments):
        seg_len = end - start
        if seg_len <= 0:
            continue
        # Sample pool within this segment
        pool_size = min(max(seg_counts[seg_idx] * 4, 15), seg_len)
        pool_indices = np.linspace(start, end - 1, pool_size, dtype=int)
        
        candidates = []
        for idx in pool_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                score = _score_frame_content(frame)
                candidates.append({
                    'index': int(idx),
                    'score': score,
                    'frame': frame,
                    'segment': seg_idx  # 0=beginning, 1=middle, 2=end
                })
        
        # Sort by content score within this segment and pick best
        candidates.sort(key=lambda x: x['score'], reverse=True)
        selected = candidates[:seg_counts[seg_idx]]
        all_selected.extend(selected)
    
    cap.release()
    
    # Sort by frame index for chronological order
    all_selected.sort(key=lambda x: x['index'])
    
    storyboard = []
    for c in all_selected:
        f = c['frame']
        h, w = f.shape[:2]
        scale = 320 / max(h, w)
        small = cv2.resize(f, (int(w * scale), int(h * scale)))
        _, buf = cv2.imencode('.jpg', small, [cv2.IMWRITE_JPEG_QUALITY, 80])
        b64 = base64.b64encode(buf).decode('utf-8')
        seg_labels = ['inicio', 'meio', 'fim']
        storyboard.append({
            'index': c['index'],
            'timestamp': f'{c["index"]/fps:.1f}s',
            'thumbnail': b64,
            'content_score': round(c['score'], 1),
            'segment': seg_labels[c['segment']]
        })
            
    return storyboard



def _score_frame_content(frame):
    """
    Heuristic to score 'interesting-ness' of a frame.
    Higher score = more detail/objects/faces.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 1. Laplacian Variance (focus/detail check)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # 2. Edge Density (object presence)
    edges = cv2.Canny(gray, 100, 200)
    edge_density = np.sum(edges > 0) / (edges.size + 1e-10)
    
    # 3. Color Variance (avoid monochromatic frames)
    small = cv2.resize(frame, (100, 100))
    color_std = np.std(small, axis=(0, 1))
    mean_color_std = np.mean(color_std)
    
    # 4. Face Detection (YOLO-like object prioritization)
    # Using built-in Haar Cascades for speed/portability
    face_boost = 0
    try:
        # Using a very small scale and minNeighbors for speed in storyboard generation
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        face_cascade = cv2.CascadeClassifier(cascade_path)
        faces = face_cascade.detectMultiScale(cv2.resize(gray, (320, 240)), 1.3, 5)
        if len(faces) > 0:
            face_boost = 500  # Large boost if faces are present
    except Exception:
        pass
    
    # Combined score
    score = (lap_var * 0.3) + (edge_density * 50000 * 0.3) + (mean_color_std * 0.1) + face_boost
    return float(score)


def _get_key_frame_indices(total_frames, count=5):
    """Get key frame indices: start, 25%, middle, 75%, end."""
    if total_frames <= count:
        return list(range(total_frames))
    positions = np.linspace(0, total_frames - 1, count, dtype=int)
    return positions.tolist()


def _get_extended_frame_indices(total_frames, existing, count=5):
    """Get additional frames between existing indices."""
    all_positions = np.linspace(0, total_frames - 1, count + len(existing) + 2, dtype=int)
    existing_set = set(existing)
    new = [p for p in all_positions if p not in existing_set]
    return new[:count]


def _extract_frames(cap, indices):
    """Extract specific frames from video."""
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        frames.append(frame if ret else None)
    return frames


def _quick_frame_analysis(frame):
    """Quick per-frame score using lightweight analyzers."""
    score_sum = 0
    count = 0
    try:
        r = frequency.analyze(frame)
        score_sum += r['score']
        count += 1
    except Exception:
        pass
    try:
        r = statistical.analyze(frame)
        score_sum += r['score']
        count += 1
    except Exception:
        pass
    return score_sum / max(count, 1)


def _temporal_analysis(frame_scores, early_detected):
    """Compute temporal consistency score."""
    if not frame_scores:
        return 50

    mean_score = np.mean(frame_scores)
    score_var = np.var(frame_scores)

    score = mean_score

    # High variance between frames is suspicious for AI
    # High variance between frames is often just video cuts/motion
    if score_var > 400 and mean_score > 50:
        score += 8
    # Very uniform scores across frames suggest batch generation
    elif score_var < 3 and len(frame_scores) > 5 and mean_score > 40:
        score += 5

    if early_detected:
        score += 10

    return min(100, max(0, score))


def _temporal_findings(frame_scores, early_detected):
    """Generate findings for temporal video analysis."""
    findings = []
    if not frame_scores:
        return ['Não foi possível analisar frames do vídeo']

    mean = np.mean(frame_scores)
    var = np.var(frame_scores)

    if early_detected:
        findings.append({'key': 'finding_temporal_early_ai'})
    if mean > 60:
        findings.append({'key': 'finding_temporal_high_mean', 'score': round(mean, 0)})
    if var > 200:
        findings.append({'key': 'finding_temporal_high_var'})
    if var < 5 and len(frame_scores) > 3:
        findings.append({'key': 'finding_temporal_low_var'})
    if not findings:
        findings.append({'key': 'finding_temporal_natural'})

    return findings


def _compute_final_score(results, width=None, height=None):
    """Weighted average of module scores with DL-boosted consensus and resolution awareness."""
    
    # Low-res compression compensation (social media filter)
    is_low_res = False
    if width and height and min(width, height) < 600:
        is_low_res = True
        
    total_weight = 0
    weighted_sum = 0
    high_signals = 0
    dl_score = None
    
    for r in results:
        name = r.get('name', '')
        score = r.get('score', 50)
        
        # Heuristics that are highly unreliable on low-res compressed media
        if is_low_res and name in ['Análise de Frequência (FFT)', 'Análise de Ruído (PRNU)', 
                                 'Score Horizontal/Vertical', 'Análise Wavelet (DWT)']:
            score *= 0.6  # 40% discount on suspiciousness due to compression noise
            r['score'] = score # Update for report
            
        w = WEIGHTS.get(name, 0.05)
        weighted_sum += score * w
        total_weight += w
        
        if name in ['dl_classifier', 'Classificador Neural (DL)']:
            dl_score = score
        elif score > 40:
            high_signals += 1

    # NEW: Smart EXIF physics mitigation
    # Heavily compressed JPEGs mimic AI by averaging out high-frequencies and natural noise.
    # If the image was exported directly from a camera hardware (valid EXIF), we mitigate
    # these physics artifacts to prevent false positive AI accusations. Screenshots and PNGs 
    # receive full unbridled AI strictness.
    import re
    meta_res = next((r for r in results if r.get('name') in ['metadata', 'Análise de Metadados (EXIF)']), None)
    safe_exif = (meta_res is not None and meta_res.get('score', 100) < 15)
    
    is_social_or_screenshot = False
    # To reliably catch screenshots passed into _analyze_image, engine's pipeline passes full file path
    # But _compute_final_score receives results. Wait, this function doesn't receive file_path!
    # Let's search inside the metadata results for filename clues if needed.
    # Actually, metadata analyzer saves filename in some cases.
    # But usually, if EXIF is safe (score < 15), it implies software like Photoshop/Camera generated it.
    # Screenshots and Whatsapp strip EXIF, making safe_exif False inherently!
    if safe_exif:
        for r in results:
            if r.get('name') not in ['dl_classifier', 'metadata', 'Classificador Neural (DL)', 'Análise de Metadados (EXIF)']:
                if r.get('score', 0) >= 30:
                    r['score'] = min(int(r['score'] * 0.45), 28)
                    # Append the mitigation flag instead of overwriting the native mathematical findings
                    r['details']['findings'].append({'key': 'finding_compression_mitigated'})

    # Re-calculate legacy average upfront for dynamic weighting.
    # We take the top 3 strongest AI signals as the representative physics score.
    legacy_modules = [r['score'] for r in results if r.get('name') not in ['dl_classifier', 'metadata', 'Classificador Neural (DL)', 'Análise de Metadados (EXIF)']]
    if len(legacy_modules) >= 3:
        top_modules = sorted(legacy_modules, reverse=True)
        legacy_avg = sum(top_modules[:3]) / 3
    else:
        legacy_avg = sum(legacy_modules) / max(1, len(legacy_modules))

    # Dynamic Weighting: 
    # If the mathematical physics modules find undeniable structural evidence of AI 
    # (legacy_avg > 50), they dynamically override an unsure DL classifier.
    # HOWEVER, a raw OS screenshot of a real photo will functionally mimic AI physics
    # (zero variance, grid alignment). If the DL text-model is ULTRA confident (< 15) 
    # that the image content is a real human/scene, we trust it over the OS math artifacts.
    if dl_score is not None:
        if legacy_avg >= 50 and 15 <= dl_score < 50:
            dl_weight = 0.15
            legacy_weight = 0.85
        elif dl_score < 15:
            dl_weight = 0.85
            legacy_weight = 0.15
        else:
            dl_weight = 0.45
            legacy_weight = 0.55
            
        raw_final = (dl_score * dl_weight) + (legacy_avg * legacy_weight)
    else:
        raw_final = legacy_avg

    final_score = raw_final
        
    return min(100, max(0, final_score))


def _compute_final_score_video(results, width=None, height=None):
    """Weighted score for video analysis with DL-boosted consensus and compression awareness."""
    
    # Low-res compression compensation (social media filter)
    is_low_res = False
    if width and height and min(width, height) < 600:
        is_low_res = True
        
    video_weights = {
        'dl_classifier': 0.35,
        'ai_texture': 0.08,
        'frequency': 0.05,
        'noise': 0.05,
        'ela': 0.05,
        'gradient': 0.04,
        'statistical': 0.04,
        'wavelet': 0.05,
        'hv_score': 0.04,
        'temporal': 0.25,
    }
    total_weight = 0
    weighted_sum = 0
    high_signals = 0
    dl_score = None
    
    for r in results:
        name = r.get('name', '')
        score = r.get('score', 50)

        # Lighten compression-prone heuristics on low-res video instead of cutting them by 50%.
        if is_low_res and name in ['frequency', 'noise', 'hv_score', 'wavelet']:
            # Max cap it softly so it doesn't cross 75 unless extreme, but don't blindly halve it.
            if score > 40:
                score = 40 + (score - 40) * 0.6
                r['score'] = score
            
        w = video_weights.get(name, 0.05)
        weighted_sum += score * w
        total_weight += w
        
        if name == 'dl_classifier':
            dl_score = score
        elif score > 40:
            high_signals += 1
            
    # Calculate legacy physics average specifically for video (Top 3 strongest signals)
    legacy_modules = [r['score'] for r in results if r.get('name') not in ['dl_classifier', 'temporal']]
    if len(legacy_modules) >= 3:
        top_modules = sorted(legacy_modules, reverse=True)
        legacy_avg = sum(top_modules[:3]) / 3
    else:
        legacy_avg = sum(legacy_modules) / max(1, len(legacy_modules))

    # Dynamic Weighting (similar to images)
    if dl_score is not None:
        if legacy_avg >= 50 and 15 <= dl_score < 50:
            final_score = (dl_score * 0.15) + (legacy_avg * 0.85)
        elif dl_score < 15:
            final_score = (dl_score * 0.85) + (legacy_avg * 0.15)
        elif dl_score >= 70:
            final_score = (dl_score * 0.70) + (legacy_avg * 0.30)
        else:
            final_score = (dl_score * 0.40) + (legacy_avg * 0.60)
    else:
        final_score = legacy_avg
    
    # NEW: Low-fidelity safety factor (compression penalty applied gently only if it struggles to be definitive)
    if is_low_res and 25 < final_score < 60:
        final_score -= 8

    # If DL is very confident (>75%), trust it heavily, but less so on low-res
    if dl_score is not None and dl_score > 75:
        dl_trust = 0.85 if not is_low_res else 0.70
        final_score = max(final_score, dl_score * dl_trust)
    
    return min(100, max(0, final_score))


def _generate_verdict(score, results):
    """Generate verdict with label, color, and key findings — binary: AI or Real."""
    if score >= 60:
        level, label, confidence, color = 'ai_high', 'verdict_ai', 'confidence_high', '#ef4444'
    elif score >= 45:
        level, label, confidence, color = 'ai_moderate', 'verdict_prob_ai', 'confidence_mod', '#f97316'
    elif score >= 35:
        level, label, confidence, color = 'ai_low', 'verdict_indicia', 'confidence_low', '#eab308'
    else:
        level, label, confidence, color = 'real', 'verdict_real', 'confidence_high', '#22c55e'

    key_findings = []
    for r in results:
        if r.get('score', 0) >= 30:
            for f in r.get('details', {}).get('findings', []):
                if not f: continue
                # Safety check: if f is a dict (localized), we don't apply legacy string filters
                if isinstance(f, str):
                    if f.startswith('Textura') or f.startswith('Estatísticas de') or 'consistente com' in f:
                        continue
                key_findings.append(f)

    return {
        'level': level, 'label': label, 'confidence': confidence,
        'color': color, 'score': round(score, 1),
        'key_findings': key_findings[:10],
    }



def _error_result(name, error):
    """Error result for failed module."""
    name_map = {
        'dl_classifier': 'dl_classifier',
        'frequency': 'frequency',
        'noise': 'noise',
        'ela': 'ela',
        'gradient': 'gradient',
        'statistical': 'statistical',
        'wavelet': 'wavelet',
        'metadata': 'metadata',
        'hv_score': 'hv_score',
        'texture': 'ai_texture',
        'temporal': 'temporal',
    }
    icons = {
        'dl_classifier': 'brain',
        'frequency': 'chart-bar', 'noise': 'microscope', 'ela': 'target', 'gradient': 'waves',
        'statistical': 'trending-up', 'wavelet': 'activity', 'metadata': 'search', 'hv_score': 'ruler',
        'texture': 'texture',
    }
    return {
        'name': name_map.get(name, name),
        'score': 50,
        'icon': icons.get(name, 'search'),
        'details': {'findings': [f'Erro na análise: {error}']},
        'visualization': None,
    }
