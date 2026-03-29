"""
Metadata / EXIF Analysis Module
Extracts and analyzes image metadata for signs of AI generation or manipulation.
"""

import numpy as np
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
from io import BytesIO
import json


# Known AI software markers
AI_SOFTWARE_MARKERS = [
    'dall-e', 'dalle', 'midjourney', 'stable diffusion', 'stablediffusion',
    'novelai', 'artbreeder', 'deepai', 'nightcafe', 'craiyon',
    'leonardo.ai', 'firefly', 'adobe firefly', 'bing image creator',
    'playground ai', 'ideogram', 'flux', 'comfyui', 'automatic1111',
    'invoke ai', 'sdxl', 'sd 1.5', 'sd 2.1', 'replicate',
    'runway', 'pika', 'sora', 'gen-2', 'gen-3',
    'imagen', 'gemini', 'copilot designer', 'chatgpt',
]

# Known camera manufacturers
CAMERA_BRANDS = [
    'canon', 'nikon', 'sony', 'fujifilm', 'olympus', 'panasonic',
    'pentax', 'leica', 'hasselblad', 'samsung', 'apple', 'google',
    'huawei', 'xiaomi', 'oppo', 'oneplus', 'motorola', 'lg',
]


def analyze(image_path):
    """
    Analyze image metadata and EXIF data.
    Returns score (0-100), details, and findings.
    """
    try:
        pil_img = Image.open(image_path)
    except Exception:
        return {
            'name': 'Análise de Metadados (EXIF)',
            'score': 50,
            'details': {'findings': ['Não foi possível ler metadados da imagem']},
            'visualization': None,
            'icon': '🔍'
        }

    metadata = {}
    exif_found = False
    camera_found = False
    software_found = None
    ai_marker_found = None
    gps_found = False

    # Extract EXIF
    exif_data = pil_img.getexif()
    if exif_data:
        exif_found = True
        for tag_id, value in exif_data.items():
            tag = TAGS.get(tag_id, tag_id)
            try:
                metadata[str(tag)] = str(value)
            except Exception:
                pass

    # Check for camera make/model
    make = metadata.get('Make', '').lower()
    model = metadata.get('Model', '')
    if make:
        camera_found = any(b in make for b in CAMERA_BRANDS)

    # Check for software
    software = metadata.get('Software', '').lower()
    processing_software = metadata.get('ProcessingSoftware', '').lower()
    all_software = software + ' ' + processing_software

    if all_software.strip():
        software_found = metadata.get('Software', '') or metadata.get('ProcessingSoftware', '')
        for marker in AI_SOFTWARE_MARKERS:
            if marker in all_software:
                ai_marker_found = marker
                break

    # Check for GPS data
    gps_info = exif_data.get(34853)
    if gps_info:
        gps_found = True

    # Check image format specifics
    format_info = pil_img.format or 'Unknown'
    mode = pil_img.mode
    size = pil_img.size

    # Check for C2PA / Content Credentials marker
    c2pa_found = False
    try:
        info = pil_img.info
        for key in info:
            if 'c2pa' in str(key).lower() or 'contentcredentials' in str(key).lower():
                c2pa_found = True
                break
    except Exception:
        pass

    # Check for common AI image dimensions (often powers of 2 or specific sizes)
    ai_dimensions = [
        (512, 512), (768, 768), (1024, 1024), (1536, 1536), (2048, 2048),
        (512, 768), (768, 512), (1024, 1536), (1536, 1024),
        (1024, 1792), (1792, 1024),
        (896, 1152), (1152, 896),
    ]
    is_ai_dimension = size in ai_dimensions

    # Sensor quality check (Webcam mitigation)
    from . import utils as forensic_utils
    sensor_score = forensic_utils.detect_low_end_sensor(None, metadata)
    
    # Score
    score = 0

    if ai_marker_found:
        score += 60  # Strong indicator

    if c2pa_found:
        score += 20  # May indicate AI origin with content credentials

    if not exif_found:
        # If no EXIF but common webcam res, don't penalize as heavily
        score += 5 if sensor_score > 0 else 15

    if exif_found and not camera_found and not software_found:
        score += 10

    if camera_found:
        score -= 20  # Camera metadata strongly suggests real

    if gps_found:
        score -= 15  # GPS data strongly suggests real camera

    if sensor_score > 0.5:
        score -= 10  # Dampen if it's a known webcam

    if is_ai_dimension:
        score += 15  # Common AI generation sizes

    # Very round DPI values or unusual
    dpi = pil_img.info.get('dpi', (72, 72))
    if isinstance(dpi, tuple) and len(dpi) == 2:
        if dpi == (72, 72) or dpi == (96, 96):
            # Most webcams use 72/96 DPI
            if sensor_score > 0.5:
                score += 0
            else:
                score += 5

    score = min(100, max(0, score))

    details = {
        'format': format_info,
        'dimensions': f'{size[0]}x{size[1]}',
        'mode': mode,
        'exif_present': exif_found,
        'camera_make': metadata.get('Make', 'N/A'),
        'camera_model': model or 'N/A',
        'software': software_found or 'N/A',
        'gps_present': gps_found,
        'ai_marker': ai_marker_found,
        'c2pa_present': c2pa_found,
        'is_common_ai_size': is_ai_dimension,
        'sensor_type': 'Webcam/Low-End' if sensor_score > 0.5 else 'Standard',
        'total_exif_tags': len(metadata),
        'findings': []
    }

    if ai_marker_found:
        details['findings'].append({'key': 'finding_meta_software', 'software': ai_marker_found})
    if c2pa_found:
        details['findings'].append({'key': 'finding_meta_software', 'software': 'C2PA/Content Credentials'})
    if sensor_score > 0.5:
        details['findings'].append({'key': 'finding_meta_webcam'})
    if camera_found:
        details['findings'].append({'key': 'finding_meta_camera', 'camera': metadata.get("Make", ""), 'model': model})
    if gps_found:
        details['findings'].append({'key': 'finding_meta_natural'}) # GPS is a natural indicator
    if not exif_found:
        details['findings'].append({'key': 'finding_meta_no_exif'})
    if is_ai_dimension:
        details['findings'].append({'key': 'finding_meta_no_exif'}) # Generic meta anomaly
    if not details['findings']:
        details['findings'].append({'key': 'finding_meta_natural'})

    return {
        'name': 'Análise de Metadados (EXIF)',
        'score': score,
        'details': details,
        'visualization': None,
        'icon': '🔍'
    }


def analyze_video(file_path):
    """
    Extract and analyze video container metadata using ffprobe.
    """
    import subprocess
    import json
    import os

    ffprobe_path = r"c:\Users\manu_\Downloads\projeto_village\Sender MiniCraques\ffmpeg\ffmpeg-8.0-essentials_build\bin\ffprobe.exe"
    
    # Fallback if ffprobe isn't there
    if not os.path.exists(ffprobe_path):
        return {
            'name': 'metadata',
            'score': 0,
            'details': {
                'format': 'N/A',
                'dimensions': 'N/A',
                'mode': 'N/A',
                'camera_make': 'N/A',
                'software': 'N/A',
                'total_exif_tags': 0,
                'findings': [{'key': 'finding_meta_no_exif'}]
            },
            'icon': 'search-mod'
        }

    try:
        cmd = [
            ffprobe_path, 
            "-v", "quiet", 
            "-print_format", "json", 
            "-show_format", 
            "-show_streams", 
            file_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        data = json.loads(result.stdout)
        
        fmt = data.get('format', {})
        streams = data.get('streams', [])
        video_stream = next((s for s in streams if s.get('codec_type') == 'video'), {})
        
        tags = fmt.get('tags', {})
        v_tags = video_stream.get('tags', {})
        all_tags = {**tags, **v_tags}
        
        software = all_tags.get('encoder') or all_tags.get('compatible_brands') or all_tags.get('major_brand', 'N/A')
        make = all_tags.get('make') or all_tags.get('com.apple.quicktime.make') or 'Unknown'
        model = all_tags.get('model') or all_tags.get('com.apple.quicktime.model') or ''
        
        # Scoring video metadata
        score = 0
        findings = []
        
        is_ai_software = any(marker in str(software).lower() for marker in AI_SOFTWARE_MARKERS)
        if is_ai_software:
            score += 50
            findings.append({'key': 'finding_meta_software', 'software': software})
            
        if not tags and not v_tags:
            score += 10
            findings.append({'key': 'finding_meta_no_exif'})
            
        if any(brand in make.lower() for brand in CAMERA_BRANDS):
            score -= 15
            findings.append({'key': 'finding_meta_camera', 'camera': make, 'model': model})
            
        if not findings:
            findings.append({'key': 'finding_meta_natural'})
            
        return {
            'name': 'metadata',
            'score': min(100, max(0, score)),
            'details': {
                'format': fmt.get('format_long_name', 'MPEG-4').split('/')[0],
                'dimensions': f"{video_stream.get('width')}x{video_stream.get('height')}",
                'mode': video_stream.get('pix_fmt', 'yuv420p'),
                'camera_make': make,
                'software': software,
                'total_exif_tags': len(all_tags),
                'findings': findings
            },
            'icon': 'search-mod'
        }
    except Exception as e:
        return {
            'name': 'metadata',
            'score': 0,
            'details': {'findings': [f'Error: {str(e)}']},
            'icon': 'search-mod'
        }
