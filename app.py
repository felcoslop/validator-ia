"""
ForensicAI — Flask Application
Persistent evaluations with unique URLs, QR codes, and PDF reports.
"""

import os
import json
import uuid
import time
from datetime import datetime
from flask import Flask, request, render_template, jsonify, send_file, url_for, redirect
from werkzeug.utils import secure_filename
import engine
import pdf_report
import yt_dlp
import socket

# Set global timeout for network operations
socket.setdefaulttimeout(60)

app = Flask(__name__)

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Easypanel / Docker optimization: use environment variables for persistent paths
# If running in Docker, these will typically point to /data
UPLOAD_FOLDER = os.getenv('UPLOADS_PATH', os.path.join(BASE_DIR, 'uploads'))
DATA_FOLDER = os.getenv('DATA_PATH', os.path.join(BASE_DIR, 'data', 'evaluations'))

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(os.path.dirname(DATA_FOLDER), exist_ok=True) # Ensure data/ exists
os.makedirs(DATA_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB

ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp', 'bmp', 'tiff', 'tif'}
ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov', 'webm', 'mkv'}
ALLOWED_EXTENSIONS = ALLOWED_IMAGE_EXTENSIONS | ALLOWED_VIDEO_EXTENSIONS


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_file_type(filename):
    ext = filename.rsplit('.', 1)[1].lower()
    if ext in ALLOWED_IMAGE_EXTENSIONS:
        return 'image'
    elif ext in ALLOWED_VIDEO_EXTENSIONS:
        return 'video'
    return None


def save_evaluation(eval_id, data):
    """Save evaluation to JSON file."""
    filepath = os.path.join(DATA_FOLDER, f'{eval_id}.json')
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_evaluation(eval_id):
    """Load evaluation from JSON file."""
    filepath = os.path.join(DATA_FOLDER, f'{eval_id}.json')
    if not os.path.exists(filepath):
        return None
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


# ---- Routes ----

@app.route('/')
def index():
    """Main upload page."""
    return render_template('index.html')


@app.route('/api/analyze', methods=['POST'])
def analyze():
    """Upload and analyze or analyze previously uploaded video with selected frames."""
    eval_id = str(uuid.uuid4())[:12]
    now = datetime.now()
    
    # Check if we are analyzing selected frames from a previous storyboard
    selected_indices = request.form.getlist('indices[]')
    temp_filename = request.form.get('temp_filename')
    
    if temp_filename:
        # Re-using already uploaded file
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], temp_filename)
        if not os.path.exists(filepath):
            return jsonify({'error': 'Arquivo temporário expirou ou não existe'}), 400
        filename = temp_filename.split('_', 1)[1]
    else:
        # Standard upload
        if 'file' not in request.files:
            return jsonify({'error': 'Nenhum arquivo enviado'}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'Nenhum arquivo selecionado'}), 400
        if not allowed_file(file.filename):
            return jsonify({'error': 'Tipo de arquivo não suportado'}), 400
            
        filename = secure_filename(file.filename)
        unique_name = f'{uuid.uuid4().hex}_{filename}'
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_name)
        file.save(filepath)

    try:
        file_type = get_file_type(filename)
        
        if file_type == 'image':
            result = engine.analyze_image(filepath, eval_id)
        elif file_type == 'video':
            result = engine.analyze_video(filepath, eval_id, selected_indices=selected_indices)
        else:
            return jsonify({'error': 'Tipo de arquivo não reconhecido'}), 400

        if 'error' in result:
             # If error happens in engine, we still want to save it for display
             result['eval_id'] = eval_id
             result['created_at_display'] = now.strftime('%d/%m/%Y às %H:%M:%S')
             save_evaluation(eval_id, result)
             return jsonify({'eval_id': eval_id, 'redirect': f'/avaliacao/{eval_id}'})

        # Add metadata
        result['eval_id'] = eval_id
        result['created_at'] = now.strftime('%Y-%m-%d %H:%M:%S')
        result['created_at_display'] = now.strftime('%d/%m/%Y às %H:%M:%S')
        result['original_filename'] = filename # Using filename from secure_filename/temp_filename

        # Save evaluation
        save_evaluation(eval_id, result)
        return jsonify({'eval_id': eval_id, 'redirect': f'/avaliacao/{eval_id}'})

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Erro durante análise: {str(e)}'}), 500
    finally:
        # Clean up only if not waiting for more steps or if explicitly told to
        if not request.form.get('keep'):
            try:
                os.remove(filepath)
            except OSError:
                pass



@app.route('/api/analyze-url', methods=['POST'])
def analyze_url():
    """Download video from URL and prepare for storyboard selection."""
    data = request.json
    url = data.get('url')
    print(f"[*] Rota /api/analyze-url atingida. URL: {url}", flush=True)

    if not url:
        return jsonify({'error': 'URL não fornecida'}), 400

    try:
        # Generate a temporary filename for the downloaded video
        eval_id = uuid.uuid4().hex[:12]
        unique_name = f'downloaded_{eval_id}'
        filepath_template = os.path.join(app.config['UPLOAD_FOLDER'], unique_name)

        # Forensic-grade robust options for social media platforms
        ydl_opts = {
            'outtmpl': f'{filepath_template}.%(ext)s',
            'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
            'merge_output_format': 'mp4',
            'max_filesize': 250 * 1024 * 1024,
            'quiet': True,
            'no_warnings': True,
            'ignoreerrors': False,
            'nocheckcertificate': True,
            'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
            'referer': 'https://www.google.com/',
            'http_headers': {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.9',
                'Sec-Fetch-Mode': 'navigate',
            }
        }

        print(f"[*] Solicitando download forense de: {url}", flush=True)
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            try:
                info = ydl.extract_info(url, download=True)
                filename = ydl.prepare_filename(info)
                print(f"[+] Download concluído: {filename}", flush=True)
            except Exception as download_error:
                print(f"[!] Erro no download yt-dlp: {str(download_error)}", flush=True)
                return jsonify({'error': f'Falha ao baixar mídia: Plafatorma bloqueou o acesso ou requer login. ({str(download_error)})'}), 400

        if not os.path.exists(filename):
            print(f"[!] Arquivo não encontrado após download: {filename}", flush=True)
            return jsonify({'error': 'Arquivo de mídia não encontrado após processamento'}), 500

        # Pass the final filename (basename) to the next step
        final_basename = os.path.basename(filename)
        return jsonify({
            'eval_id': eval_id,
            'temp_filename': final_basename,
            'redirect': f'/selecionar-frames?file={final_basename}'
        })

    except Exception as e:
        print(f"[!] Erro crítico em analyze_url: {str(e)}", flush=True)
        return jsonify({'error': f'Erro interno ao processar URL: {str(e)}'}), 500


@app.route('/api/storyboard', methods=['POST'])
def storyboard():
    """Extract storyboard frames for video before full analysis."""
    if 'file' not in request.files:
        return jsonify({'error': 'Nenhum arquivo enviado'}), 400
    
    file = request.files['file']
    if not allowed_file(file.filename):
        return jsonify({'error': 'Tipo de arquivo não suportado'}), 400
        
    filename = secure_filename(file.filename)
    unique_name = f'{uuid.uuid4().hex}_{filename}'
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_name)
    file.save(filepath)
    
    try:
        frames = engine.extract_video_storyboard(filepath)
        if not frames:
            return jsonify({'error': 'Não foi possível extrair frames do vídeo'}), 500
            
        return jsonify({
            'temp_filename': unique_name,
            'redirect': f'/selecionar-frames?file={unique_name}'
        })
    except Exception as e:
        return jsonify({'error': f'Erro ao extrair storyboard: {str(e)}'}), 500


@app.route('/selecionar-frames')
def select_frames_page():
    """Render the frame selection page."""
    temp_filename = request.args.get('file')
    if not temp_filename:
        return redirect(url_for('index'))
    
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], temp_filename)
    frames = engine.extract_video_storyboard(filepath)
    if not frames:
        return redirect(url_for('index'))
        
    return render_template('select_frames.html', temp_filename=temp_filename, frames=frames)


@app.route('/api/shuffle-frames')
def shuffle_frames():
    """Re-extract storyboard frames with random sampling offset."""
    temp_filename = request.args.get('file')
    if not temp_filename:
        return jsonify({'error': 'Missing file parameter'}), 400
    
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], temp_filename)
    if not os.path.exists(filepath):
        return jsonify({'error': 'File not found'}), 404
    
    # Add random offset to get different frames
    import random
    count = random.choice([20, 22, 24, 26])  # Slight variation in count
    frames = engine.extract_video_storyboard(filepath, count=count)
    if not frames:
        return jsonify({'error': 'Failed to extract frames'}), 500
    
    return jsonify({'frames': frames})



@app.route('/avaliacao/<eval_id>')
def evaluation_page(eval_id):
    """Evaluation results page with persistent URL."""
    data = load_evaluation(eval_id)
    if not data:
        return render_template('not_found.html'), 404
    return render_template('evaluation.html', data=data, eval_id=eval_id)


@app.route('/validar/<eval_id>')
def validate_page(eval_id):
    """Validation page (like ClickSign) with timestamp and download."""
    data = load_evaluation(eval_id)
    if not data:
        return render_template('not_found.html'), 404
    return render_template('validate.html', data=data, eval_id=eval_id)





@app.route('/api/qrcode/<eval_id>')
def qrcode_image(eval_id):
    """Generate QR code for evaluation URL."""
    import qrcode
    from io import BytesIO

    base_url = request.host_url.rstrip('/')
    url = f'{base_url}/validar/{eval_id}'

    qr = qrcode.QRCode(version=1, box_size=10, border=4)
    qr.add_data(url)
    qr.make(fit=True)
    img = qr.make_image(fill_color='black', back_color='white')

    buf = BytesIO()
    img.save(buf, format='PNG')
    buf.seek(0)

    return send_file(buf, mimetype='image/png')


@app.route('/api/pdf/<eval_id>')
def download_pdf(eval_id):
    """Generate and download PDF report."""
    data = load_evaluation(eval_id)
    if not data:
        return jsonify({'error': 'Avaliação não encontrada'}), 404

    base_url = request.host_url.rstrip('/')
    pdf_path = pdf_report.generate(data, eval_id, base_url)

    return send_file(
        pdf_path,
        as_attachment=True,
        download_name=f'ForensicAI-{eval_id}.pdf',
        mimetype='application/pdf'
    )


@app.route('/api/health')
def health():
    return jsonify({'status': 'ok', 'version': '1.0.0'})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)
