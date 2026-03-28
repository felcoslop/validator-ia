"""
ForensicAI — PDF Report Generator
Creates styled PDF reports with analysis results, visualizations, and frame captures.
"""

import os
import io
import base64
import tempfile
from datetime import datetime

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm, cm
from reportlab.lib.colors import HexColor, white, black
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle,
    PageBreak, HRFlowable, KeepTogether
)
from reportlab.graphics.shapes import Drawing, Rect, String, Circle
from reportlab.graphics.charts.piecharts import Pie

import qrcode
from PIL import Image as PILImage

from translations import TRANSLATIONS


# Colors (Light Mode Professional Palette)
bg_white = white
bg_light = HexColor('#f8fafc')
NAVY = HexColor('#0f172a')
SLATE = HexColor('#475569')
ACCENT = HexColor('#2563eb')
BORDER = HexColor('#e2e8f0')

# Severity Colors
GREEN = HexColor('#15803d')
YELLOW = HexColor('#b45309')
ORANGE = HexColor('#c2410c')
RED = HexColor('#b91c1c')

def _score_color(score):
    if score < 30: return GREEN
    if score < 50: return YELLOW
    if score < 75: return ORANGE
    return RED

from PIL import Image as PILImage, ImageDraw

def _b64_to_image(b64_string, max_width=None, max_height=None, rounded=True):
    """Convert base64 string to ReportLab Image with optional rounded corners."""
    if not b64_string:
        return None
    try:
        if isinstance(b64_string, str) and b64_string.startswith('data:image'):
            b64_string = b64_string.split(',')[1]
            
        img_data = base64.b64decode(b64_string)
        buf = io.BytesIO(img_data)
        pil = PILImage.open(buf).convert("RGBA")
        
        # Rounded corners logic
        if rounded:
            w, h = pil.size
            if w > 10 and h > 10:
                radius = 40 # standard radius
                mask = PILImage.new('L', pil.size, 0)
                draw = ImageDraw.Draw(mask)
                draw.rounded_rectangle((0, 0) + pil.size, radius=radius, fill=255)
                pil.putalpha(mask)

        w, h = pil.size
        # Adjust size maintaining aspect ratio
        if max_width and w > max_width:
            ratio = max_width / w
            w, h = int(w * ratio), int(h * ratio)
        if max_height and h > max_height:
            ratio = max_height / h
            w, h = int(w * ratio), int(h * ratio)

        # Save back to buffer
        out_buf = io.BytesIO()
        pil.save(out_buf, format='PNG')
        out_buf.seek(0)
        return Image(out_buf, width=w, height=h)
    except Exception as e:
        print(f"Error processing PDF image: {e}")
        return None

def generate(data, eval_id, base_url, lang='pt'):
    """Generate professional forensic PDF report."""
    
    def _(key, **kwargs):
        if isinstance(key, dict):
            params = key.copy()
            k = params.pop('key', '')
            text = TRANSLATIONS.get(lang, TRANSLATIONS['pt']).get(k, k)
            try:
                return text.format(**params)
            except:
                return text
        return TRANSLATIONS.get(lang, TRANSLATIONS['pt']).get(key, str(key))
    pdf_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'pdfs')
    os.makedirs(pdf_dir, exist_ok=True)
    pdf_path = os.path.join(pdf_dir, f'{eval_id}.pdf')

    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=A4,
        leftMargin=20 * mm,
        rightMargin=20 * mm,
        topMargin=20 * mm,
        bottomMargin=20 * mm,
    )

    styles = getSampleStyleSheet()

    # Professional Typography
    title_style = ParagraphStyle(
        'Title_Custom', parent=styles['Title'],
        fontSize=26, textColor=NAVY, spaceAfter=2 * mm,
        fontName='Helvetica-Bold', alignment=TA_LEFT
    )
    subtitle_style = ParagraphStyle(
        'Subtitle_Custom', parent=styles['Normal'],
        fontSize=10, textColor=SLATE, spaceAfter=8 * mm,
        fontName='Helvetica', leading=14
    )
    heading_style = ParagraphStyle(
        'Heading_Custom', parent=styles['Heading2'],
        fontSize=12, textColor=NAVY, spaceBefore=6 * mm, spaceAfter=3 * mm,
        fontName='Helvetica-Bold', textTransform='uppercase'
    )
    body_style = ParagraphStyle(
        'Body_Custom', parent=styles['Normal'],
        fontSize=9, textColor=black, spaceAfter=2 * mm,
        fontName='Helvetica'
    )
    meta_label_style = ParagraphStyle(
        'Meta_Label', parent=styles['Normal'],
        fontSize=8, textColor=SLATE, fontName='Helvetica-Bold'
    )
    meta_value_style = ParagraphStyle(
        'Meta_Value', parent=styles['Normal'],
        fontSize=9, textColor=NAVY, fontName='Helvetica'
    )
    finding_style = ParagraphStyle(
        'Finding_Custom', parent=styles['Normal'],
        fontSize=9, textColor=HexColor('#1e293b'), spaceAfter=1.5 * mm,
        leftIndent=4 * mm, leading=11
    )
    verdict_label_style = ParagraphStyle(
        'Verdict_Label', parent=styles['Normal'],
        fontSize=14, fontName='Helvetica-Bold', textColor=white,
        alignment=TA_CENTER, spaceBefore=4 * mm
    )

    elements = []
    verdict = data.get('verdict', {})
    modules = data.get('modules', [])
    validation_url = f'{base_url}/validar/{eval_id}'

    # Header Table (Logo | Metadata)
    logo_para = Paragraph('Forensic<font color="#2563eb">AI</font>', title_style)
    sub_para = Paragraph(_('reverse_engineering_lab').upper(), subtitle_style)
    
    header_data = [
        [logo_para, Paragraph(f'ID: <font color="#2563eb">{eval_id}</font>', meta_label_style)],
        [sub_para, Paragraph(f"{_('emission').upper()}: {datetime.now().strftime('%d/%m/%Y %H:%M')}", meta_label_style)]
    ]
    header_table = Table(header_data, colWidths=[115 * mm, 55 * mm])
    header_table.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2 * mm),
    ]))
    elements.append(header_table)
    elements.append(HRFlowable(width='100%', thickness=1.5, color=NAVY, spaceAfter=8*mm))

    # Metadata Grid
    meta_data = [
        [Paragraph(_('unique_identifier').upper(), meta_label_style), Paragraph(_('origin_file').upper(), meta_label_style), Paragraph(_('native_resolution').upper(), meta_label_style)],
        [Paragraph(eval_id, meta_value_style), Paragraph(data.get('original_filename', '-'), meta_value_style), Paragraph(data.get('dimensions', '-'), meta_value_style)],
        [Spacer(1, 2*mm), Spacer(1, 2*mm), Spacer(1, 2*mm)],
        [Paragraph(_('processing_date').upper(), meta_label_style), Paragraph(_('duration_fps').upper(), meta_label_style), Paragraph(_('hardware_agent').upper(), meta_label_style)],
        [Paragraph(data.get('created_at_display', '-'), meta_value_style), 
         Paragraph(f"{data.get('duration', 'N/A')} @ {data.get('fps', 'N/A')} FPS" if data.get('type') == 'video' else f"N/A ({_('static')})", meta_value_style),
         Paragraph('Forensic Engine v1.0', meta_value_style)]
    ]
    meta_table = Table(meta_data, colWidths=[55 * mm, 65 * mm, 50 * mm])
    meta_table.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
    ]))
    elements.append(meta_table)
    elements.append(Spacer(1, 10 * mm))

    # Verdict Highlights Box
    v_color = _score_color(data.get('final_score', 0))
    verdict_data = [
        [Paragraph(f"{_('verdict').upper()}: {_(verdict.get('label', 'N/A'))}", verdict_label_style)],
        [Paragraph(f"{_('synthetic_generation_probability').upper()}: {data.get('final_score')}%", 
                   ParagraphStyle('VScore', parent=body_style, textColor=white, alignment=TA_CENTER, fontName='Helvetica-Bold'))]
    ]
    v_table = Table(verdict_data, colWidths=[170 * mm])
    v_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), v_color),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8 * mm),
        ('TOPPADDING', (0,0), (-1,-1), 4 * mm),
    ]))
    elements.append(v_table)
    elements.append(Spacer(1, 4 * mm))

    # Key Findings
    elements.append(Paragraph(_('detected_vectors_of_interest').upper(), heading_style))
    for f in verdict.get('key_findings', []):
        elements.append(Paragraph(f'[-] {_(f)}', finding_style))
    
    elements.append(Spacer(1, 8 * mm))

    # ===== ORIGINAL MEDIA =====
    thumb_b64 = data.get('original_thumbnail', '')
    if thumb_b64:
        elements.append(Paragraph(_('analyzed_media_object').upper(), heading_style))
        rl_img = _b64_to_image(thumb_b64, max_width=170 * mm, max_height=100 * mm, rounded=True)
        if rl_img:
            elements.append(rl_img)
        elements.append(Spacer(1, 8 * mm))

    # ===== MODULE RESULTS =====
    elements.append(Paragraph('MATRIZ DE VETORES FORENSES', heading_style))
    elements.append(Spacer(1, 2 * mm))

    # Summary table
    summary_data = [['Vetor de Análise', 'Score', 'Status Probabilístico']]
    for mod in modules:
        score = mod.get('score', 0)
        status_key = 'status_normal' if score < 30 else ('status_attention' if score < 50 else ('status_suspicious' if score < 75 else 'status_critical'))
        status = _(status_key).upper()
        name = _(mod.get('name', '')).upper()
        summary_data.append([name, f"{score}%", status])

    summary_table = Table(summary_data, colWidths=[90 * mm, 30 * mm, 50 * mm])
    summary_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('BACKGROUND', (0, 0), (-1, 0), HexColor('#f1f5f9')),
        ('TEXTCOLOR', (0, 0), (-1, 0), NAVY),
        ('GRID', (0, 0), (-1, -1), 0.5, BORDER),
        ('ALIGN', (1, 0), (2, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    elements.append(summary_table)
    elements.append(Spacer(1, 10 * mm))

    # Detail per module
    for mod in modules:
        mod_elements = []
        mod_elements.append(Paragraph(f"{_('analysis').upper()}: {_(mod.get('name', '')).upper()}", heading_style))
        
        findings = mod.get('details', {}).get('findings', [])
        for f in findings:
            mod_elements.append(Paragraph(f'[+] {_(f)}', finding_style))

        # Visualization
        viz = mod.get('visualization')
        if viz:
            rl_img = _b64_to_image(viz, max_width=170 * mm, max_height=140 * mm, rounded=True)
            if rl_img:
                mod_elements.append(Spacer(1, 4 * mm))
                mod_elements.append(rl_img)

        mod_elements.append(Spacer(1, 8 * mm))
        elements.append(KeepTogether(mod_elements))

    # ===== VIDEO FRAMES =====
    frame_thumbs = data.get('frame_thumbnails', [])
    if frame_thumbs:
        elements.append(PageBreak())
        elements.append(Paragraph(_('analyzed_frame_chronology').upper(), heading_style))
        elements.append(Spacer(1, 2 * mm))

        # 2 frames per row
        row = []
        for i, ft in enumerate(frame_thumbs):
            f_box = []
            f_box.append(Paragraph(f"{_('frame').upper()} #{ft.get('index', i)} - {ft.get('timestamp', '')}", 
                                  ParagraphStyle('F', parent=body_style, fontSize=7, fontName='Helvetica-Bold')))
            rl_img = _b64_to_image(ft.get('thumbnail', ''), max_width=80 * mm, max_height=50 * mm, rounded=True)
            if rl_img:
                f_box.append(rl_img)
            f_box.append(Spacer(1, 4 * mm))
            row.append(f_box)
            
            if len(row) == 2 or i == len(frame_thumbs) - 1:
                col_widths = [85 * mm] * len(row)
                t = Table([row], colWidths=col_widths)
                t.setStyle(TableStyle([('VALIGN', (0,0), (-1,-1), 'TOP')]))
                elements.append(t)
                row = []

    # Final Page: Certification
    elements.append(PageBreak())
    elements.append(Spacer(1, 40 * mm))
    # Certification QR
    elements.append(Paragraph(_('digital_authenticity_validation').upper(), 
                               ParagraphStyle('C', parent=body_style, alignment=TA_CENTER, fontName='Helvetica-Bold', fontSize=10, textColor=NAVY)))
    elements.append(Spacer(1, 8 * mm))
    
    try:
        qr = qrcode.QRCode(version=1, box_size=8, border=2)
        qr.add_data(validation_url)
        qr.make(fit=True)
        img_qr = qr.make_image(fill_color='black', back_color='white')
        buf = io.BytesIO()
        img_qr.save(buf, format='PNG')
        buf.seek(0)
        
        qr_img = Image(buf, width=50 * mm, height=50 * mm)
        qr_table = Table([[qr_img]], colWidths=[170 * mm])
        qr_table.setStyle(TableStyle([('ALIGN', (0,0), (-1,-1), 'CENTER')]))
        elements.append(qr_table)
    except: pass

    elements.append(Paragraph(f'<font color="#2563eb"><u>{validation_url}</u></font>', 
                               ParagraphStyle('C', parent=body_style, alignment=TA_CENTER, fontSize=8)))

    # Footer on every page
    from datetime import timedelta, timezone
    offset = timezone(timedelta(hours=-3))
    now_br = datetime.now(offset).strftime("%d/%m/%Y %H:%M:%S")

    def draw_footer(canvas, doc):
        canvas.saveState()
        canvas.setFont('Helvetica', 8)
        canvas.setStrokeColor(SLATE)
        canvas.setLineWidth(0.5)
        canvas.line(20*mm, 15*mm, 190*mm, 15*mm)
        copyright_text = _('copyright_notice') if 'copyright_notice' in TRANSLATIONS.get(lang, {}) else "© 2026 ForensicAI"
        footer_text = f"{copyright_text} • {_('investigation_matrix')}: {eval_id} • {_('generated_at')}: {now_br} (GMT-3)"
        canvas.drawCentredString(105*mm, 10*mm, footer_text)
        canvas.setFont('Helvetica', 7)
        canvas.drawRightString(190*mm, 10*mm, f"{_('page')} {canvas.getPageNumber()}")
        canvas.restoreState()

    doc.build(elements, onFirstPage=draw_footer, onLaterPages=draw_footer)
    return pdf_path
