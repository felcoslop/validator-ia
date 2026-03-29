"""Full diagnostic: decompose ALL module sub-metrics for AI video vs real photo."""
import cv2
import numpy as np
import sys
sys.path.insert(0, '.')

# Load images
img_real = cv2.imread(r"c:\Users\manu_\Downloads\_MG_8269 (1).JPG")
h, w = img_real.shape[:2]
if max(h, w) > 1024:
    scale = 1024 / max(h, w)
    img_real = cv2.resize(img_real, (int(w * scale), int(h * scale)))

cap = cv2.VideoCapture(r"c:\Users\manu_\Downloads\detecção ia\whatsapp_tiktok_video.mp4")
ret, frame_ai = cap.read()
cap.release()
h, w = frame_ai.shape[:2]
if max(h, w) > 1024:
    scale = 1024 / max(h, w)
    frame_ai = cv2.resize(frame_ai, (int(w * scale), int(h * scale)))

# --- FREQUENCY ---
from analyzers import frequency as freq_mod
for label, img in [("REAL", img_real), ("AI_VIDEO", frame_ai)]:
    r = freq_mod.analyze(img)
    print(f"\n=== FREQUENCY: {label} === score={r['score']}")
    for k, v in r['details'].items():
        if k != 'findings':
            print(f"  {k}: {v}")

# --- STATISTICAL ---
from analyzers import statistical as stat_mod
for label, img in [("REAL", img_real), ("AI_VIDEO", frame_ai)]:
    r = stat_mod.analyze(img)
    print(f"\n=== STATISTICAL: {label} === score={r['score']}")
    for k, v in r['details'].items():
        if k != 'findings':
            print(f"  {k}: {v}")

# --- GRADIENT (cross-metric detail) ---
from analyzers.gradient import _local_entropy, _edge_coherence, _analyze_transitions, _detect_halos, _texture_regularity
for label, img in [("REAL", img_real), ("AI_VIDEO", frame_ai)]:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64)
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(gx**2 + gy**2)
    dirr = np.arctan2(gy, gx)
    ent = _local_entropy(gray, 16)
    mean_g = np.mean(mag)
    std_g = np.std(mag)
    cv = std_g / (mean_g + 1e-10)
    ec = _edge_coherence(dirr, mag)
    ts = _analyze_transitions(ent, mag)
    hs = _detect_halos(mag, gray)
    rg = _texture_regularity(mag)
    print(f"\n=== GRADIENT: {label} ===")
    print(f"  mean_grad={mean_g:.2f} std={std_g:.2f} cv={cv:.4f}")
    print(f"  edge_coh={ec:.4f} trans={ts:.4f} halo={hs:.4f} reg={rg:.4f}")

# --- TEXTURE ---
from analyzers import texture as tex_mod
for label, img in [("REAL", img_real), ("AI_VIDEO", frame_ai)]:
    r = tex_mod.analyze(img)
    print(f"\n=== TEXTURE: {label} === score={r['score']}")
    for k, v in r['details'].items():
        if k != 'findings':
            print(f"  {k}: {v}")
