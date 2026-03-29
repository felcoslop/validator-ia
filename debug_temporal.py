"""Debug: extract temporal sub-metrics for AI video."""
import cv2
import numpy as np
import sys
sys.path.insert(0, '.')
from analyzers import video as vid_mod

video_path = r"c:\Users\manu_\Downloads\detecção ia\whatsapp_tiktok_video.mp4"
cap = cv2.VideoCapture(video_path)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
n_samples = min(10, total_frames)
frame_indices = np.linspace(0, total_frames - 1, n_samples, dtype=int)
frames = []
for idx in frame_indices:
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, frame = cap.read()
    if ret:
        frames.append(frame)
cap.release()

if len(frames) < 2:
    print("ERRO: poucos frames")
    sys.exit(1)

noise_cons = vid_mod._temporal_noise_consistency(frames)
spec_stab = vid_mod._spectral_stability(frames)
flow_score = vid_mod._optical_flow_analysis(frames)
flicker = vid_mod._detect_flickering(frames)

print(f"=== TEMPORAL DIAGNOSTIC (AI VIDEO) ===")
print(f"  noise_consistency: {noise_cons:.4f}")
print(f"  spectral_stability: {spec_stab:.4f}")
print(f"  flow_anomaly:      {flow_score:.4f}")
print(f"  flicker_score:     {flicker:.4f}")

# Also check per-frame frequency/stat values
from analyzers import frequency, statistical
f_scores = []
s_scores = []
for f in frames[:5]:
    fr = frequency.analyze(f, mode='video')
    sr = statistical.analyze(f, mode='video')
    f_scores.append(fr['score'])
    s_scores.append(sr['score'])
    print(f"  Frame: freq={fr['score']} stat={sr['score']}")

mean_f = np.mean(f_scores)
mean_s = np.mean(s_scores)
mean_all = (mean_f + mean_s) / 2
print(f"\n  Mean Frame Score: {mean_all:.2f}")

# Recalculate score based on current logic
score = 0
if noise_cons > 0.5: score += 25
elif noise_cons > 0.2: score += 15
elif noise_cons > 0.1: score += 8

if spec_stab > 0.5: score += 25
elif spec_stab > 0.2: score += 15
elif spec_stab > 0.1: score += 8

score += min(15, int(flow_score * 15))
score += min(30, int(mean_all * 0.45))
score += min(15, int(flicker * 15))

print(f"\n  PREDICTED TEMPORAL SCORE: {score}")
