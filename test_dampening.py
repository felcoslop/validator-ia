import sys
import json
from engine import analyze_image, analyze_video

files = [
    r"c:\Users\manu_\Downloads\whatsapp_tiktok_video.mp4", # Fallback downloaded video or exact video if I can find it
    r"c:\Users\manu_\Downloads\_MG_8269 (1).JPG"
]

for f in files:
    print(f"\n=========================================")
    print(f"--- TESTANDO ARQUIVO: {f} ---")
    print(f"=========================================")
    try:
        if f.endswith('.mp4'):
            res = analyze_video(f, "test")
        else:
            res = analyze_image(f, "test")
            
        if 'error' in res:
            print(f"ERRO INTERNO NO ENGINE: {res['error']}")
        else:
            print(f"SCORE FINAL: {res.get('final_score')}")
            for mod in res.get('modules', []):
                name = mod.get('name')
                score = mod.get('score')
                print(f" -> {name}: {score}")
    except Exception as e:
        print(f"ERRO AO PROCESSAR {f}: {e}")
