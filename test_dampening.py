import sys
import json
from engine import analyze_image, analyze_video

files = [
    r"c:\Users\manu_\Downloads\detecção ia\WhatsApp Video 2026-03-28 at 17.01.55.mp4",
    r"c:\Users\manu_\Downloads\detecção ia\whatsapp_tiktok_video.mp4",
    r"c:\Users\manu_\Downloads\detecção ia\uploads\downloaded_24a732baacad.mp4",
    r"c:\Users\manu_\Downloads\detecção ia\uploads\c985c68563e24036b547d1fe73a2f24b_Captura_de_tela_2026-03-28_171722.png",
    r"c:\Users\manu_\Downloads\detecção ia\uploads\0f10b986670a494688d48bce88899f19_Captura_de_tela_2026-03-28_155147.png",
    r"c:\Users\manu_\Downloads\detecção ia\Captura de tela 2026-01-20 140028.png",
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
