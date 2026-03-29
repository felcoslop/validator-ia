import os
import sys
from engine import analyze_image

def diag():
    img_path = r"c:\Users\manu_\Downloads\detecção ia\WhatsApp Image 2026-03-28 at 16.53.31 copy.jpeg"
    result = analyze_image(img_path, "diag_copy")
    print(f"RES: {result['final_score']}")
    for m in result['modules']:
        print(f"MOD | {m['name']} | {m['score']}")

if __name__ == "__main__":
    diag()
