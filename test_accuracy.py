import sys, os, json, traceback
import engine

BASE_DIR = r"c:\Users\manu_\Downloads\detecção ia"
ai_img = os.path.join(BASE_DIR, "Captura de tela 2026-03-28 155147.png")
real_img = os.path.join(BASE_DIR, "Captura de tela 2026-01-20 140028.png")

def test_file(path, expected_type):
    print(f"\n{'='*60}")
    print(f"  {expected_type}: {os.path.basename(path)}")
    print(f"{'='*60}")
    if not os.path.exists(path):
        print(f"  FILE NOT FOUND: {path}")
        return
    try:
        result = engine.analyze_image(path, "test-eval")
        if 'error' in result:
            print(f"  ENGINE ERROR: {result['error']}")
            return
        score = result.get('final_score', 0)
        verdict = result.get('verdict', {})
        print(f"  FINAL SCORE: {score}%")
        print(f"  VERDICT: {verdict.get('label')} ({verdict.get('confidence')})")
        print(f"\n  MODULE BREAKDOWN:")
        for m in result.get('modules', []):
            s = m.get('score', 0)
            name = m.get('name', '?')
            findings = m.get('details', {}).get('findings', [])
            print(f"    {s:3d}%  {name}")
            for f in findings:
                print(f"         -> {f}")
    except Exception:
        traceback.print_exc()

if __name__ == "__main__":
    test_file(ai_img, "AI IMAGE (Expected ~98%)")
    test_file(real_img, "REAL IMAGE (Expected ~0%)")
