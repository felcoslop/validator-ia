import engine

def test_user_scenario():
    # User-reported results for the Instagram Reel (360x640)
    # 78.9% total score was reported previously.
    
    results = [
        {
            'name': 'Classificador Neural (DL)',
            'score': 80.9,
            'details': {'findings': ['Sinais de geração por IA detectados']}
        },
        {
            'name': 'Análise de Frequência (FFT)',
            'score': 65.0, # Guessing based on "incompatível"
            'details': {'findings': ['Espectro de frequência incompatível']}
        },
        {
            'name': 'Análise de Ruído (PRNU)',
            'score': 60.0, # Guessing based on "Variância uniforme"
            'details': {'findings': ['Nível de ruído incompatível']}
        },
        {
            'name': 'Score Horizontal/Vertical',
            'score': 55.0, # Guessing based on "Picos periódicos 8H+10V"
            'details': {'findings': ['Picos periódicos nos eixos H/V']}
        },
        {
            'name': 'Análise Wavelet (DWT)',
            'score': 50.0,
            'details': {'findings': ['Forte anisotropia direcional']}
        },
        {
            'name': 'Análise Temporal de Vídeo',
            'score': 45.0,
            'details': {'findings': ['Consistência suspeita']}
        }
    ]
    
    width, height = 360, 640
    
    print("--- SIMULAÇÃO DE ANÁLISE ---")
    score = engine._compute_final_score_video(results, width, height)
    verdict = engine._generate_verdict(score, results)
    
    print(f"SCORE FINAL: {score}%")
    print(f"VEREDITO: {verdict['label']} ({verdict['level']})")
    print("Scores Individuais após compensação:")
    for r in results:
        print(f"  - {r['name']}: {r['score']:.1f}")

if __name__ == "__main__":
    test_user_scenario()
