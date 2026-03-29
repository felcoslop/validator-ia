(function(){
    'use strict';

    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const preview = document.getElementById('preview');
    const previewMedia = document.getElementById('preview-media');
    const pfName = document.getElementById('pf-name');
    const pfSize = document.getElementById('pf-size');
    const btnRemove = document.getElementById('pf-remove');
    const btnAnalyze = document.getElementById('btn-analyze');
    const secUpload = document.getElementById('sec-upload');
    const secLoading = document.getElementById('sec-loading');
    const progressFill = document.getElementById('progress-fill');
    const loadingText = document.getElementById('loading-text');
    const loadSteps = document.getElementById('load-steps');

    let selectedFile = null;

    // Drop zone
    dropZone.addEventListener('click', () => fileInput.click());
    dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('dragover'); });
    dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
    dropZone.addEventListener('drop', e => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
    });
    fileInput.addEventListener('change', e => {
        if (e.target.files.length) handleFile(e.target.files[0]);
    });

    // Tab Switching
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const tabId = btn.getAttribute('data-tab');
            document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            btn.classList.add('active');
            document.getElementById(`tab-${tabId}`).classList.add('active');
        });
    });

    // Paste support (Ctrl+V)
    window.addEventListener('paste', e => {
        const items = (e.clipboardData || e.originalEvent.clipboardData).items;
        for (const item of items) {
            if (item.kind === 'file') {
                const file = item.getAsFile();
                if (file) handleFile(file);
            }
        }
    });

    // URL Fetching
    const btnFetchUrl = document.getElementById('btn-fetch-url');
    const videoUrlInput = document.getElementById('video-url');

    if (btnFetchUrl) {
        btnFetchUrl.addEventListener('click', async () => {
            const url = videoUrlInput.value.trim();
            if (!url) {
                showStatus('Por favor, cole um link válido do Instagram, TikTok ou Facebook.', 'error');
                return;
            }

            try {
                btnFetchUrl.disabled = true;
                const originalHtml = btnFetchUrl.innerHTML;
                btnFetchUrl.innerHTML = '<span class="spinner"></span> Processando...';
                
                showStatus('Iniciando extração e download da mídia remota... Isso pode levar alguns segundos.', 'info');

                const res = await fetch('/api/analyze-url', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ url })
                });

                const data = await res.json();
                if (data.error) throw new Error(data.error);

                showStatus('Download concluído! Redirecionando para seleção de frames...', 'info');
                
                // Give a small delay for the user to see the success message
                setTimeout(() => {
                    window.location.href = data.redirect;
                }, 1000);

            } catch (e) {
                showStatus(e.message, 'error');
                btnFetchUrl.disabled = false;
                btnFetchUrl.innerHTML = '<svg width="18" height="18" stroke="white" style="margin-right:8px"><use href="#icon-search"/></svg> Baixar e Analisar';
            }
        });
    }

    btnRemove.addEventListener('click', clearFile);
    btnAnalyze.addEventListener('click', () => {
        if (!selectedFile) return;

        // Skip the waiting modal purely for local development
        if (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1') {
            startAnalysis();
            return;
        }

        const adModal = document.getElementById('ad-modal');
        const countdownEl = document.getElementById('ad-countdown');
        const statusEl = document.getElementById('ad-status-text');
        const closeBtn = document.getElementById('close-ad-btn');

        adModal.classList.remove('hidden');
        adModal.classList.add('active');
        let count = 10;
        const messages = [
            "Sincronizando vetores de detecção...",
            "Carregando modelos de Deep Learning...",
            "Analisando artefatos de difusão...",
            "Validando assinaturas digitais...",
            "Quase pronto para a matriz..."
        ];

        const timer = setInterval(() => {
            count--;
            countdownEl.textContent = count;
            if (count % 3 === 0) {
                statusEl.textContent = messages[Math.floor(Math.random() * messages.length)];
            }
            if (count <= 0) {
                clearInterval(timer);
                statusEl.textContent = "Investigação Liberada!";
                closeBtn.disabled = false;
                closeBtn.innerHTML = "Ver Resultados Forenses";
                closeBtn.onclick = () => {
                    adModal.classList.remove('active');
                    adModal.classList.add('hidden');
                    startAnalysis();
                };
            }
        }, 1000);
    });

    function handleFile(file) {
        const ext = file.name.split('.').pop().toLowerCase();
        const valid = ['jpg','jpeg','png','webp','bmp','tiff','tif','mp4','avi','mov','webm','mkv'];
        if (!valid.includes(ext)) { alert('Formato não suportado: ' + ext); return; }
        if (file.size > 50*1024*1024) { alert('Arquivo muito grande (máx 50MB)'); return; }
        selectedFile = file;
        showPreview(file);
    }

    function showPreview(file) {
        const isVideo = file.type.startsWith('video') || ['mp4','avi','mov','webm','mkv'].includes(file.name.split('.').pop().toLowerCase());
        const url = URL.createObjectURL(file);
        previewMedia.innerHTML = '';
        if (isVideo) {
            const v = document.createElement('video');
            v.src = url; v.controls = true; v.muted = true;
            v.style.maxWidth = '100%'; v.style.maxHeight = '100%';
            previewMedia.appendChild(v);
        } else {
            const img = document.createElement('img');
            img.src = url; img.alt = file.name;
            previewMedia.appendChild(img);
        }
        pfName.textContent = file.name;
        pfSize.textContent = fmtSize(file.size);
        dropZone.style.display = 'none';
        preview.classList.remove('hidden');
    }

    function clearFile() {
        selectedFile = null;
        fileInput.value = '';
        preview.classList.add('hidden');
        dropZone.style.display = '';
    }

    function fmtSize(b) {
        if (b < 1024) return b + ' B';
        if (b < 1048576) return (b/1024).toFixed(1) + ' KB';
        return (b/1048576).toFixed(1) + ' MB';
    }

    async function startAnalysis() {
        if (!selectedFile) return;

        const isVideo = selectedFile.type.startsWith('video') || 
                        ['mp4','avi','mov','webm','mkv'].includes(selectedFile.name.split('.').pop().toLowerCase());

        secUpload.classList.add('hidden');
        secLoading.classList.remove('hidden');

        const steps = loadSteps.querySelectorAll('.ls');
        let prog = 0;

        const texts = isVideo ? [
            'Iniciando captura de vídeo...',
            'Decodificando frames...',
            'Gerando storyboard para seleção...',
            'Otimizando visualização...',
            'Quase pronto...',
        ] : [
            'Extraindo espectro de frequência...',
            'Analisando padrão de ruído PRNU...',
            'Calculando Error Level Analysis...',
            'Processando gradientes...',
            'Verificando distribuição estatística...',
            'Decompondo sub-bandas wavelet...',
            'Lendo metadados EXIF...',
            'Computando scores H/V...',
            'Fusionando resultados...',
        ];
        let ti = 0;

        const intProg = setInterval(() => {
            prog = Math.min(prog + Math.random()*7, 92);
            progressFill.style.width = prog + '%';
            const ai = Math.floor((prog/92)*steps.length);
            steps.forEach((s,i) => {
                s.classList.toggle('done', i < ai);
                s.classList.toggle('active', i === ai);
            });
        }, 250);

        const intText = setInterval(() => {
            loadingText.textContent = texts[ti % texts.length];
            ti++;
        }, 1200);

        try {
            const fd = new FormData();
            fd.append('file', selectedFile);
            
            // Choose endpoint based on type
            const endpoint = isVideo ? '/api/storyboard' : '/api/analyze';
            const res = await fetch(endpoint, { method:'POST', body:fd });

            clearInterval(intProg);
            clearInterval(intText);

            if (!res.ok) {
                const err = await res.json();
                throw new Error(err.error || 'Erro no processamento');
            }

            const result = await res.json();

            progressFill.style.width = '100%';
            steps.forEach(s => { s.classList.remove('active'); s.classList.add('done'); });
            loadingText.textContent = isVideo ? 'Vídeo processado! Redirecionando para seleção...' : 'Análise concluída! Redirecionando...';

            // Redirect to results or selection page
            setTimeout(() => {
                window.location.href = result.redirect;
            }, 600);

        } catch(e) {
            clearInterval(intProg);
            clearInterval(intText);
            let msg = e.message;
            if (msg.includes('fetch') || msg.includes('NetworkError')) {
                msg = 'O servidor forense está reiniciando ou está temporariamente inacessível. Aguarde alguns segundos e tente novamente.';
            }
            alert('Erro: ' + msg);
            secLoading.classList.add('hidden');
            secUpload.classList.remove('hidden');
            clearFile();
        }
    }

    function showStatus(msg, type) {
        // Find existing status elements
        const statusEl = document.getElementById('loading-text');
        if (statusEl) {
            statusEl.textContent = msg;
            statusEl.style.color = type === 'error' ? '#ef4444' : 'var(--accent-blue)';
        } else {
            // Fallback for screens without loading-text
            console.log(`[Status ${type}]: ${msg}`);
            if (type === 'error') alert(msg);
        }
    }

})();
