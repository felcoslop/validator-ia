let html5QrCode = null;

async function openScanner() {
    const modal = document.getElementById('scanner-modal');
    modal.classList.remove('hidden');
    modal.classList.add('active');
    document.body.style.overflow = 'hidden';

    // Initialize the scanner
    if (!html5QrCode) {
        html5QrCode = new Html5Qrcode("reader");
    }

    // Check for Secure Context (HTTPS/Localhost) - Required for camera access
    if (!window.isSecureContext) {
        alert("Erro de Segurança: O acesso à câmera requer uma conexão segura (HTTPS) ou localhost. Se você está acessando via IP em outro dispositivo, a câmera será bloqueada pelo navegador.");
        closeScanner();
        return;
    }

    const config = { 
        fps: 10, 
        qrbox: { width: 250, height: 250 },
        aspectRatio: 1.0
    };

    try {
        await html5QrCode.start(
            { facingMode: "environment" }, 
            config, 
            (decodedText) => {
                // Success callback
                if (decodedText.startsWith('http')) {
                    html5QrCode.stop().then(() => {
                        window.location.href = decodedText;
                    });
                }
            }
        );
    } catch (err) {
        console.warn("Câmera traseira não encontrada, tentando qualquer câmera:", err);
        try {
            await html5QrCode.start(
                { facingMode: "user" }, 
                config, 
                (decodedText) => {
                    if (decodedText.startsWith('http')) {
                        html5QrCode.stop().then(() => {
                            window.location.href = decodedText;
                        });
                    }
                }
            );
        } catch (err2) {
            console.error("Erro final ao acessar câmera:", err2);
            alert("Não foi possível acessar a câmera.\n\nPossíveis causas:\n1. Permissão negada no navegador.\n2. Conexão não segura (HTTP - use HTTPS).\n3. Câmera sendo usada por outro app.");
            closeScanner();
        }
    }
}

function closeScanner() {
    if (html5QrCode && html5QrCode.isScanning) {
        html5QrCode.stop().then(() => {
            document.getElementById('scanner-modal').classList.remove('active');
            document.body.style.overflow = '';
        }).catch(err => {
            console.error("Erro ao parar scanner:", err);
            document.getElementById('scanner-modal').classList.remove('active');
            document.body.style.overflow = '';
        });
    } else {
        document.getElementById('scanner-modal').classList.remove('active');
        document.getElementById('scanner-modal').classList.add('hidden');
        document.body.style.overflow = '';
    }
}
