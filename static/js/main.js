document.addEventListener('DOMContentLoaded', function() {
    const voiceInput = document.getElementById('voice-input');
    const handwritingInput = document.getElementById('handwriting-input');
    const voiceFileName = document.getElementById('voice-file-name');
    const handwritingFileName = document.getElementById('handwriting-file-name');
    const analyzeBtn = document.getElementById('analyze-btn');
    const recordBtn = document.getElementById('record-btn');
    
    let isRecording = false;
    let voiceFile = null;
    let handwritingFile = null;

    // File upload handling
    function handleFileUpload(file, type) {
        if (type === 'voice') {
            voiceFile = file;
            voiceFileName.textContent = file.name;
        } else {
            handwritingFile = file;
            handwritingFileName.textContent = file.name;
        }
        updateAnalyzeButton();
    }

    // Drag and drop handling
    ['voice-drop-area', 'handwriting-drop-area'].forEach(id => {
        const dropArea = document.getElementById(id);
        const type = id.split('-')[0];

        dropArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropArea.classList.add('dragover');
        });

        dropArea.addEventListener('dragleave', () => {
            dropArea.classList.remove('dragover');
        });

        dropArea.addEventListener('drop', (e) => {
            e.preventDefault();
            dropArea.classList.remove('dragover');
            const file = e.dataTransfer.files[0];
            handleFileUpload(file, type);
        });

        dropArea.addEventListener('click', () => {
            const input = document.getElementById(`${type}-input`);
            input.click();
        });
    });

    voiceInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFileUpload(e.target.files[0], 'voice');
        }
    });

    handwritingInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFileUpload(e.target.files[0], 'handwriting');
        }
    });

    // Record functionality
    recordBtn.addEventListener('click', async () => {
        if (isRecording) return;
    
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            const mediaRecorder = new MediaRecorder(stream, {
                mimeType: 'audio/webm' // Change to WebM format
            });
            const chunks = [];
    
            mediaRecorder.ondataavailable = (e) => chunks.push(e.data);
            mediaRecorder.onstop = async () => {
                // Create WebM blob
                const webmBlob = new Blob(chunks, { type: 'audio/webm' });
                
                // Convert WebM to WAV using FFmpeg.wasm or audio context
                const audioContext = new (window.AudioContext || window.webkitAudioContext)();
                const audioBuffer = await audioContext.decodeAudioData(await webmBlob.arrayBuffer());
                
                // Create WAV file
                const wavBlob = await convertToWav(audioBuffer);
                const file = new File([wavBlob], 'recorded_audio.wav', { type: 'audio/wav' });
                handleFileUpload(file, 'voice');
                isRecording = false;
                recordBtn.textContent = 'Record Voice (5s)';
            };
    
            mediaRecorder.start();
            isRecording = true;
            recordBtn.textContent = 'Recording...';
    
            setTimeout(() => {
                mediaRecorder.stop();
                stream.getTracks().forEach(track => track.stop());
            }, 5000);
        } catch (err) {
            console.error('Error accessing microphone:', err);
            alert('Unable to access microphone. Please ensure you have granted permission.');
        }
    });

    // Analysis handling
    analyzeBtn.addEventListener('click', async () => {
        const formData = new FormData();
        if (voiceFile) formData.append('voice', voiceFile);
        if (handwritingFile) formData.append('handwriting', handwritingFile);

        try {
            analyzeBtn.disabled = true;
            analyzeBtn.textContent = 'Analyzing...';

            const response = await fetch('/analyze', {
                method: 'POST',
                body: formData
            });

            const results = await response.json();
            displayResults(results);
        } catch (err) {
            console.error('Analysis error:', err);
            alert('An error occurred during analysis. Please try again.');
        } finally {
            analyzeBtn.disabled = false;
            analyzeBtn.textContent = 'Analyze';
        }
    });

    function displayResults(results) {
        const resultsSection = document.getElementById('results');
        resultsSection.innerHTML = '';

        if (results.combined) {
            addResultBox('combined', results.combined);
        }
        if (results.voice_analysis) {
            addResultBox('voice', results.voice_analysis);
        }
        if (results.handwriting_analysis) {
            addResultBox('handwriting', results.handwriting_analysis);
        }

        resultsSection.scrollIntoView({ behavior: 'smooth' });
    }

    function addResultBox(type, data) {
        const box = document.createElement('div');
        box.className = 'result-box';
        
        const title = type.charAt(0).toUpperCase() + type.slice(1);
        let content = `<h3>${title} Analysis</h3>`;
        
        if (type === 'combined') {
            content += `
                <div class="prediction ${data.prediction === 'Yes' ? 'positive' : 'negative'}">
                    Prediction: ${data.prediction}
                </div>
                <div class="confidence">
                    Confidence: ${(data.confidence * 100).toFixed(2)}%
                </div>
            `;
        } else if (type === 'voice') {
            content += `
                <div class="prediction ${data.prediction === 'Yes' ? 'positive' : 'negative'}">
                    Prediction: ${data.prediction}
                </div>
                <div class="risk-factors">
                    Risk Factors: ${data.risk_factors}/4
                    <ul>
                        ${data.risk_details.map(detail => `<li>${detail}</li>`).join('')}
                    </ul>
                </div>
            `;
        } else {
            content += `
                <div class="prediction ${data.prediction === 'Yes' ? 'positive' : 'negative'}">
                    Prediction: ${data.prediction}
                </div>
                <div class="confidence">
                    Confidence: ${(data.confidence * 100).toFixed(2)}%
                </div>
            `;
        }
        
        box.innerHTML = content;
        document.getElementById('results').appendChild(box);
    }

    function updateAnalyzeButton() {
        analyzeBtn.disabled = !voiceFile && !handwritingFile;
    }
    // Add WAV conversion function
    function convertToWav(audioBuffer) {
        const numOfChannels = audioBuffer.numberOfChannels;
        const length = audioBuffer.length * numOfChannels * 2;
        const buffer = new ArrayBuffer(44 + length);
        const view = new DataView(buffer);
        
        // Write WAV header
        writeUTFBytes(view, 0, 'RIFF');
        view.setUint32(4, 36 + length, true);
        writeUTFBytes(view, 8, 'WAVE');
        writeUTFBytes(view, 12, 'fmt ');
        view.setUint32(16, 16, true);
        view.setUint16(20, 1, true);
        view.setUint16(22, numOfChannels, true);
        view.setUint32(24, audioBuffer.sampleRate, true);
        view.setUint32(28, audioBuffer.sampleRate * numOfChannels * 2, true);
        view.setUint16(32, numOfChannels * 2, true);
        view.setUint16(34, 16, true);
        writeUTFBytes(view, 36, 'data');
        view.setUint32(40, length, true);

        // Write audio data
        const channelData = [];
        for (let channel = 0; channel < numOfChannels; channel++) {
            channelData.push(audioBuffer.getChannelData(channel));
        }

        let offset = 44;
        for (let i = 0; i < audioBuffer.length; i++) {
            for (let channel = 0; channel < numOfChannels; channel++) {
                const sample = Math.max(-1, Math.min(1, channelData[channel][i]));
                view.setInt16(offset, sample * 0x7FFF, true);
                offset += 2;
            }
        }

        return new Blob([buffer], { type: 'audio/wav' });
    }

    function writeUTFBytes(view, offset, string) {
        for (let i = 0; i < string.length; i++) {
            view.setUint8(offset + i, string.charCodeAt(i));
        }
    }

});