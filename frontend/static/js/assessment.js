// =========================================
// WAV Conversion Utility
// Browsers record as WebM/Opus. This converts
// to proper PCM WAV so the Python backend can
// read it with librosa/soundfile (no ffmpeg).
// =========================================
async function convertBlobToWav(blob) {
    const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    const arrayBuffer = await blob.arrayBuffer();
    const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);
    audioCtx.close();

    // Encode as 16-bit PCM WAV (mono, original sample rate)
    const numChannels = 1;
    const sampleRate = audioBuffer.sampleRate;
    // Mix down to mono
    let samples;
    if (audioBuffer.numberOfChannels === 1) {
        samples = audioBuffer.getChannelData(0);
    } else {
        const left = audioBuffer.getChannelData(0);
        const right = audioBuffer.getChannelData(1);
        samples = new Float32Array(left.length);
        for (let i = 0; i < left.length; i++) {
            samples[i] = (left[i] + right[i]) / 2;
        }
    }

    const bytesPerSample = 2; // 16-bit
    const blockAlign = numChannels * bytesPerSample;
    const dataSize = samples.length * blockAlign;
    const buffer = new ArrayBuffer(44 + dataSize);
    const view = new DataView(buffer);

    // WAV header
    function writeString(offset, str) {
        for (let i = 0; i < str.length; i++) view.setUint8(offset + i, str.charCodeAt(i));
    }
    writeString(0, 'RIFF');
    view.setUint32(4, 36 + dataSize, true);
    writeString(8, 'WAVE');
    writeString(12, 'fmt ');
    view.setUint32(16, 16, true); // fmt chunk size
    view.setUint16(20, 1, true);  // PCM
    view.setUint16(22, numChannels, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * blockAlign, true);
    view.setUint16(32, blockAlign, true);
    view.setUint16(34, 16, true); // bits per sample
    writeString(36, 'data');
    view.setUint32(40, dataSize, true);

    // PCM samples (clamp to [-1, 1] then scale to int16)
    let offset = 44;
    for (let i = 0; i < samples.length; i++) {
        const s = Math.max(-1, Math.min(1, samples[i]));
        view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
        offset += 2;
    }

    return new Blob([buffer], { type: 'audio/wav' });
}

// =========================================
// Pre-loading Guide Overlays
// =========================================
const voiceGuideOverlay = document.getElementById('voiceGuideOverlay');
const eyeGuideOverlay = document.getElementById('eyeGuideOverlay');

// Voice guide: dismiss on click
if (voiceGuideOverlay) {
    voiceGuideOverlay.addEventListener('click', () => {
        voiceGuideOverlay.classList.add('fade-out');
        setTimeout(() => {
            voiceGuideOverlay.style.display = 'none';
        }, 450);
    });
}

// Eye guide: dismiss on click
if (eyeGuideOverlay) {
    eyeGuideOverlay.addEventListener('click', () => {
        eyeGuideOverlay.classList.add('fade-out');
        setTimeout(() => {
            eyeGuideOverlay.style.display = 'none';
        }, 450);
    });
}

// =========================================
// Story Reader Lines (30 seconds, ~10 lines)
// =========================================
const storyLines = [
    "The morning sun cast long shadows across the quiet garden.",
    "A small bird perched on the wooden fence, watching carefully.",
    "The old man walked slowly down the path, his shoes crunching softly.",
    "He paused to admire the roses that had bloomed overnight.",
    "A gentle breeze carried the scent of jasmine through the air.",
    "Children laughed in the distance, their voices clear and bright.",
    "The clock tower struck nine, echoing across the empty square.",
    "She placed the warm cup of tea on the table by the window.",
    "Rain began to drizzle, tapping softly on the glass pane above.",
    "The story ended as the last drop of sunlight faded away."
];

let storyInterval = null;
let storyLineIndex = 0;

function startStoryReader() {
    const panel = document.getElementById('storyReaderPanel');
    const idleText = document.getElementById('storyIdleText');
    if (!panel) return;

    // Remove idle text
    if (idleText) idleText.style.display = 'none';

    storyLineIndex = 0;
    showStoryLine();

    storyInterval = setInterval(() => {
        storyLineIndex++;
        if (storyLineIndex < storyLines.length) {
            showStoryLine();
        } else {
            clearInterval(storyInterval);
        }
    }, 3000); // 3 seconds per line = 30s for 10 lines
}

function showStoryLine() {
    const panel = document.getElementById('storyReaderPanel');
    if (!panel) return;

    // Remove any old story-line elements
    const old = panel.querySelectorAll('.story-line');
    old.forEach(el => el.remove());

    const lineEl = document.createElement('div');
    lineEl.className = 'story-line';
    lineEl.textContent = storyLines[storyLineIndex];
    lineEl.style.animation = 'none'; // reset
    panel.appendChild(lineEl);

    // Trigger reflow then apply animation
    void lineEl.offsetWidth;
    lineEl.style.animation = 'storyLineAppear 2.8s ease forwards';
}

function stopStoryReader() {
    if (storyInterval) {
        clearInterval(storyInterval);
        storyInterval = null;
    }
    const panel = document.getElementById('storyReaderPanel');
    if (panel) {
        const old = panel.querySelectorAll('.story-line');
        old.forEach(el => el.remove());
    }
}

// =========================================
// Core Assessment State
// =========================================
let currentStep = 1;
let voiceBlob = null;
let eyeSessionData = [];
let handwritingFile = null;
let voiceProb = null;
let eyeProb = null;
let hwProb = null;

// Progress bar
const progressBar = document.getElementById('progress');

// Secure context check
if (window.isSecureContext === false && window.location.hostname !== 'localhost') {
    document.getElementById('secureContextWarning').style.display = 'block';
}

// Step management
function showStep(step) {
    currentStep = step;
    document.querySelectorAll('.step').forEach(s => s.classList.remove('active'));
    document.getElementById(`step${step}`).classList.add('active');
    progressBar.style.width = `${(step / 3) * 100}%`;

    // Show eye guide overlay when transitioning to step 2
    if (step === 2 && eyeGuideOverlay) {
        eyeGuideOverlay.style.display = 'flex';
        eyeGuideOverlay.classList.remove('fade-out');
        eyeGuideOverlay.style.opacity = '0';
        eyeGuideOverlay.style.animation = 'guideFadeIn 0.5s ease forwards';
        // Re-trigger video playback
        const vid = eyeGuideOverlay.querySelector('video');
        if (vid) vid.play().catch(() => {});
    }
}

function getAvailableProbabilities() {
    const payload = {};

    if (typeof voiceProb === 'number' && !Number.isNaN(voiceProb)) {
        payload.voice = voiceProb;
    }
    if (typeof eyeProb === 'number' && !Number.isNaN(eyeProb)) {
        payload.eye = eyeProb;
    }
    if (typeof hwProb === 'number' && !Number.isNaN(hwProb)) {
        payload.handwriting = hwProb;
    }

    return payload;
}

// Step 1: Voice Recording
const recordBtn = document.getElementById('recordBtn');
const stopRecordBtn = document.getElementById('stopRecordBtn');
const skipVoiceBtn = document.getElementById('skipVoiceBtn');
if (skipVoiceBtn) skipVoiceBtn.onclick = () => showStep(2);
const simulateVoiceBtn = document.getElementById('simulateVoiceBtn');
const voiceStatus = document.getElementById('voiceStatus');
const voiceTimer = document.getElementById('voiceTimer');
const canvas = document.getElementById('waveform');
const ctx = canvas.getContext('2d');
let mediaRecorder;
let audioChunks = [];
let recordingTimeout;
let countdownInterval;

simulateVoiceBtn.onclick = () => {
    voiceStatus.style.display = 'block';
    voiceStatus.innerText = "Simulating audio processing...";
    setTimeout(() => {
        voiceProb = 0.35; // Example score
        showStep(2);
    }, 1500);
};

recordBtn.onclick = async () => {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder = new MediaRecorder(stream);
        audioChunks = [];
        
        mediaRecorder.ondataavailable = (event) => {
            audioChunks.push(event.data);
        };
        
        mediaRecorder.onstop = async () => {
            clearInterval(countdownInterval);
            clearTimeout(recordingTimeout);
            voiceTimer.style.display = 'none';
            voiceStatus.style.display = 'block';
            voiceStatus.innerText = "Converting audio to WAV...";
            voiceStatus.style.color = "#64748b";
            stopRecordBtn.style.display = 'none';
            stopStoryReader();
            
            // Browser records as WebM/Opus — convert to real PCM WAV
            const rawBlob = new Blob(audioChunks, { type: mediaRecorder.mimeType || 'audio/webm' });
            let wavBlob;
            try {
                wavBlob = await convertBlobToWav(rawBlob);
            } catch (convErr) {
                console.error('WAV conversion failed:', convErr);
                // Fallback: send raw blob and hope backend can handle it
                wavBlob = rawBlob;
            }
            voiceBlob = wavBlob;

            // Upload voice
            voiceStatus.innerText = "Uploading & analyzing voice...";
            const formData = new FormData();
            formData.append('audio', wavBlob, 'voice.wav');
            
            try {
                const response = await fetch('/api/voice_assessment', {
                    method: 'POST',
                    body: formData
                });
                
                if (!response.ok) {
                    throw new Error(`Server returned ${response.status}`);
                }
                
                const data = await response.json();
                voiceProb = data.probability;
                voiceStatus.innerText = "Success! Moving to Step 2...";
                setTimeout(() => showStep(2), 1000);
            } catch (err) {
                console.error('Voice upload failed:', err);
                voiceStatus.innerText = "Voice processing failed. You can still continue to the next step.";
                voiceStatus.style.color = "#ef4444";
                skipVoiceBtn.style.display = 'inline-block';
            }
        };
        
        mediaRecorder.start();
        recordBtn.style.display = 'none';
        stopRecordBtn.style.display = 'inline-block';
        voiceTimer.style.display = 'block';
        voiceStatus.style.display = 'none';

        // Start the story reader
        startStoryReader();
        
        // Automatic stop after 30 seconds
        let timeLeft = 30;
        voiceTimer.innerText = `${timeLeft}s`;
        
        countdownInterval = setInterval(() => {
            timeLeft--;
            if (voiceTimer) voiceTimer.innerText = `${timeLeft}s`;
            if (timeLeft <= 0) {
                clearInterval(countdownInterval);
            }
        }, 1000);
        
        recordingTimeout = setTimeout(() => {
            if (mediaRecorder.state === 'recording') {
                mediaRecorder.stop();
            }
        }, 30000);
        
        // Simple visualizer
        const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        const source = audioCtx.createMediaStreamSource(stream);
        const analyzer = audioCtx.createAnalyser();
        source.connect(analyzer);
        
        function draw() {
            if (mediaRecorder.state === 'recording') {
                requestAnimationFrame(draw);
                const bufferLength = analyzer.frequencyBinCount;
                const dataArray = new Uint8Array(bufferLength);
                analyzer.getByteTimeDomainData(dataArray);
                
                ctx.fillStyle = '#ffffff';
                ctx.fillRect(0, 0, canvas.width, canvas.height);
                ctx.lineWidth = 2;
                ctx.strokeStyle = '#288080';
                ctx.beginPath();
                
                const sliceWidth = canvas.width * 1.0 / bufferLength;
                let x = 0;
                
                for (let i = 0; i < bufferLength; i++) {
                    const v = dataArray[i] / 128.0;
                    const y = v * canvas.height / 2;
                    if (i === 0) ctx.moveTo(x, y);
                    else ctx.lineTo(x, y);
                    x += sliceWidth;
                }
                ctx.lineTo(canvas.width, canvas.height / 2);
                ctx.stroke();
            }
        }
        draw();
    } catch (err) {
        console.error('Microphone access denied or failed:', err);
        let errorMsg = `Could not access microphone: ${err.message || 'Unknown error'}`;
        
        if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
            errorMsg = 'Microphone permission denied. Please allow access in your browser settings (click the lock icon in the address bar).';
        } else if (err.name === 'NotFoundError' || err.name === 'DevicesNotFoundError') {
            errorMsg = 'No microphone found. Please connect a microphone and try again.';
        } else if (err.name === 'NotReadableError' || err.name === 'TrackStartError') {
            errorMsg = 'Microphone is already in use by another application or blocked at the OS level.';
        } else if (window.isSecureContext === false) {
            errorMsg = 'Microphone access requires a secure connection. Please use http://localhost:5000 exactly (not 127.0.0.1 or an IP).';
        }
        
        if (window.location.hostname === '127.0.0.1') {
            errorMsg += ' TIP: Try using http://localhost:5000 instead of 127.0.0.1, as some browsers treat 127.0.0.1 as insecure.';
        }
        
        alert(errorMsg);
        voiceStatus.innerText = errorMsg;
        voiceStatus.style.color = "#ef4444";
        voiceStatus.style.display = 'block';
        skipVoiceBtn.style.display = 'inline-block';
        simulateVoiceBtn.style.display = 'inline-block';
        recordBtn.style.display = 'inline-block';
        stopRecordBtn.style.display = 'none';
    }
};

stopRecordBtn.onclick = () => {
    mediaRecorder.stop();
};

// Step 2: Eye Tracking (Structured Task Protocol)
const startEyeBtn = document.getElementById('startEyeBtn');
const skipEyeBtn = document.getElementById('skipEyeBtn');
const eyeStatus = document.getElementById('eyeStatus');
const videoElement = document.getElementById('inputVideo');
const target = document.getElementById('target');
const outputCanvas = document.getElementById('outputCanvas');
const outCtx = outputCanvas.getContext('2d');
const taskName = document.getElementById('taskName');
const taskInstruction = document.getElementById('taskInstruction');
const eyeTimer = document.getElementById('eyeTimer');

const eyeProtocol = [
    { name: "Fixation Baseline", duration: 15, instruction: "Stare at the center white dot. Do not move your eyes.", type: "fixation" },
    { name: "Prosaccade", duration: 20, instruction: "Follow the red moving dot as quickly as possible.", type: "prosaccade" },
    { name: "Smooth Pursuit", duration: 30, instruction: "Follow the slow moving dot smoothly.", type: "pursuit" },
    { name: "Blink Task", duration: 10, instruction: "Blink naturally while looking at the center dot.", type: "blink" }
];

skipEyeBtn.onclick = () => showStep(3);

startEyeBtn.onclick = async () => {
    eyeSessionData = [];
    startEyeBtn.style.display = 'none';
    skipEyeBtn.style.display = 'none';
    eyeStatus.style.display = 'none';
    let currentTaskName = "";
    
    const faceMesh = new FaceMesh({
        locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`
    });
    
    faceMesh.setOptions({
        maxNumFaces: 1,
        refineLandmarks: true,
        minDetectionConfidence: 0.5,
        minTrackingConfidence: 0.5
    });
    
    faceMesh.onResults((results) => {
        if (results.multiFaceLandmarks) {
            const landmarks = results.multiFaceLandmarks[0];
            const leftIris = landmarks[468];
            const rightIris = landmarks[473];
            
            eyeSessionData.push({
                timestamp: Date.now() / 1000,
                gaze_x: (leftIris.x + rightIris.x) / 2.0,
                gaze_y: (leftIris.y + rightIris.y) / 2.0,
                target_x: target.style.display !== 'none' ? (parseFloat(target.style.left) / 800) : 0.5,
                target_y: target.style.display !== 'none' ? (parseFloat(target.style.top) / 400) : 0.5,
                ear: calculateEAR(landmarks),
                task: currentTaskName
            });
        }
    });
    
    const camera = new Camera(videoElement, {
        onFrame: async () => { await faceMesh.send({ image: videoElement }); },
        width: 640, height: 480
    });
    await camera.start();

    // Run Protocol
    for (const taskConfig of eyeProtocol) {
        currentTaskName = taskConfig.name;
        taskName.innerText = taskConfig.name;
        taskInstruction.innerText = taskConfig.instruction;
        target.style.display = 'block';
        
        let startTime = Date.now();
        const durationMs = taskConfig.duration * 1000;
        
        await new Promise(resolve => {
            function update() {
                const elapsed = Date.now() - startTime;
                const remaining = Math.max(0, Math.ceil((durationMs - elapsed) / 1000));
                eyeTimer.innerText = `${remaining}s`;
                
                // Animate target based on task type
                updateTargetPosition(taskConfig.type, elapsed, durationMs);
                
                if (elapsed < durationMs) {
                    requestAnimationFrame(update);
                } else {
                    resolve();
                }
            }
            update();
        });
    }
    
    target.style.display = 'none';
    // Stop camera
    if (videoElement.srcObject) {
        videoElement.srcObject.getTracks().forEach(track => track.stop());
    }
    finishEyeTracking();
};

function calculateEAR(landmarks) {
    // Indices for eyes in MediaPipe Face Mesh
    const leftEye = [362, 385, 387, 263, 373, 380];
    const rightEye = [33, 160, 158, 133, 153, 144];
    
    function getDist(p1, p2) {
        return Math.sqrt(Math.pow(landmarks[p1].x - landmarks[p2].x, 2) + Math.pow(landmarks[p1].y - landmarks[p2].y, 2));
    }
    
    const leftEAR = (getDist(leftEye[1], leftEye[5]) + getDist(leftEye[2], leftEye[4])) / (2 * getDist(leftEye[0], leftEye[3]));
    const rightEAR = (getDist(rightEye[1], rightEye[5]) + getDist(rightEye[2], rightEye[4])) / (2 * getDist(rightEye[0], rightEye[3]));
    return (leftEAR + rightEAR) / 2;
}

function updateTargetPosition(type, elapsed, total) {
    let x = 400, y = 200; // Center
    
    if (type === "prosaccade") {
        // Random jumps every 2 seconds
        const jumpInterval = 2000;
        const seed = Math.floor(elapsed / jumpInterval);
        const random = (s) => Math.sin(s * 12345.67) * 0.5 + 0.5;
        x = 100 + random(seed) * 600;
        y = 50 + random(seed + 1) * 300;
    } else if (type === "pursuit") {
        // Smooth horizontal motion
        x = 400 + 300 * Math.sin((elapsed / 3000) * 2 * Math.PI);
        y = 200;
    }
    
    target.style.left = `${x}px`;
    target.style.top = `${y}px`;
}

async function finishEyeTracking() {
    eyeStatus.style.display = 'block';
    eyeStatus.innerText = "Processing eye data...";
    
    try {
        const response = await fetch('/api/eye_assessment', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ session_data: eyeSessionData })
        });
        
        if (!response.ok) {
            throw new Error(`Server returned ${response.status}`);
        }
        
        const data = await response.json();
        eyeProb = data.probability;
        eyeStatus.innerText = "Success! Moving to Step 3...";
        setTimeout(() => showStep(3), 1000);
    } catch (err) {
        console.error('Eye assessment upload failed:', err);
        eyeStatus.innerText = "Assessment processing failed. You can still continue.";
        eyeStatus.style.color = "#ef4444";
        skipEyeBtn.style.display = 'inline-block';
    }
}

// Step 3: Handwriting
const spiralUpload = document.getElementById('spiralUpload');
const preview = document.getElementById('spiralPreview');
const previewContainer = document.getElementById('previewContainer');
const analyzeBtn = document.getElementById('analyzeBtn');
const skipHwBtn = document.getElementById('skipHwBtn');
const hwStatus = document.getElementById('hwStatus');

skipHwBtn.onclick = () => showFinalResults();

spiralUpload.onchange = (e) => {
    handwritingFile = e.target.files[0];
    if (handwritingFile) {
        const reader = new FileReader();
        reader.onload = (event) => {
            preview.src = event.target.result;
            previewContainer.style.display = 'block';
        };
        reader.readAsDataURL(handwritingFile);
    }
};

analyzeBtn.onclick = async () => {
    if (!handwritingFile) return alert('Please upload a spiral image.');
    
    hwStatus.style.display = 'block';
    hwStatus.innerText = "Analyzing handwriting...";
    analyzeBtn.style.display = 'none';
    
    const formData = new FormData();
    formData.append('image', handwritingFile);
    
    try {
        const response = await fetch('/api/handwriting_assessment', {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error(`Server returned ${response.status}`);
        }
        
        const data = await response.json();
        hwProb = data.probability;
        hwStatus.innerText = "Analysis complete! Finalizing results...";
        setTimeout(() => showFinalResults(), 1000);
    } catch (err) {
        console.error('Handwriting analysis failed:', err);
        hwStatus.innerText = "Handwriting analysis failed. You can still see final results.";
        hwStatus.style.color = "#ef4444";
        skipHwBtn.style.display = 'inline-block';
    }
};

async function showFinalResults() {
    const payload = getAvailableProbabilities();
    if (Object.keys(payload).length === 0) {
        alert('Please complete at least one assessment before viewing results.');
        return;
    }

    // Fuse all results
    let data;
    try {
        const response = await fetch('/api/fuse_results', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        data = await response.json();
        if (!response.ok) {
            throw new Error(data.error || `Server returned ${response.status}`);
        }
    } catch (err) {
        alert(`Unable to generate final results: ${err.message}`);
        return;
    }
    
    // Display results
    document.querySelectorAll('.step').forEach(s => s.classList.remove('active'));
    document.getElementById('resultsStep').classList.add('active');
    progressBar.style.width = '100%';
    
    document.getElementById('finalVerdict').innerText = data.verdict;
    document.getElementById('finalProbText').innerText = `Confidence Score: ${(data.final_probability * 100).toFixed(1)}%`;
    
    document.getElementById('voiceBar').style.width = `${((voiceProb ?? 0) * 100)}%`;
    document.getElementById('voiceScore').innerText = voiceProb == null ? 'Skipped' : `${(voiceProb * 100).toFixed(0)}%`;
    
    document.getElementById('eyeBar').style.width = `${((eyeProb ?? 0) * 100)}%`;
    document.getElementById('eyeScore').innerText = eyeProb == null ? 'Skipped' : `${(eyeProb * 100).toFixed(0)}%`;
    
    document.getElementById('hwBar').style.width = `${((hwProb ?? 0) * 100)}%`;
    document.getElementById('hwScore').innerText = hwProb == null ? 'Skipped' : `${(hwProb * 100).toFixed(0)}%`;
}
