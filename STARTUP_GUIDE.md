# 🧠 Parkinson's Detection System — Startup Guide

## Quick Start (TL;DR)

```bash
cd "c:\Users\ayush\Downloads\fianl mera project"
python backend/app.py
```
Then open **http://localhost:5000** in your browser.

---

## Prerequisites

| Requirement | Details |
|---|---|
| **Python** | 3.8 or higher |
| **OS** | Windows 10/11 |
| **Browser** | Chrome or Edge (recommended for camera/mic access) |
| **Hardware** | Webcam + Microphone (for assessments) |

---

## Step-by-Step Guide

### 1. Open a Terminal

- Press `Win + R`, type `cmd` or `powershell`, and press Enter.
- Or open **Windows Terminal** from the Start menu.

### 2. Navigate to the Project Folder

```bash
cd "c:\Users\ayush\Downloads\fianl mera project"
```

### 3. (Optional) Verify Dependencies Are Installed

If you're unsure whether all packages are installed, run:

```bash
pip install -r requirements.txt
```

> [!NOTE]
> The project requires: Flask, NumPy, Pandas, Librosa, SciPy, SoundFile, Praat-Parselmouth, MediaPipe, OpenCV, Scikit-learn, Scikit-image, PyTorch, Torchvision, Joblib, and Tqdm.

### 4. Start the Application

**Option A — Development Server (recommended for local use):**

```bash
python backend/app.py
```

This starts Flask in debug mode with auto-reload. You'll see output like:

```
Voice model (SVM) loaded successfully.
Handwriting CNN model loaded successfully.
Starting Flask application...
 * Running on http://127.0.0.1:5000
```

**Option B — Production Server (for network/demo use):**

```bash
pip install waitress   # only needed once
python serve.py
```

This uses Waitress (a production WSGI server) and makes the app accessible to other devices on the same network.

### 5. Open the Application

Open your browser and go to:

| URL | Page |
|---|---|
| **http://localhost:5000** | 🏠 Homepage |
| **http://localhost:5000/assessment** | 🔬 Assessment (Voice, Handwriting, Eye Tracking) |
| **http://localhost:5000/dashboard** | 📊 Dashboard (History & Trends) |

### 6. Stop the Application

Press `Ctrl + C` in the terminal where the server is running.

---

## Application Pages

### 🏠 Homepage
The landing page with a futuristic UI. Enter your Medical ID or Email to get started.

### 🔬 Assessment Page
A 3-step multimodal diagnostic tool:
1. **Voice Analysis** — Record a 30-second voice sample (say "aaa" or read text aloud)
2. **Handwriting Analysis** — Upload a spiral/wave drawing image
3. **Eye Tracking** — Follow on-screen targets using your webcam

### 📊 Dashboard
View historical assessment results with trend charts showing confidence scores over time.

---

## Available Models

| Model | File | Status |
|---|---|---|
| Voice (SVM) | `models/voice_svm_model.pkl` | ✅ Active |
| Voice Scaler | `models/voice_scaler.pkl` | ✅ Active |
| Handwriting (CNN) | `models/handwriting_cnn_model.pth` | ✅ Active |
| Eye Tracking | Heuristic-based | ✅ Built-in |

> [!TIP]
> If you train a new XGBoost voice model, save it as `models/voice_xgb_model.pkl` and the app will automatically prefer it over the SVM model.

---

## Troubleshooting

| Problem | Solution |
|---|---|
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` |
| Port 5000 already in use | Close other apps using that port, or edit `app.py` to use a different port |
| Camera/Mic not working | Use **http://localhost:5000** (not 127.0.0.1) and allow browser permissions |
| Models not loading | Check that the `models/` directory contains the `.pkl` and `.pth` files |

---

## Project Structure

```
fianl mera project/
├── backend/
│   ├── app.py              ← Main Flask application
│   └── database.py         ← SQLite database manager
├── frontend/
│   ├── templates/          ← HTML pages (index, assessment, dashboard)
│   └── static/             ← CSS, JS, and video assets
├── voice_engine/           ← Voice preprocessing & feature extraction
├── handwriting_engine/     ← Handwriting image processing & CNN model
├── eye_tracking/           ← Eye movement analysis
├── fusion_engine/          ← Multimodal result fusion
├── models/                 ← Trained ML model files
├── serve.py                ← Production server (Waitress)
├── requirements.txt        ← Python dependencies
└── STARTUP_GUIDE.md        ← This file
```
