
<div align="center">

# 🤟 Sign Language to Speech
### Real-Time Indian Sign Language (ISL) Recognition System

A real-time **ISL → Speech** pipeline combining **YOLOv11**, **MediaPipe Holistic**, and a custom **multi-branch CNN + BiLSTM** model to translate sign language gestures into spoken words via a browser-based web app.

</div>

---

## 📌 Table of Contents

1. [Core Idea](#-core-idea)
2. [Proposed Architecture](#-proposed-architecture)
3. [Dataset Strategy](#-dataset-strategy)
4. [Model Components](#-model-components)
5. [Implementation Roadmap](#-implementation-roadmap)
6. [Project Structure](#-project-structure)
7. [Training Pipeline](#-training-pipeline)
8. [Web App Inference Pipeline](#-web-app-inference-pipeline)
9. [How to Run](#-how-to-run)
10. [Controls](#-controls)
11. [Supported Signs](#-supported-signs-20-classes)

---

## 💡 Core Idea

Traditional sign language recognition systems either process raw RGB video (computationally heavy) or rely on glove-based hardware (non-accessible). This project takes a **landmark-first approach**:

> Instead of learning from raw pixels, we extract structured body, face, and hand landmarks using MediaPipe — then model the **temporal dynamics** of those landmarks across a fixed sequence of frames using a deep learning model.

**Key design decisions:**
- **Landmark-based** input → lightweight, background-invariant, generalizable
- **Multi-branch architecture** → each body part (pose, face, left hand, right hand) gets its own specialized encoder
- **Binary masking** → handles frames where a landmark type is partially or fully absent
- **YOLOv11 pre-cropping** → zooms into the signer during training to improve landmark extraction accuracy
- **WebSocket streaming** → enables real-time, low-latency inference from browser to backend

---

## 🏗️ Proposed Architecture

The system is split into two stages: an offline **Training Pipeline** and a live **Web App Inference Pipeline**.

```
                        ┌─────────────────────────────────────────────────────┐
                        │                 TRAINING STAGE                      │
                        │                                                     │
   Raw Videos ──► Frame Extraction ──► YOLOv11 Crop ──► MediaPipe Holistic   │
                                                              │                │
                              ┌───────────┬────────────┬─────┴──────┐         │
                              ▼           ▼            ▼            ▼         │
                           Pose(33,4) Face(468,3) LHand(21,3) RHand(21,3)    │
                              │           │            │            │          │
                        Binary Masks (1.0 if present, 0.0 if missing)         │
                              │                                                │
                        CNN + BiLSTM Branch (×4, one per body part)           │
                              │                                                │
                        Feature Concat → FC Classifier → Sign Label           │
                        └─────────────────────────────────────────────────────┘

                        ┌─────────────────────────────────────────────────────┐
                        │               INFERENCE STAGE (Web App)             │
                        │                                                     │
   Browser Webcam ──► WebSocket ──► FastAPI Backend ──► MediaPipe Holistic    │
                                                              │                │
                        Rolling Buffer (deque, 64 frames)                     │
                              │                                                │
                        CNN + BiLSTM Model ──► Softmax ──► Predicted Word     │
                              │                                                │
                        WebSocket Response ──► Browser TTS (Web Speech API)   │
                        └─────────────────────────────────────────────────────┘
```

---

## 📊 Dataset Strategy

### Data Collection

| Property | Detail |
|---|---|
| **Sign Language** | Indian Sign Language (ISL) |
| **Number of Classes** | 20 isolated sign words |
| **Input Format** | Short video clips per sign |
| **Frame Target** | 64 frames per video (fixed-length extraction) |
| **Dataset Split** | 80% Train / 20% Validation (stratified) |

### Supported Sign Classes (20)

```
alright  · bad       · beautiful  · blind    · deaf
good     · good_afternoon · good_morning · happy   · he
hello    · how_are_you   · i         · it      · loud
no_sign  · quiet    · sad       · she     · you
```

### Landmark Representation

Each video is encoded as a **fixed 64-frame sequence** of four landmark arrays:

| Body Part | Landmarks | Features/Point | Shape |
|---|---|---|---|
| **Pose** | 33 | x, y, z, visibility | `(64, 33, 4)` |
| **Face** | 468 | x, y, z | `(64, 468, 3)` |
| **Left Hand** | 21 | x, y, z | `(64, 21, 3)` |
| **Right Hand** | 21 | x, y, z | `(64, 21, 3)` |

Alongside each array, a **binary mask** `(64,)` tracks frame-level landmark presence:
- `1.0` → landmarks detected in this frame
- `0.0` → landmarks missing (occluded, out of frame, etc.)

This masking makes the model robust to **partial observations** without data imputation.

---

## 🧠 Model Components

### Overview: Multi-Branch CNN + BiLSTM

The model (`SignLanguageModel`) has **four parallel branches** — one for each landmark type — followed by a shared classifier.

```
Input (64 frames)
      │
      ├── Pose  (33, 4)   → LandmarkBranch → Pose Feature  (256-d)
      ├── Face  (468, 3)  → LandmarkBranch → Face Feature  (256-d)
      ├── LHand (21, 3)   → LandmarkBranch → LHand Feature (256-d)
      └── RHand (21, 3)   → LandmarkBranch → RHand Feature (256-d)
                                                   │
                                             Concatenate (1024-d)
                                                   │
                                         Linear(1024 → 256) + ReLU + Dropout(0.5)
                                                   │
                                         Linear(256 → num_classes)
                                                   │
                                               Logits → Predicted Sign
```

### LandmarkBranch (per body part)

Each branch applies the following sequence:

```
Frame-wise CNN                    Temporal BiLSTM
─────────────────                 ─────────────────────────────
Conv1d(in, 64, k=1)               BiLSTM(cnn_out, hidden=160)
BatchNorm + ReLU + Dropout(0.2)   → Forward + Backward hidden states
Conv1d(64, cnn_out, k=1)          → Masked mean-pooling over T frames
BatchNorm + ReLU                  → Feature vector (hidden×2 = 320-d)
Mean-pool over landmarks N        
→ Per-frame embedding (cnn_out)   
```

**Masking in BiLSTM:**
```
output = BiLSTM(x) * mask          # zero out missing frames
feat   = output.sum(T) / mask.sum(T)   # mean only over valid frames
```

### Hyperparameters (Best from Optuna Tuning)

| Parameter | Value |
|---|---|
| `cnn_out` | 128 (web) / 64 (training) |
| `lstm_hidden` | 160 (web) / 96 (training) |
| `bidirectional` | True |
| `fc_dropout` | 0.5 |
| `learning_rate` | 0.000489 |
| `weight_decay` | 0.000777 |
| `optimizer` | Adam |
| `lr_scheduler` | StepLR (step=50, γ=0.5) |
| `epochs` | 200 |
| `batch_size` | 4 |
| `sequence_length` | 64 frames |

---

## 🗺️ Implementation Roadmap

### Phase 1 — Data & Preprocessing ✅
- [x] Raw video collection (ISL gestures)
- [x] Fixed-length frame extraction (64 frames/video)
- [x] YOLOv11 person detection & signer cropping
- [x] MediaPipe Holistic landmark extraction (pose + face + hands)
- [x] Binary mask generation for missing landmarks

### Phase 2 — Model Development ✅
- [x] Multi-branch CNN + BiLSTM architecture design
- [x] Per-branch masked mean-pooling
- [x] Hyperparameter tuning with Optuna
- [x] Training loop with best-model checkpointing

### Phase 3 — Web App ✅
- [x] FastAPI backend with WebSocket support
- [x] Real-time webcam capture in browser
- [x] Client-side MediaPipe landmark overlay (visual feedback)
- [x] Server-side rolling buffer + on-demand inference
- [x] Text-to-Speech output via Web Speech API

### Phase 4 — Future Enhancements 🔜
- [ ] ONNX export for faster inference (<200ms latency target)
- [ ] Expand to 100+ ISL classes
- [ ] Continuous/sentence-level sign recognition
- [ ] Mobile app (iOS/Android)
- [ ] Browser extension for real-time video call captioning
- [ ] User personalization & adaptive learning

---

## 📂 Project Structure

```
Sign-Language-to-Speech-main/
│
├── datasets/                        # (not committed) Training data
│   ├── raw/                         #   Raw sign language videos
│   ├── frames/                      #   Extracted frames (64/video)
│   ├── yolo_crops/                  #   YOLO-cropped signer regions
│   └── landmarks_masked/            #   Final .npy landmark + mask arrays
│       ├── X_pose.npy  X_face.npy  X_lhand.npy  X_rhand.npy
│       ├── M_pose.npy  M_face.npy  M_lhand.npy  M_rhand.npy
│       └── y.npy
│
├── models/                          # Model definitions & weights
│   ├── sign_model.py                #   Full SignLanguageModel (training)
│   ├── final_sign_model.py          #   Finalized model variant
│   ├── small_sign_model.py          #   Lightweight variant
│   ├── tunable_sign_model.py        #   Optuna-tunable variant
│   ├── saved_model.h5               #   Keras model (legacy)
│   ├── yolo11n.pt                   #   ⚠️ YOLOv11 weights (download separately)
│   └── best_web_model.pth           #   ⚠️ Trained PyTorch model (after training)
│
├── scripts/                         # Standalone pipeline scripts
│   ├── extract_frames.py            #   Step 1: Video → frames
│   ├── run_yolo_detection.py        #   Step 2: Frames → YOLO crops
│   ├── extract_landmarks.py         #   Step 3: Crops → landmark .npy files
│   ├── mask_only.py                 #   Step 4: Generate binary mask arrays
│   ├── normalize_landmarks.py       #   Optional: normalize landmark coords
│   ├── augment_yolo_crops.py        #   Optional: augment cropped frames
│   ├── train.py                     #   Step 5: Train CNN+BiLSTM model
│   ├── tune.py                      #   Optuna hyperparameter search
│   ├── debug_train.py               #   Debug training loop
│   └── record_sign.py               #   Record new sign videos via webcam
│
├── src/                             # Reusable source modules
│   ├── frame_extractor.py           #   Fixed-frame video sampler
│   ├── detection/
│   │   └── yolo_detector.py         #   YOLOv11 inference wrapper
│   ├── landmarks/
│   │   ├── landmark_extractor.py    #   MediaPipe Holistic wrapper
│   │   └── dataset.py               #   PyTorch Dataset + DataLoader factory
│   └── preprocessing/
│       └── frame_extractor.py       #   Preprocessing utilities
│
├── web_app/                         # FastAPI web application
│   ├── app_ws.py                    #   Main server (WebSocket + REST)
│   ├── model_def.py                 #   CNN+BiLSTM model (inference copy)
│   ├── inference.py                 #   Inference utilities
│   ├── requirements.txt             #   Web app dependencies
│   └── static/
│       ├── index.html               #   Frontend (webcam + UI)
│       └── js/                      #   Client-side JavaScript
│
├── notebooks/                       # Jupyter exploration notebooks
│   ├── data_checks.ipynb            #   Dataset inspection & visualization
│   ├── augment_checks.ipynb         #   Augmentation experiments
│   └── model_inference.ipynb        #   Offline model testing
│
├── isl-env/                         # Virtual environment (local)
├── requirements.txt                 # Root Python dependencies
├── training_history.csv             # Epoch-wise loss & accuracy log
├── optuna_tune_summary.csv          # Hyperparameter search results
├── WORKFLOW_DIAGRAM.md              # Detailed workflow diagrams
├── CHALLENGES_AND_FUTURE.md         # Challenges & roadmap notes
└── README.md                        # This file
```

---

## 🔧 Training Pipeline

The model is trained offline in **5 sequential steps**. Run all scripts from the project root with the virtual environment active.

### Step 1 — Extract Frames from Videos

```bash
python scripts/extract_frames.py \
    --input_dir  datasets/raw \
    --output_dir datasets/frames
```
- Samples exactly **64 frames** per video (uniform temporal sampling)
- Saves frames as `.jpg` files, one sub-folder per video

### Step 2 — Detect & Crop Signer with YOLOv11

```bash
python scripts/run_yolo_detection.py \
    --frames_dir datasets/frames \
    --output_dir datasets/yolo_crops \
    --model_path models/yolo11n.pt
```
- Runs YOLOv11 (`conf=0.3`) on each frame
- Crops the **largest detected person bounding box**
- Improves downstream landmark quality by removing background clutter

### Step 3 — Extract MediaPipe Landmarks

```bash
python scripts/extract_landmarks.py \
    --crops_dir  datasets/yolo_crops \
    --output_dir datasets/landmarks
```
- Runs **MediaPipe Holistic** on each cropped frame
- Outputs four `.npy` arrays per split: `X_pose`, `X_face`, `X_lhand`, `X_rhand`, `y`

### Step 4 — Generate Binary Masks

```bash
python scripts/mask_only.py
```
- Reads `datasets/landmarks/` arrays
- Writes `M_pose`, `M_face`, `M_lhand`, `M_rhand` mask arrays to `datasets/landmarks_masked/`
- Mask value `1.0` = landmarks detected; `0.0` = all-zero (absent)

### Step 5 — Train the Model

```bash
python scripts/train.py
```
- Loads masked landmark arrays from `datasets/landmarks_masked/`
- Trains `SignLanguageModel` for **200 epochs** with Adam + StepLR
- Saves best checkpoint by validation accuracy → `models/best_web_model.pth`
- Saves full history to `training_history.csv`

```
Training Pipeline Flow:
─────────────────────────────────────────────────────────────────────
 datasets/raw/           →  datasets/frames/      →  datasets/yolo_crops/
 (raw .mp4/.mov)            (64 JPEGs per video)      (signer-cropped frames)
       ↓                          ↓                          ↓
 extract_frames.py        run_yolo_detection.py      extract_landmarks.py
                                                            ↓
                                                   datasets/landmarks/
                                                   (X_pose, X_face, …, y)
                                                            ↓
                                                       mask_only.py
                                                            ↓
                                                   datasets/landmarks_masked/
                                                   (+ M_pose, M_face, …)
                                                            ↓
                                                        train.py
                                                            ↓
                                                   models/best_web_model.pth
─────────────────────────────────────────────────────────────────────
```

---

## 🌐 Web App Inference Pipeline

The web app runs a **FastAPI + WebSocket** server that processes webcam frames in real time.

### Architecture

```
Browser                                   FastAPI Backend
──────────────────────────────────────    ────────────────────────────────────
1. User clicks "Start Camera"
   → Webcam stream starts
   → MediaPipe.js draws live
     pose/hand landmarks (visual HUD)

2. Every frame:
   → Raw frame encoded as base64
   → Sent over WebSocket to /ws    ──►   Received as base64 string
                                         Decoded → cv2 image (640×480)
                                         MediaPipe Holistic extracts:
                                           pose (33,4), face (468,3)
                                           lhand (21,3), rhand (21,3)
                                         Appended to rolling deque(64)
                                         Binary masks updated

3. User presses 's':
   → Sends "START" command         ──►   Buffer RESET
                                         inference_enabled = True

4. When buffer full (64 frames)
   and inference_enabled:          ──►   Stack deques → 4 tensors
                                         Feed to SignLanguageModel
                                         Softmax → argmax → label
                                         Confidence threshold = 0.5
                                         If conf < 0.5 → "no_sign"

5. Result received:                ◄──   Send JSON: {"word": "hello"}
   → Displayed on screen
   → Web Speech API speaks word
   → inference_enabled = False
```

### Key Parameters

| Parameter | Value |
|---|---|
| `SEQUENCE_LENGTH` | 64 frames |
| `CONFIDENCE_THRESHOLD` | 0.5 |
| `model_complexity` (MediaPipe) | 2 (highest accuracy) |
| `num_classes` | 20 |
| Backend framework | FastAPI + Uvicorn |
| Communication | WebSocket (`/ws`) |
| Speech output | Browser Web Speech API (TTS) |

---

## 🚀 How to Run

### 1. Activate the Virtual Environment

```powershell
cd "h:\DEEP LEARNING PROJECT\Sign-Language-to-Speech-main"
isl-env\Scripts\activate
```

### 2. Install Dependencies

```powershell
# PyTorch (CPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# PyTorch (NVIDIA GPU — CUDA 12.1)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# All other packages
pip install mediapipe ultralytics scikit-learn matplotlib fastapi uvicorn websockets pydantic python-multipart opencv-python numpy pandas
```

### 3. Download YOLOv11 Weights

```python
from ultralytics import YOLO
import shutil
YOLO("yolo11n.pt")                          # auto-downloads
shutil.move("yolo11n.pt", "models/yolo11n.pt")
```

### 4. Train the Model (or use pre-trained)

```powershell
python scripts/train.py                     # produces models/best_web_model.pth
```

### 5. Start the FastAPI Server

```powershell
uvicorn web_app.app_ws:app --reload
```

### 6. Open the Web App

Navigate to `http://localhost:8000` in Chrome → click **Start Camera**.

---

## 🎮 Controls

| Key / Button | Action |
|---|---|
| `s` | Reset buffer and **start inference** |
| `r` | Reset buffer only (no inference) |
| **Speak Translation** | Hear the predicted sign word aloud (TTS) |

---

## 📋 Supported Signs (20 Classes)

| | | | | |
|---|---|---|---|---|
| alright | bad | beautiful | blind | deaf |
| good | good_afternoon | good_morning | happy | he |
| hello | how_are_you | i | it | loud |
| no_sign | quiet | sad | she | you |

---

## 👥 Credits

- **Sainava Modak**
- **Kartik Rajput**

## 🙏 Acknowledgements

- [MediaPipe](https://github.com/google-ai-edge/mediapipe) — Holistic landmark extraction
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) — Person detection & cropping
- [PyTorch](https://github.com/pytorch/pytorch) — Deep learning framework
- [FastAPI](https://fastapi.tiangolo.com/) — Backend web server
- [Optuna](https://optuna.org/) — Hyperparameter optimization
