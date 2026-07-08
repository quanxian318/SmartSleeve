# Smart Health Sleeve 智能健康袖套

> Language / 语言：**English** ｜ [简体中文 (README_cn.md)](README_cn.md)

A wearable EMG rehabilitation-training system built on **RDK X5 + ESP32**. Silver-fabric dry electrodes capture surface EMG (sEMG); a `TCN + MLP` deep-learning model predicts muscle-activation ratios, accelerated on the Horizon **BPU** (27 ms → 1.98 ms). A WeChat Mini Program visualizes muscle state in real time via a **3D digital twin**, and an **offline Chinese voice assistant** guides the rehab session.

---

## Table of Contents

- [System Architecture](#system-architecture)
- [Dual-Model Cross-Validation](#dual-model-cross-validation)
- [Model Info](#model-info)
- [Directory & File Reference](#directory--file-reference)
  - [Root](#root)
  - [1_RDK_X5_System — On-device deployment + voice assistant](#1_rdk_x5_system--on-device-deployment--voice-assistant)
  - [2_EMG_Model — EMG prediction model](#2_emg_model--emg-prediction-model)
  - [3_Quantization — Quantization analysis tools](#3_quantization--quantization-analysis-tools)
  - [4_ESP32_Firmware — ESP32 firmware](#4_esp32_firmware--esp32-firmware)
  - [5_Mini_Program — WeChat Mini Program](#5_mini_program--wechat-mini-program)
  - [6_PCB — Hardware design](#6_pcb--hardware-design)
  - [7_3D_Models — 3D arm models](#7_3d_models--3d-arm-models)
- [Voice Interaction Subsystem](#voice-interaction-subsystem)
- [Quick Start](#quick-start)
- [Hardware Pinout](#hardware-pinout)
- [Requirements](#requirements)
- [Versions](#versions)
- [License](#license)

---

## System Architecture

```
┌─ Hardware ────────────────────────────────────────────────────┐
│  5V power → custom expansion board → ESP32 + silver-fabric      │
│            electrodes → EMG acquisition → BLE broadcast         │
│  RDK X5 + USB camera → skeleton tracking → BPU inference        │
│  MX1508 motor drivers ×8 → 1030 coreless motors → haptic cue    │
└──────────────────────────────────────────────────────────────┘

┌─ Communication ──────────────────────────────────────────────┐
│  ESP32   ──BLE────────────→ WeChat Mini Program                │
│  RDK X5  ──WiFi WebSocket──→ WeChat Mini Program (ws://<ip>:8765)│
└──────────────────────────────────────────────────────────────┘

┌─ Application ────────────────────────────────────────────────┐
│  Mini Program: 3D digital twin + lightweight RandomForest +   │
│                training management                            │
│  RDK X5      : AnchorCalibTCN BPU inference (27 ms → 1.98 ms)  │
│                + voice assistant                              │
└──────────────────────────────────────────────────────────────┘
```

## Dual-Model Cross-Validation

The Mini Program runs a lightweight RandomForest for real-time 3D rendering, while the RDK X5 runs the larger TCN model together with camera skeleton tracking as ground truth — cross-checking *predicted EMG* against *BLE-measured EMG* for accuracy correction.

---

## Model Info

| Attribute | Value |
|-----------|-------|
| Architecture | AnchorCalibTCN (TCN + MLP fusion) |
| Parameters | 2,193,730 |
| Input | `[1, 26, 1, 64]` (motion 10ch + calib 16ch) |
| Output | `[biceps_ratio, triceps_ratio]` |
| ONNX opset | 11 |
| BPU arch | bayes-e (RDK X5) |

### Inference Performance

| Platform | Latency | FPS | Model size |
|----------|---------|-----|------------|
| RDK X5 CPU (onnxruntime) | 27.2 ms | 37 | 8.6 MB |
| RDK X5 BPU | **1.98 ms** | **499** | 2.5 MB |

### INT8 Quantization Accuracy

| Metric | Value |
|--------|-------|
| Cosine similarity | 0.9935 |
| R² | 0.9298 |
| Pearson r | 0.9648 |
| MAE | 0.2105 |
| RMSE | 0.3356 |

---

## Directory & File Reference

> The tables below cover **every file** in the repository. Files marked *empty* are 0-byte placeholders whose functionality is not yet committed.

### Root

| File | Size | Description |
|------|------|-------------|
| `README.md` | — | English documentation (this file, default landing page) |
| `README_cn.md` | 15 KB | Detailed Chinese documentation |
| `.gitignore` | 389 B | Git ignore rules (model caches, temp files, etc.) |

### 1_RDK_X5_System — On-device deployment + voice assistant

All on-device runtime scripts, ROS2 nodes, inference wrappers, the voice assistant and model assets.

| File | Size | Description |
|------|------|-------------|
| `start_all.sh` | 16 KB | One-shot launcher: brings up skeleton tracking / EMG bridge / cross-validation / voice assistant / WebSocket; cleans up child processes on exit |
| `body_angle_node.py` | 15 KB | Body-skeleton angle detection ROS2 node: camera → AI body detection → publishes `/body_arm_angles` |
| `ros2_emg_bridge.py` | 14 KB | EMG inference bridge ROS2 node (v1): angle → TCN → publishes `/virtual_emg` |
| `ros2_emg_bridge_v2.py` | 15 KB | EMG bridge v2, with `--ml_predict` and `--load_calib` calibration loading |
| `tcn_bpu_predictor.py` | 6 KB | BPU inference wrapper (replaces onnxruntime); loads the `.bin` model for hardware-accelerated inference |
| `emg_cross_validation_v2.py` | 26 KB | Cross-validation: TCN prediction vs real EMG, plus muscle-compensation / electrode-quality checks; publishes `/emg_validation` + `/emg_alerts` |
| `emg_deploy.py` | 4.5 KB | Standalone inference script (runs without ROS2 for quick model checks) |
| `udp_emg_receiver.py` | **empty** | UDP EMG / heart-rate receiver (0-byte placeholder, TBD) |
| `screen_server.py` | 19 KB | On-board screen HTTP/WebSocket data service (local visualization UI) |
| `ws_server.py` | 7 KB | ROS2 → WebSocket bridge, exposes `ws://0.0.0.0:8765` for the Mini Program |
| `voice_demo_v7.py` | 82 KB | **Voice assistant v7**: VAD/push-to-talk modes, offline Whisper ASR, keyword+LLM intent, priority TTS, training state machine, background alert monitor (see [Voice Interaction Subsystem](#voice-interaction-subsystem)) |
| `voice_demo_v6.py` | 38 KB | Voice assistant v6 (legacy, kept for reference) |
| `voice_agent_tts.py` | **empty** | TTS module (0-byte placeholder; the actual TTS logic is inlined in v7) |
| `scripts/emg-system.service` | 987 B | systemd unit for auto-start on boot |
| `motion_scaler_63subj.pkl` | 823 B | Motion-feature StandardScaler (fit on the 63-subject dataset) |
| `calib_scaler_63subj.pkl` | 967 B | Calibration-vector StandardScaler |
| `calibration_config_63subj.json` | 730 B | Calibration config (default calibration vector, channel defs) |
| `anchorcalib_tcn_bpu_v2.bin` | 2.4 MB | BPU-compiled model (bayes-e), loaded on device |

### 2_EMG_Model — EMG prediction model

Training scripts, weights and deployment artifacts for the deep-learning model.

**`training/` — training scripts + weights**

| File | Size | Description |
|------|------|-------------|
| `model.py` | 11 KB | Original PyTorch model (Conv1d TCN + MLP fusion) |
| `model_bpu.py` | 8.5 KB | BPU-native model (Conv2d, adapted to Horizon operator constraints) |
| `train.py` | 47 KB | LOSO (Leave-One-Subject-Out) cross-validation training script |
| `losses.py` | 2.3 KB | Combined loss (MSE + correlation + smoothness terms) |
| `cache_data.py` | 535 B | Data preprocessing & caching |
| `export_bpu.py` | 15 KB | Original BPU ONNX export |
| `export_bpu_v2.py` | 6.4 KB | ONNX opset=11 export (used for the current deployment) |
| `migrate_weights.py` | 7.6 KB | Conv1d → Conv2d weight migration tool |
| `adapt_zenodo.py` | 4.8 KB | Zenodo public-dataset adapter |
| `adapt_lucchetti.py` | 5.7 KB | Lucchetti dataset adapter |
| `anchorcalib_tcn.pt` | 8.4 MB | 10-subject model weights |
| `anchorcalib_tcn_63subj.pt` | 8.5 MB | 63-subject model weights (primary) |
| `anchorcalib_tcn_bpu_v2.pt` | 8.4 MB | BPU-migrated Conv2d weights |
| `motion_scaler_63subj.pkl` | 823 B | Motion-feature scaler (matches on-device copy) |
| `calib_scaler_63subj.pkl` | 967 B | Calibration-vector scaler (matches on-device copy) |

**`deployed/` — deployment artifacts**

| File | Size | Description |
|------|------|-------------|
| `anchorcalib_tcn_bpu_v2.onnx` | 8.4 MB | BPU-ready ONNX (opset=11, runnable via onnxruntime) |
| `anchorcalib_tcn_bpu_v2.bin` | 2.4 MB | BPU-compiled model (`hb_mapper` output) |

**Other**

| File | Size | Description |
|------|------|-------------|
| `report.html` | 6.9 KB | BPU deployment accuracy / performance report (with quantization charts) |

### 3_Quantization — Quantization analysis tools

INT8 quantization accuracy evaluation and operator-level analysis.

| File | Size | Description |
|------|------|-------------|
| `quant_analysis.py` | 13 KB | Overall INT8 quantization accuracy (cosine sim / R² / MAE, etc.) |
| `operator_level_quant.py` | 20 KB | Low-level operator quantization experiments |
| `onnx_node_analysis.py` | 20 KB | Per-node (operator) error analysis on ONNX |
| `compare_models.py` | 21 KB | FP32 / ONNX / BPU three-way output comparison |
| `gen_figures.py` | 37 KB | Batch generation of report/paper figures |

### 4_ESP32_Firmware — ESP32 firmware

Arduino Nano ESP32 (ABX00083) firmware: BLE broadcast, PWM motor drive, ADC acquisition.

| File | Size | Description |
|------|------|-------------|
| `PulseSensorAmped_Arduino_1dot2.ino` | 6.8 KB | Main sketch: pulse ADC acquisition + PWM motors + main loop |
| `BLE_Manager.ino` | 5.6 KB | BLE Heart Rate Service + custom EMG service, NOTIFY streaming |
| `WiFi_Manager.ino` | 2.6 KB | WiFi management (reserved; BLE is primary) |
| `Interrupt.ino` | 5.4 KB | Timer ISR: pulse-peak detection & IBI computation |
| `ABX00083-datasheet.pdf` | 3.3 MB | Arduino Nano ESP32 official datasheet |

### 5_Mini_Program — WeChat Mini Program

Native WeChat Mini Program + Three.js (r108), with both patient and doctor roles.

**Entry / global config**

| File | Size | Description |
|------|------|-------------|
| `app.js` | 4.2 KB | App entry; `globalData` and cloud init |
| `app.json` | 1.4 KB | Global routes, tabBar, window config |
| `app.wxss` | 1.4 KB | Global styles |
| `project.config.json` | 988 B | DevTools project config |
| `sitemap.json` | 191 B | WeChat indexing config |
| `package.json` | 238 B | npm deps (threejs-miniprogram, pako) |
| `package-lock.json` | 915 B | Dependency lock |

**`utils/` — utilities**

| File | Size | Description |
|------|------|-------------|
| `dataManager.js` | 11 KB | Global sensor-data singleton: WebSocket connection, shared state, subscriber pattern, `runInference()` orchestrator |
| `rfInference.js` | 4.7 KB | Parses the binary RandomForest format; `predict(features)` → `[biceps, triceps]` |
| `roleManager.js` | 4.3 KB | Doctor / patient role management |
| `scorer.js` | 7.9 KB | Training-action scoring algorithm |

**`pages/` — pages** (each with `.js / .json / .wxml / .wxss`)

| Page | Main script | Description |
|------|-------------|-------------|
| `index/` | 40 KB | Home: BLE connect to ESP32, live HR/EMG, 2D arm animation, AI diagnosis, RDK WebSocket init |
| `digitalTwin/` | 35 KB | 3D digital-twin page (breakdown below) |
| `history/` | 13 KB | Profile / health records; cloud DB read-write of training history |
| `roleSelect/` | 1.3 KB | First-launch doctor / patient role selection |
| `taskDetail/` | 15 KB | Training-task detail & execution |
| `taskPublish/` | 5.6 KB | Doctor: publish a training task |
| `patientDetail/` | 3.5 KB | Doctor: view a single patient |
| `actionRecord/` | 6 KB | Doctor: record a standard-action template |

**`pages/digitalTwin/` — 3D digital-twin internals**

| File | Size | Description |
|------|------|-------------|
| `digitalTwin.js` | 35 KB | Page logic: three-tier model download/cache, render loop, ML inference orchestration |
| `gltfLoader.js` | 80 KB | GLTF/GLB loader (adapted for the Mini Program) |
| `armModel.js` | 11 KB | Procedural 3D arm hierarchy (shoulderPivot → upper arm → elbowPivot → forearm → wristPivot) |
| `muscleMaterials.js` | 4 KB | EMG → color map (green → yellow → orange → red) |
| `orbitControls.js` | 5.7 KB | Touch orbit/zoom control + `raycastMuscles()` tap detection |
| `threeAdapter.js` | 2 KB | WebGL renderer factory wrapping threejs-miniprogram |

**`components/navigation-bar/` — custom nav-bar component**

| File | Description |
|------|-------------|
| `navigation-bar.js / .json / .wxml / .wxss` | Custom top navigation-bar component |

**`cloudfunctions/` — cloud functions** (each has `config.json / index.js / package.json`)

| Function | Description |
|----------|-------------|
| `setUserRole/` | Set user role (doctor / patient) |
| `joinDoctor/` | Patient binds to a doctor via invite code |
| `joinDoctorByQR/` | Patient binds to a doctor via QR scan |
| `manageTask/` | Training-task CRUD |
| `fetchMyTasks/` | Fetch the current user's task list |
| `actionTemplates/` | Standard-action template read/write |
| `getDoctorPatients/` | Doctor lists their patients |
| `getAiReport/` | Generate an AI rehab report (calls an LLM) |

**`images/` — tab icons**

| File | Description |
|------|-------------|
| `tab-monitor.png` / `tab-monitor-active.png` | Monitor tab icon (normal / active) |
| `tab-twin.png` / `tab-twin-active.png` | Digital-twin tab icon |
| `tab-user.png` / `tab-user-active.png` | Profile tab icon |

**`miniprogram_npm/` — built npm packages**

| File | Size | Description |
|------|------|-------------|
| `threejs-miniprogram/index.js` | 583 KB | Three.js r108, Mini Program build |
| `pako/index.js` | 224 KB | gzip inflate library (decompresses `model.bin.gz`) |
| `pako/index.js.map` | 265 KB | pako sourcemap |

### 6_PCB — Hardware design

| File | Size | Description |
|------|------|-------------|
| `Altium_SmartSleeve1.zip` | 1.1 MB | Custom expansion-board PCB project (Altium Designer; schematic + PCB layout) |

### 7_3D_Models — 3D arm models

Multiple LOD variants of the Z-Anatomy model + a Blender script.

| File | Size | Description |
|------|------|-------------|
| `arm_skinned.glb` | 4.3 MB | Skinned/rigged version (most complete) |
| `arm_model.glb` | 2.8 MB | Full model |
| `arm_model_clean.glb` | 2.7 MB | Cleaned version (redundant nodes removed) |
| `arm_model_r108.glb` | 1.5 MB | Three.js r108-compatible version |
| `arm_model_compressed.glb` | 0.6 MB | Compressed version (loaded by the Mini Program) |
| `GLTFLoader_raw.js` | 80 KB | GLTF loader source (reference) |
| `add_armature.py` | 6 KB | Blender armature-binding / export script |

---

## Voice Interaction Subsystem

The RDK X5 vision side ships a **local Chinese voice assistant** (`1_RDK_X5_System/voice_demo_v7.py`) that forms a *listen → understand → speak* loop around elbow flexion–extension rehab, and runs fully offline.

```
Jieli wireless mic → faster-whisper (offline zh ASR) → 3-tier intent → PriorityTTS speech
      ↑                                                  ↓
   16 kHz capture                    keyword → DeepSeek LLM → echo fallback
                                                  ↓
   ROS2 subscribes /body_arm_angles, /virtual_emg, /emg_validation, /emg_alerts, /raw_emg
```

| Stage | Implementation |
|-------|----------------|
| ASR | faster-whisper (tiny preferred, base fallback, int8); **push-to-talk** and **VAD/daemon** modes |
| Intent | 3-tier cascade: keyword match → DeepSeek (`deepseek-chat`) multi-turn chat → echo fallback; silent degradation when offline |
| TTS | `PriorityTTSSpeaker`: online edge-tts (`zh-CN-XiaoxiaoNeural`) + offline espeak-ng fallback, WAV cache/prewarm, 4-level priority queue, high-priority alerts interrupt current speech |
| State | `SystemState` thread-safely aggregates angle/EMG/validation/alerts/HR; training state machine (start/pause/resume/stop, debounced rep counting) + voice calibration |
| Monitor | `TrainingMonitor` every 5 s: electrode-dropout / compensation / form alerts, HR high/low warnings, periodic progress, step prompts, inactivity reminders |

**Supported voice commands**: start / pause / resume / stop training, training report, AI report, diagnosis, status, current angle, muscle status, start calibration, volume control (louder, softer, set to 50%), help.

Launch:

```bash
python3 voice_demo_v7.py --daemon    # daemon (VAD auto-trigger, recommended)
python3 voice_demo_v7.py --vad       # voice-activity-detection mode
python3 voice_demo_v7.py             # push-to-talk (debug)
python3 voice_demo_v7.py --no-llm    # keyword-only, disable AI chat
```

DeepSeek API key (optional, for AI chat & rehab summaries):

```bash
# Get a key: https://platform.deepseek.com/api_keys
echo 'sk-your-api-key' > ~/.deepseek_key
```

> Note: AI chat depends on `voice_llm.py` (the DeepSeek client), which is **not** currently committed to this repo. Without it, v7 gracefully falls back to keyword + echo mode; all other features are unaffected.

---

## Quick Start

### 1. RDK X5 on-device deployment

```bash
scp 1_RDK_X5_System/* root@<rdk-ip>:/root/
cp /root/scripts/emg-system.service /etc/systemd/system/
systemctl daemon-reload && systemctl enable emg-system
/root/start_all.sh          # launch skeleton / EMG bridge / cross-validation / voice / WebSocket
```

Verify:

```bash
ros2 topic list                 # list topics
ss -lntp | grep 8765            # check the WebSocket port
```

### 2. Model training (GPU server)

```bash
cd 2_EMG_Model/training
python train.py --data_dir ../data/ --output_dir ./results/
```

### 3. BPU compilation (requires Horizon J5 OpenExplorer toolchain)

```bash
hb_mapper makertbin --model-type onnx --fast-perf \
  --model anchorcalib_tcn_bpu_v2.onnx \
  --march bayes-e -i merged_input 1x26x1x64
```

### 4. ESP32 firmware

1. Open `4_ESP32_Firmware/PulseSensorAmped_Arduino_1dot2.ino` in Arduino IDE
2. Select board `Arduino Nano ESP32 (ABX00083)`
3. Wire: pulse sensor → A0, EMG electrodes → instrumentation amp → ADC
4. After flashing, the serial monitor (115200) shows the startup log

### 5. WeChat Mini Program

Open `5_Mini_Program/` in WeChat DevTools, run **Tools → Build npm**, then compile. On first entry to the digital-twin page the AI model (~6 MB) is auto-downloaded from cloud storage and cached.

---

## Hardware Pinout

| Component | Connection |
|-----------|-----------|
| Power | 5V DC into the custom expansion board; on-board boost/divider circuitry |
| PPG pulse sensor | GND / SIG=A0 / 5V (broken out on the board) |
| EMG electrodes (silver fabric) | 3.5mm TRS → AD8221 + TL074 instrumentation amp (±9V supply, conditioned to 0–3.3V) |
| MX1508 motor drivers | GPIO-controlled, 8 channels (integrated on the board, 1 kHz PWM) |
| RDK X5 | USB camera, serial comms |

---

## Requirements

| Component | Requirement |
|-----------|-------------|
| RDK X5 | Ubuntu 22.04 ARM64, hobot-dnn ≥ 3.0, ROS2 Humble |
| Training server | Python 3.10+, PyTorch 2.x, CUDA 12+ |
| BPU compilation | Ubuntu 20.04, Python 3.8, Horizon J5 OE v1.1.77+ |
| ESP32 | Arduino Nano ESP32 (ABX00083), PulseSensor Playground |
| Mini Program | WeChat DevTools, threejs-miniprogram v0.0.8 (Three.js r108), pako v2.1.0 |
| Voice assistant | alsa-utils, espeak-ng, ffmpeg, edge-tts, faster-whisper |

---

## Versions

- Model: AnchorCalibTCN 63-subject
- BPU toolchain: hbdk 3.49.13 / hb_mapper 1.24.1
- RDK X5 runtime: hobot-dnn 3.0.4 / hbrt 3.15.55.0
- ESP32: Arduino Nano ESP32

---

## License

For academic research and educational use only. Hardware design files are under **CC BY-NC-SA 4.0**.
