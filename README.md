# 🛡️ Fall Detection System for Elderly Care

> **AI-powered real-time fall detection** using Computer Vision and Machine Learning — designed for elderly safety monitoring.

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?logo=opencv)](https://opencv.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10-orange?logo=google)](https://mediapipe.dev/)
[![Platform](https://img.shields.io/badge/Platform-Mac%20%7C%20Raspberry%20Pi%205-lightgrey)]()
[![License](https://img.shields.io/badge/License-Educational-yellow)]()

---

## 📋 Overview

ระบบตรวจจับการล้มสำหรับผู้สูงอายุ โดยใช้ **MediaPipe Pose Estimation** ร่วมกับ **Random Forest Classifier** เพื่อวิเคราะห์ท่าทางและตรวจจับการล้มแบบ Real-time พร้อมแจ้งเตือนผ่าน LINE และ Buzzer

### ✨ Key Features

- 🎯 **Real-time Fall Detection** — ตรวจจับการล้มแบบเรียลไทม์ด้วย State Machine (NORMAL → FALLING → FALLEN → RECOVERING)
- 🤖 **Hybrid AI** — Rule-based velocity detection + Random Forest ML model
- 📱 **LINE Messaging API** — แจ้งเตือนผ่าน LINE พร้อมรูปภาพ
- 🔔 **Multi-channel Alerts** — Buzzer (GPIO), Sound alert, LINE notification
- 📊 **High Accuracy** — 96.7% accuracy, 93.1% recall บน test set (645 samples)
- 🍓 **Raspberry Pi 5 Ready** — รองรับการ deploy บน Pi 5 พร้อม Camera Module

---

## 🏗️ System Architecture

```
┌─────────────┐    ┌──────────────┐    ┌─────────────────┐    ┌──────────────┐
│   Camera    │───▶│  MediaPipe   │───▶│  Fall Detector   │───▶│    Alerts    │
│ (Webcam/Pi) │    │  Pose Est.   │    │  State Machine   │    │ LINE/Buzzer  │
└─────────────┘    └──────────────┘    │  + ML Model      │    └──────────────┘
                                       └─────────────────┘
                                              │
                                     ┌────────┴────────┐
                                     │  Random Forest  │
                                     │   Classifier    │
                                     └─────────────────┘
```

### State Machine Flow

```
NORMAL ──(fall detected)──▶ FALLING ──(impact confirmed)──▶ FALLEN ──(recovery)──▶ RECOVERING ──▶ NORMAL
                                                              │
                                                         ⚠️ ALERT!
                                                    (LINE + Buzzer + Sound)
```

---

## 📁 Project Structure

```
Senior_Project/
├── fall_detection_mac.py           # 🎬 Main detection script 
├── alerts.py                       # 🔔 Alert module (LINE, 
├── train_plot_mac.py               # 🏋️ Model training & 
├── fall_model_from_sequences.pkl   # 🧠 Trained Random Forest 
├── detected_falls/                 # 📸 Saved fall detection 

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Webcam or video file

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/fall-detection-elderly.git
cd fall-detection-elderly
```

### 2. Setup Environment

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Run Detection

```bash
python3 fall_detection_mac.py
```

### 4. Change Video Source

Edit line 16 in `fall_detection_mac.py`:

```python
VIDEO_SOURCE = 'image/vdo2.mp4'     # Video file
VIDEO_SOURCE = 0                     # Webcam
VIDEO_SOURCE = 'rtsp://...'          # IP Camera
```

### ⌨️ Keyboard Controls

| Key | Action |
|-----|--------|
| `Q` | Quit |
| `R` | Reset detector |
| `Space` | Pause |

---

## 📊 Model Performance

### Classification Report

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Fall | 0.947 | 0.931 | 0.939 | 174 |
| Person | 0.975 | 0.981 | 0.978 | 471 |
| **Accuracy** | | | **0.967** | **645** |

### Confusion Matrix

```
              Predicted
            Person  Fall
Actual  Person [ 462    9 ]
        Fall   [  12  162 ]
```

### Video Test Results (100% Accuracy)

| Video | Expected | Result |
|-------|----------|--------|
| Apple Watch Fall | Fall | ✅ Detected |
| vdo2.mp4 | Fall | ✅ Detected |
| fall-01-cam0.mp4 | Fall | ✅ Detected |
| vdo1.mp4 | No Fall | ✅ Not detected |
| vdo3.mp4 | No Fall | ✅ Not detected |
| Walking.mp4 | No Fall | ✅ Not detected |

---

## 📱 LINE Messaging API Setup

> ⚠️ LINE Notify ปิดบริการ 31 มี.ค. 2025 — ใช้ Messaging API แทน

1. สร้าง Channel ที่ [LINE Developers Console](https://developers.line.biz/console/)
2. เลือก **Messaging API** → Copy **Channel Access Token**
3. เพิ่ม Bot เป็นเพื่อน → หา **User ID**
4. ตั้งค่า Environment Variables:

```bash
export LINE_CHANNEL_ACCESS_TOKEN="your_channel_token"
export LINE_USER_ID="your_user_id"
```

5. ทดสอบ:

```bash
python3 alerts.py
```

---

## 🍓 Deploy on Raspberry Pi 5

### Hardware Requirements

| Item | Description | Est. Price |
|------|-------------|------------|
| Raspberry Pi 5 | 8GB RAM recommended | ฿2,500 |
| WEBCAM | 1080p 30fps | ฿1,200 |
| Active Cooler | Fan + Heatsink | ฿400 |
| Power Supply | 27W USB-C | ฿500 |
| Buzzer | Active 5V | ฿20 |

### Quick Setup

```bash
# Install OS: Raspberry Pi OS (64-bit) Bookworm
sudo apt update && sudo apt upgrade -y

# System dependencies
sudo apt install -y python3-pip python3-venv python3-opencv \
    libatlas-base-dev libhdf5-dev

# Python environment
python3 -m venv ~/fall_env
source ~/fall_env/bin/activate
pip install -r requirements_pi.txt
```

### Buzzer Wiring

```
GPIO27 (Pin 13) ──→ Buzzer (+)
GND    (Pin 14) ──→ Buzzer (-)
```

### Performance Benchmarks

| Platform | Resolution | FPS |
|----------|------------|-----|
| Mac M4 | 1080p | 30 |
| Pi 5 (8GB) | 640×480 | 15–20 |
| Pi 5 (4GB) | 640×480 | 10–15 |

> 📖 See [MIGRATION_PI5.md](MIGRATION_PI5.md) for the complete migration guide.

---

## 🔧 Training Your Own Model

```bash
# Prepare dataset in dataset/ directory (train/valid/test splits)
# Then run:
python3 train_plot_mac.py
```

This will:
1. Extract pose keypoints from images using MediaPipe
2. Train a Random Forest Classifier
3. Generate confusion matrix, feature importance, and learning curves
4. Save the model to `fall_model_from_sequences.pkl`

---

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| Camera not found | `sudo raspi-config` → Interface → Camera → Enable |
| Low FPS | Reduce resolution or set `model_complexity=0` |
| LINE not sending | Verify token and internet connection |
| Buzzer not working | Check wiring and GPIO pin |
| Memory error | Use Pi 5 8GB or increase swap |

---

## 🛠️ Tech Stack

- **Pose Estimation**: [MediaPipe](https://mediapipe.dev/) Pose
- **ML Model**: Random Forest (scikit-learn)
- **Computer Vision**: OpenCV
- **Notifications**: LINE Messaging API
- **Hardware**: Raspberry Pi 5 + Camera Module 3
- **GPIO**: RPi.GPIO (Buzzer control)

---

## 📄 License

For educational purposes — **Senior Project, Department of Electrical Engineering, Chulalongkorn University**

---

<p align="center">
  <i>Built with ❤️ for elderly safety</i>
</p>

