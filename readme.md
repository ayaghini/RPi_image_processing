# Raspberry Pi Image Processing with YOLO v8 & NCNN  
A concise, step‑by‑step guide to set up real‑time object detection **and distance estimation** on a Raspberry Pi (4 or 5) using the [Ultralytics YOLO v8](https://docs.ultralytics.com) models and the high‑performance [Tencent NCNN](https://github.com/Tencent/ncnn) inference engine.

> ✨ **Why this workflow?**  
> YOLO v8 offers state‑of‑the‑art accuracy, while NCNN squeezes maximum FPS out of ARM devices by running pure C++ inference with NEON and Vulkan back‑ends—no heavyweight Python runtime on the critical path.

---

## Table of Contents  
1. [Prerequisites](#prerequisites)  
2. [System prep](#system-prep)  
3. [Project environment](#project-environment)  
4. [Install YOLO & NCNN](#install-yolo--ncnn)  
5. [Quick test (Python)](#quick-test-python)  
6. [Export model → NCNN](#export-model-to-ncnn)  
7. [Run real‑time detection (NCNN C++)](#run-real-time-detection-ncnn-c)  
8. [Camera calibration](#camera-calibration)  
9. [Distance estimation](#distance-estimation)  
10. [Windows workflow](#windows-workflow)  
11. [Troubleshooting](#troubleshooting)  
12. [References](#references)

---

## Prerequisites  
| Component | Minimum spec | Notes |
|-----------|--------------|-------|
| Raspberry Pi | 4 Model B (2 GB) **or** Pi 5 | Enable the CSI or USB camera in `raspi-config`. |
| OS | Raspberry Pi OS Bookworm (64‑bit) | Update/upgrade before starting. |
| Camera | Raspberry Pi HQ Camera **or** USB UVC cam | Resolution ≥ 640×480 recommended. |
| Power | 5 V ⎓ 3 A | Heavy loads during inference. |

---

## System prep  
```bash
sudo apt update && sudo apt full-upgrade -y
# Optional but recommended:
sudo raspi-config nonint do_camera 0      # Enable camera
sudo reboot
```

---

## Project environment  
```bash
# 1 Create a workspace
mkdir -p ~/yolo && cd ~/yolo

# 2 Python virtual env that can still access system libs
python3 -m venv --system-site-packages venv
source venv/bin/activate
```

---

## Install YOLO & NCNN  
### 1 Standard (latest)  
```bash
pip install --upgrade pip
pip install ultralytics ncnn
```

### 2 **Patch for current NCNN export bug**  
The NCNN exporter in v8.3.71+ (April 2025) throws `AttributeError: 'dict' object has no attribute 'byte'`.  
Work‑around: pin the tool‑chain to the last known good versions.  
```bash
pip install --no-cache-dir     ultralytics==8.3.70     torch==2.2.2      torchvision==0.17.2
```
*(Keep an eye on the upstream issue: <https://github.com/ultralytics/ultralytics/issues/19091>)*

---

## Quick test (Python)  
```bash
# Download a tiny model (3.2 MB) and run detection from USB cam 0
yolo detect predict model=yolov8n.pt source=0 show=True
```
> If you need **more FPS**, jump to the NCNN section below.

---

## Export model → NCNN  
```bash
# Replace yolov8n.pt with your custom or larger .pt model
yolo export model=yolov8n.pt format=ncnn
```
*Outputs two files:*  

* `yolov8n_ncnn_model.param`  
* `yolov8n_ncnn_model.bin`

---

## Run real‑time detection (NCNN C++)  
The Ultralytics helper script wraps the native NCNN C++ demo:

```bash
python yolo_detect.py     --model yolov8n_ncnn_model     --source 0     --resolution 640x480
```
*Expect 25–30 FPS on a Pi 4, up to 45 FPS on a Pi 5 (Vulkan backend).*

---

## Camera calibration  
Accurate **distance estimation** requires an intrinsic camera matrix & distortion coefficients.  
```bash
python camera_calibration.py        --rows 10 --cols 7        --square 0.025        --frames 20
# ➜ calibration.npz (saved in current folder)
```
| Flag | Meaning |
|------|---------|
| `--rows`, `--cols` | Inner corners on the chessboard. |
| `--square` | Physical square size in **metres** (25 mm ↔ 0.025). |
| `--frames` | Number of good board detections to average. |

📄 **Chessboard PDF** – print on A4, glue to flat board:  
<https://calib.io/pages/camera-calibration-pattern-generator>

---

## Distance estimation  
```bash
python detect_with_distance.py        --model yolov8x.pt        --calib calibration.npz         --height 1.63        --cam 0
```
| Flag | Purpose |
|------|---------|
| `--height` | Real‑world target height (m). For people, use average eye level (1.63 m). |
| `--calib` | The `.npz` generated in the previous step. |

Algorithm: **pinhole camera model** → depth ≈ (focal_length × real_height) / bbox_pixel_height

---

## Windows workflow  
1. Install Python 3.11 × 64 **and** Git for Windows.  
2. Open *PowerShell* (Admin) →  
   ```powershell
   py -m venv venv; venv\Scripts\Activate.ps1
   pip install ultralytics ncnn
   ```
3. Substitute the camera index (`--cam 0`) with your webcam’s *DirectShow* ID if multiple cams exist.

---

## Troubleshooting  
| Symptom | Fix |
|---------|-----|
| `Illegal instruction (core dumped)` on import ncnn | `sudo apt install -y libprotobuf23 libvulkan1` |
| Black preview window | Check `v4l2-ctl --list-devices` and correct **source** id. |
| Low FPS | Use `yolov8n.pt` or compile NCNN with `-DNCNN_VULKAN=ON`. |
| Distorted colours | Add `--video-mjpeg` for UVC cams that output MJPEG. |

---

## References  
- Ultralytics YOLO v8 Docs  
- Tencent NCNN Docs & Pi benchmark results  
- Zhang, Z. “Flexible Camera Calibration by Viewing a Plane from Unknown Orientations.” ICCV 1999  

---

*Made with ❤️ & caffeine – last updated **23 Apr 2025***  
