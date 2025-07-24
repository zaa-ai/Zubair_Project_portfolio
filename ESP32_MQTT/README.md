
## 🔍 Project Overview

When seconds count, relying on remote servers can introduce dangerous latency or fail entirely. This system:

- Continuously samples accelerometer + gyroscope data  
- Runs a quantized 1D-CNN model entirely on the ESP32  
- Triggers an immediate SOS alert over Wi-Fi (via IFTTT/MQTT)  
- Delivers a full “fall + impact” detection + notification cycle in under 2 seconds, with zero cloud inference  

---

## ⚙️ Architecture & Workflow
┌──────────┐ I²C DMA ┌─────────┐ Inference ┌─────────┐ HTTP POST ┌─────────┐
│ MPU6050 │ ───────────▶ │ ESP32 │ ─────────────▶ │ CNN │ ─────────────▶ │ IFTTT │
│ (IMU) │ │ (ring │ │ (TFLite │ │ Webhook │
│ │ │ buffer)│ │ 1D-CNN) │ │ │
└──────────┘ └─────────┘ └─────────┘ └─────────┘


1. **Data Collection**  
   - 400+ two-second windows sampled at 50 Hz  
   - Labels: `Fall` vs. `No-Fall`  

2. **Model Training (`fall_detection_train.py`)**  
   - 1D-CNN: Conv1D→Pool→Conv1D→Pool→Flatten→Dense→Softmax  
   - >95 % validation accuracy  

3. **Quantization & Conversion (`fall_model_convert.py`)**  
   - Post-training int8 quantization  
   - Final TFLite size ~60 KB, inference time 30–50 ms/window  

4. **On-Device Deployment**  
   - Continuous DMA sampling into a ring buffer  
   - Inference loop with <2 s end-to-end latency  
   - SOS alert via HTTP POST to an IFTTT webhook (debounced)

---
## Performance & Impact

- Model footprint: ~60 KB (int8)

- Inference: 30–50 ms per window

- End-to-end latency: <2 s

- Accuracy: >95 % on held-out validation set

- Efficiency: 4× smaller & 3× faster than float baseline

