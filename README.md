# Zubair_Project_Portfolio

_All my work in Embedded Systems, IoT & Artificial Intelligence_

---

# Zubair Ahmed Ansari

> **Master’s in Automation & Robotics** | Embedded AI & Edge-AI Enthusiast  
> 📍 Dortmund, Germany | 📧 ansarizubair9943@gmail.com | [LinkedIn](https://www.linkedin.com/in/zubairahmed-ansari-251379194)

---

## 🚀 About Me

I’m a research-driven engineer with hands-on experience in embedded systems, TinyML, and secure AI at the edge. Currently completing my thesis at Elmos Semiconductor on AI-driven regression-test optimization. See my full CV for details.

---

## 🛠️ Technical Skills

- **Languages:** Python · C++ · SystemVerilog · MATLAB  
- **Embedded & IoT:** NVIDIA Jetson Orin · ESP32 · FreeRTOS · TensorFlow Lite Micro  
- **ML & Signal Processing:** 1D-CNN/RNN · FFT feature extraction · Quantization · Adversarial-robust ML  
- **Tools & CI/CD:** GitLab CI · Docker · Bamboo · Git · Jira · Confluence  
- **Hardware & CAD:** eCadstar Zuken · Altium Designer · PCB layout · ISO 26262 compliance

---

## 📂 Projects

### **1. Unsupervised Learning of Parametric Optimization Problems**  
**Tools & Stack:** Python · CVXPY · TensorFlow/Keras · NumPy · scikit-learn

- **Problem:**  
  Approximate solutions to a convex, parameterized optimization:
  \[
    \min_{x,y,z} \frac{1}{xyz}
    \quad\text{s.t.}\;
    xy + yz + xz \le a,\;
    b\,y \le x,\;
    x,y,z > 0
  \]
- **Approach:**  
  - Generated 10 000+ \((a,b)\to(x,y,z)\) samples via CVXPY.  
  - **Supervised NN:** 3×900-node ReLU layers, L2 regularization, MSE loss.  
  - **Unsupervised NN:** 3×50-node ReLU layers, custom constraint-penalty loss.
- **Results:**  
  - **Supervised:** MSE ≈ 3.58 × 10⁻⁵  
  - **Unsupervised:** MSE ≈ 3.57 × 10⁻⁵  
  - Extensive hyperparameter grid-search (layers, neurons, activations, regularization).

---

### **2. Fall Detection & Emergency Alert System**  
**Stack:** ESP32 DevKitC · MPU6050 · TensorFlow Lite Micro · C/C++ · IFTTT/MQTT · tinyML

- **Objective:**  
  Real-time, on-device detection of free-fall + impact with automatic SOS alerts, implemented with tiny ML into the esp32 for Robotic fleet System
- **Approach:**  
  - Collected 400+ labeled 2 s windows (50 Hz) of accel/gyro data for “Fall” vs. “No-Fall.”  
  - Trained a lightweight 1D-CNN (Conv1D → Pooling → Dense) with >95 % validation accuracy.  
  - Post-training int8 quantization: model size ~60 KB, inference 30–50 ms/window.
- **Deployment Highlights:**  
  - Continuous I²C DMA sampling into a ring buffer.  
  - On “Fall,” ESP32 wakes Wi-Fi and POSTs a JSON alert to an IFTTT webhook (debounced).  
  - Full pipeline latency <2 s, fully on-device.
- **Impact:**  
  - **4×** smaller model footprint, **3×** faster inference vs. float baseline.  
  - Privacy-preserving, ultra-low latency for collison avoidance.

---

### **3. PDN Decoupling Capacitor Optimization (Multi-Agent RL)**  
**Stack:** Python · Ray RLlib (QMIX) · eCADSTAR PCB Editor · PI/EMI Analysis

- **Objective:**  
  Automate optimal placement of decoupling capacitors to meet impedance targets with minimal components.
- **Approach:**  
  - Custom MARL environment in Ray RLlib: two agents/place action & observation spaces.  
  - QMIX algorithm (mixing_embed_dim=32) for coordinated decisions.  
  - On-the-fly eCADSTAR PI/EMI simulations (via CSV) to compute reward = % frequencies under threshold + cap-usage bonus.
- **Results:**  
  - **12-port boards:** Converged in ~150 iterations to 1–2 caps achieving target impedance.  
  - **8-port boards:** Found optimal cap combos; highlighted hyperparameter tuning needs.  
  - PI/EMI plots confirm all optimal layouts stay below threshold.
- **Impact:**  
  Reduced manual trial-and-error by ~50%, enabling AI-driven PCB design automation.

---

### **4. Automated Regression-Test Log Clustering & Binning(Machine Learning)**  
**Stack:** Python · pandas · regex · TF-IDF · TruncatedSVD · HDBSCAN · KMeans

- **Objective:**  
  Streamline nightly CI by clustering UVM failure logs into meaningful bins for rapid triage.
- **Approach:**  
  - **Log Parsing:** Regex extraction of severity counts, module names, timestamps.  
  - **Feature Engineering:**  
    - TF-IDF on first UVM_ERROR messages.  
    - Counts of UVM_INFO/WARNING/ERROR, unique modules, error timings.  
  - **Hybrid Matrix:** 85 % TF-IDF + 15 % numeric features, normalized & stacked.  
  - **Clustering:**  
    - Truncated SVD → HDBSCAN for density-based clusters.  
    - KMeans sweep (k=2–10) optimizing ARI vs. ground truth.
- **Results:**  
  - HDBSCAN: high ARI & AMI, low noise.  
  - KMeans: strong silhouette scores for tight, separated clusters.
- **Impact:**  
  Reduced manual log-review effort by ~50 %, enabling automated CI binning.

---

## 🎓 Education

- **M.Sc. Automation & Robotics**, TU Dortmund   
- **B.Eng. Mechatronics**, Mumbai University 
