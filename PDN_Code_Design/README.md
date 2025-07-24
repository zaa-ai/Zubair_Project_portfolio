# PDN Decoupling Capacitor Optimization (Multi-Agent RL)

**Project Group WISE 2023/24**  
**Authors:** Darshana Anil Hosurkar, Emmanuel Hernandez, Zubair Ahmed Ansari  
**Supervisors:** Dr. Ing. Werner John, Prof. Dr. Ing. Jürgen Götze, M.Sc. Emre Ecik, M.Sc. Nima Ghafarian Shoaee, M.Sc. Julian Withöft  
**Date:** June 16, 2024  

---

## 🚀 Overview

This repository implements a QMIX-based Multi-Agent Reinforcement Learning (MARL) approach to automatically place decoupling capacitors on PCB Power Delivery Networks (PDNs), minimizing component count while satisfying impedance targets 

---

## 🛠️ Tech Stack

- **RL Framework:** [Ray RLlib (QMIX)](https://docs.ray.io/en/latest/rllib.html)  
- **Simulation:** eCADSTAR PI/EMI Analysis (CSV export)  
- **Language:** Python 3.8+  
- **Core Libraries:** `ray[rllib]`, `numpy`, `pandas`, `csv`

---

## Key Results
- 12-port PCB:

  - Converged in ~150 episodes to optimal layouts using just 1–2 capacitors.

  - Achieved target impedance with 50 % fewer manual trials. 

- 8-port PCB (single & mixed cap types):

  - Identified optimal capacitor combinations; highlighted the need for hyperparameter tuning in mixed-type scenarios.

## Analysis
- Reward Trajectories: steady rise toward convergence, with rectangular-board layouts consistently achieving higher rewards.

- Impedance Validation: all proposed layouts maintain impedance below the target threshold when re-simulated in eCADSTAR.

