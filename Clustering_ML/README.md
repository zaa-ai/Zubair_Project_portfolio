# UVM Regression-Test Log Clustering & Binning

Automate CI triage by clustering UVM failure logs into meaningful bins for rapid investigation.

---

## 🔍 Project Overview

When nightly regression tests produce thousands of logs, manually sorting them is time-consuming.  
This pipeline:

- Parses raw UVM/SystemVerilog log files  
- Extracts numeric (severity counts, module counts, timing) and text (error messages) features  
- Builds a hybrid feature matrix (85% TF-IDF text + 15% numeric)  
- Reduces dimensions via Truncated SVD  
- Clusters with HDBSCAN (or KMeans)  
- Outputs labeled clusters for automated CI binning  

---

## ⚙️ Prerequisites

- Python 3.8+  
- Dependencies listed in `requirements.txt`

---

## 🚀 Installation

1. **Clone the repo**  
   ```bash
   git clone <your-repo-url>
   cd <repo-dir>

## Results
- Clusters are printed to console and can be mapped back to each log file for CI integration.

- To evaluate, hook in your ground-truth labels and enable evaluate_clustering() calls in main().