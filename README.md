# Manual MLOps Pipeline – Assignment 1

**Course:** DA5402 – MLOps  
**Student Name:** Shlok Shetty  
**Roll Number:** MM22B003  

---

## Video Submission

[Manual MLOps Pipeline – Assignment 1 (Google Drive)](https://drive.google.com/drive/folders/1j2ULviIEhPvqRfPimoATgckVRMDRervZ?usp=drive_link)

---

## Overview

This repository contains the implementation of **Assignment 1**, which focuses on building a **manual end-to-end MLOps pipeline** without using automated MLOps tools such as **MLflow** or **DVC**.

The objective of this assignment is to understand the practical challenges involved in managing **data**, **models**, **deployment**, and **monitoring** manually, and to appreciate the importance of automated MLOps systems in real-world machine learning workflows.

---

## Problem Statement

The task is to build a **Predictive Maintenance system** using the **AI4I 2020 dataset**.  
The system predicts whether a machine will fail based on sensor readings and operational parameters.

Beyond model training, the assignment emphasizes:

- Data versioning  
- Model versioning  
- Deployment  
- Monitoring  
- Drift simulation  
- Retraining logic  

---

## Key Components Implemented

### 1. Data Versioning (Manual)

- Raw data stored separately from processed data  
- Multiple data versions (`v1`, `v2`, `v3`) created based on preprocessing changes  
- Each data version is documented using a `manifest.txt` file that records:
  - Input dataset  
  - Preprocessing steps  
  - Output files  
  - Target variable  

---

### 2. Model Training and Versioning

- Models trained using different configurations and data versions  
- Each trained model is saved as a versioned artifact (`model_vX.pkl`)  
- For every model version, a corresponding metadata JSON file is created containing:
  - Data version used  
  - Hyperparameters  
  - Training and test metrics  
  - Training timestamp  
  - Git commit hash  

---

### 3. Configuration Management

- A central `config.yaml` file is used as the single source of truth  
- Controls:
  - Data version selection  
  - Model versioning  
  - Deployment settings  
  - Monitoring thresholds  
- Ensures reproducibility and prevents accidental mismatches between data, model, and deployment  

---

### 4. Model Deployment

- Model deployed locally using **FastAPI**  
- The inference service dynamically loads the active model version specified in `config.yaml`  
- Each deployment event is logged in `deployment_log.csv`  
- API provides:
  - Health check endpoint  
  - Prediction endpoint with probability and threshold-based decision  

---

### 5. Prediction Logging and Monitoring

- All inference requests are logged in `prediction_log.json`  
- A monitoring script compares:
  - Production performance  
  - Baseline metrics stored in model metadata  
- Metrics monitored include:
  - Error rate  
  - Recall (critical for predictive maintenance)  
- Retraining is triggered when performance degradation exceeds configured thresholds  

---

### 6. Drift Simulation and Retraining

- A simulated **Day-2 dataset** is created by modifying sensor distributions  
- Drifted data is passed through the deployed API  
- Monitoring detects degradation and triggers retraining  
- Drifted data is appended to the training data  
- A new model version is trained and redeployed  
- Post-retraining stability is verified using monitoring and smoke tests  

---

## Project Structure

├── config.yaml
├── README.md
├── data/
│ ├── raw/
│ ├── processed/
│ └── production/
├── models/
│ ├── model_v*.pkl
│ └── metadata_v*.json
├── src/
│ ├── data_prep.py
│ ├── train.py
│ ├── inference.py
│ ├── monitor.py
│ ├── run_day2_inference.py
│ └── smoke_tests.py
├── deployment_log.csv
└── prediction_log.json


---

## Learning Outcomes

This assignment highlights:

- How difficult manual MLOps becomes as system complexity increases  
- How easy it is to lose track of data and model lineage without tooling  
- Why automated MLOps platforms are essential in production systems  

The implementation demonstrates the complete machine learning lifecycle — from raw data to deployment, monitoring, and retraining — entirely managed manually.

---

## Notes

- No automated MLOps tools were used  
- All versioning, tracking, and monitoring logic was implemented explicitly  
- The focus of this assignment is **process correctness and reproducibility**, not metric optimization  
