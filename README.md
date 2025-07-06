# Computational-model-for-Brain-heart-Interaction-Analysis

This repository presents a cross-modal machine learning pipeline designed to explore brain-heart interactions for cognitive state classification, leveraging EEG, ECG, and synthetic physiological signals. The project demonstrates how ECG-derived features can approximate EEG cognitive markers, enabling non-invasive mental state monitoring via wearable devices.




🚀 Project Highlights

•Multimodal Dataset: EEG, ECG, and other physiological signals recorded during memory and listening tasks from OpenNeuro.<br/>
•Low-Cost Cognitive Monitoring: Demonstrates potential of ECG alone to infer cognitive states, paving the way for wearable mental health monitoring.<br/>
•Synthetic Data Generation: Poincaré Sympathetic-Vagal Synthetic Data Generation (SV-SDG) model simulates realistic HRV signals influenced by brain activity.<br/>
• Cross-Modal Learning Framework: Maps ECG features to EEG space, enabling cognitive state classification using only ECG signals.<br/>
•Model Explainability: SHAP and t-SNE visualizations enhance transparency and interpretability.<br/>




🛠️ Working Pipeline

✅ Data Acquisition
OpenNeuro dataset containing EEG, ECG, and physiological recordings from subjects performing memory and listening tasks.

✅ Preprocessing
EEG: Filtering, epoching, baseline correction using MNE.<br/>
ECG: R-peak detection, epoching, normalization via NeuroKit2.<br/>

✅ Feature Extraction
EEG Features:
•Spectral Bandpower (Delta, Theta, Alpha, Beta, Gamma bands)<br/>
•Connectivity measures<br/>
•Catch-22 time series descriptors<br/>
ECG Features:
•Heart Rate Variability (HRV): meanNN, SDNN, RMSSD, SD1, SD2<br/>
•Catch-22 time series descriptors<br/>

✅ Synthetic Data Generation: SV-SDG Model
•Generates synthetic HRV signals using Poincaré plot dynamics.<br/>
•Models cardiac responses influenced by EEG-derived sympathetic-vagal activity.<br/>

✅ Cross-Modal Learning Framework
•EEG-Based Classifiers: Random Forest and XGBoost trained on EEG features to classify cognitive states (e.g., Listening vs Memory Load tasks).<br/>
•ECG-to-EEG Mapping: Regression models predict EEG feature space from ECG-derived HRV features.<br/>
•ECG-Only Cognitive Classification: Project ECG features into EEG space and classify cognitive states using EEG-trained models.<br/>

✅ Model Evaluation & Explainability
•Accuracy and F1-score for binary and multi-class classification tasks.<br/>
•SHAP for feature importance interpretation.<br/>
•t-SNE for visualizing feature space separability.<br/>




📊 Key Result

•High classification accuracy using EEG features for cognitive state detection.<br/>
•Demonstrated feasibility of predicting EEG features from ECG-derived HRV metrics.<br/>
•ECG-only cognitive state classification shows promising results with potential for real-world, wearable deployment.<br/>



📑 Reference

This project is part of my M.Tech thesis at the Kerala University of Digital Sciences, Innovation and Technology, titled:

Computational Model for Brain-Heart Interaction Analysis
Malavika Pradeep, 2025 
