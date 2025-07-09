# Computational Model for Brain-Heart Interaction Analysis

This repository presents a **cross-modal machine learning pipeline** designed to explore brain-heart interactions for **cognitive state classification**, leveraging **EEG**, **ECG**, and **synthetic physiological signals**. It demonstrates how **ECG-derived features** can approximate EEG cognitive markers, enabling **non-invasive mental state monitoring** using wearables.

---

##  **Project Highlights**

- ✅ **Multimodal Dataset**: EEG, ECG, and physiological signals recorded during memory and listening tasks from OpenNeuro.
- ✅ **Low-Cost Cognitive Monitoring**: Demonstrates potential of ECG alone to infer cognitive states — ideal for wearable deployment.
- ✅ **Synthetic Data Generation**: Uses the **Poincaré Sympathetic-Vagal Synthetic Data Generation (SV-SDG)** model to simulate HRV signals influenced by EEG.
- ✅ **Cross-Modal Learning**: Maps ECG-derived HRV features to EEG space, enabling classification without EEG.
- ✅ **Model Explainability**: SHAP and t-SNE visualizations provide transparency and interpretation.

---

##  **Working Pipeline**

### ✅ 1. **Data Acquisition**
- OpenNeuro dataset with EEG, ECG, and physiological recordings during cognitive tasks (memory & listening).

### ✅ 2. **Preprocessing**
- **EEG**:
  - Filtering
  - Epoching
  - Baseline correction using MNE
- **ECG**:
  - R-peak detection
  - Epoching
  - Signal normalization using NeuroKit2

### ✅ 3. **Feature Extraction**
- **EEG Features**:
  - Spectral Bandpower (Delta, Theta, Alpha, Beta, Gamma)
  - Functional connectivity measures
  - Catch-22 time series descriptors

- **ECG Features**:
  - HRV Metrics: `meanNN`, `SDNN`, `RMSSD`, `SD1`, `SD2`
  - Catch-22 descriptors

### ✅ 4. **Synthetic Data Generation**
- **SV-SDG Model**:
  - Generates synthetic HRV signals using Poincaré plot dynamics.
  - Simulates vagal/sympathetic influence from EEG.

### ✅ 5. **Cross-Modal Learning Framework**
- **EEG-Based Classifiers**:
  - Random Forest and XGBoost models trained to classify cognitive states.
- **ECG-to-EEG Mapping**:
  - Regression models predict EEG features from ECG HRV features.
- **ECG-Only Classification**:
  - Map ECG into EEG space and classify cognitive states using EEG-trained models.

### ✅ 6. **Model Evaluation & Explainability**
- Classification metrics: Accuracy and F1-score for binary & multi-class tasks.
- Explainability:
  - **SHAP**: Feature importance
  - **t-SNE**: Visualizing feature clusters

---

##  **Key Results**

- High classification accuracy using EEG features.
- Strong regression performance mapping ECG to EEG space.
- Promising ECG-only classification — supports future wearable applications.

---

##  **Installation & Requirements**

Install required dependencies:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost shap mne neurokit2
