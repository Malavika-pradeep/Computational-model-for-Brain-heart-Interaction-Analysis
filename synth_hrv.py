!pip install neurokit2

import pickle
import neurokit2 as nk
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Load your raw ECG pickle (assume shape: [n_trials, signal_length])
with open("/content/drive/MyDrive/processed_ecg_data.pkl", "rb") as f:
    ecg_trials = pickle.load(f)




# 1) pick the flattened trial
raw_ecg_trial = ecg_trials['data'][0, 0, 0, :].flatten()

# 2) find the first & last non-NaN sample
valid_idx = np.where(~np.isnan(raw_ecg_trial))[0]
start, end = valid_idx[0], valid_idx[-1] + 1

# 3) crop to the valid window
ecg_cropped = raw_ecg_trial[start:end]
print(f"Using samples {start}–{end} (length {len(ecg_cropped)} samples)")

# 1. Get the flattened trial with NaNs
trial = ecg_trials['data'][0, 0, 0, :].flatten()

# 2. Remove *all* NaNs, collapsing the valid segments together
valid_signal = trial[~np.isnan(trial)]
print(f"Original length: {len(trial)}, After dropping NaNs: {len(valid_signal)}")

# 3. (Optional) If it’s still too short, skip this trial
if len(valid_signal) < 5 * fs:  # e.g. require at least 5 seconds of data
    raise ValueError("Trial too short after NaN removal.")

# 4. Clean and detect R-peaks on the fully concatenated valid signal
ecg_clean = nk.ecg_clean(valid_signal, sampling_rate=fs)
_, info = nk.ecg_peaks(ecg_clean, sampling_rate=fs)
rpeaks = info["ECG_R_Peaks"]
print("Detected R-peaks:", len(rpeaks), "→ indices (first 10):", rpeaks[:10])

# 5. Compute RR intervals (s) and review mean HR
rr = np.diff(rpeaks) / fs
print("First 10 RR (s):", rr[:10])
print(f"Mean RR: {np.mean(rr):.3f} s  →  HR ≃ {60/np.mean(rr):.1f} bpm")

# --- 3) Compute sliding Poincaré SD1 & SD2 ---
def poincare_sd1_sd2(rr, window_beats=60):
    sd1, sd2 = [], []
    for i in range(len(rr) - window_beats + 1):
        seg = rr[i:i+window_beats]
        diff = np.diff(seg)
        SD1 = np.sqrt(np.var(diff)/2)
        SD2 = np.sqrt(2*np.var(seg) - np.var(diff)/2)
        sd1.append(SD1)
        sd2.append(SD2)
    return np.array(sd1), np.array(sd2)

sd1, sd2 = poincare_sd1_sd2(rr, window_beats=60)

# --- 4) Build modulation signal m(t) at 4 Hz ---
fs_mod = 4
t_mod = np.arange(len(sd1)) / fs_mod
wS, wV = 2*np.pi*0.1, 2*np.pi*0.25
m = sd2 * np.sin(wS*t_mod) + sd1 * np.sin(wV*t_mod)

# --- 5) Generate synthetic RR intervals via IPFM ---
def generate_synthetic_rr(duration_s, m, fs_mod, hr_bpm):
    dt = 1.0/fs_mod
    mu_hr = hr_bpm/60.0
    integral, t, idx = 0.0, 0.0, 0
    rr_times = []
    while t < duration_s:
        integral += (mu_hr + (m[idx] if idx<len(m) else 0))*dt
        if integral >= 1.0:
            rr_times.append(t)
            integral -= 1.0
        t += dt; idx += 1
    return np.diff(rr_times)

duration = len(valid_signal)/fs
hr_bpm = 60/np.mean(rr)
synthetic_rr = generate_synthetic_rr(duration, m, fs_mod, hr_bpm)

# --- 6) Convert to milliseconds & view ---
synthetic_rr_ms = synthetic_rr * 1000
print("Synthetic RR (ms):", synthetic_rr_ms[:10])

# --- 7) Reconstruct a simple synthetic ECG waveform ---
from scipy.signal import resample
def build_ecg_from_rr(rr_ms, fs=250, qrs_ms=100):
    qrs_len = int(qrs_ms*fs/1000)
    t = np.linspace(-1,1,qrs_len)
    template = np.exp(-t**2*20)
    total = int(np.sum(rr_ms)*fs/1000) + qrs_len
    ecg = np.zeros(total); idx=0
    for rr in rr_ms:
        idx += int(rr*fs/1000)
        if idx+qrs_len<total:
            ecg[idx:idx+qrs_len] += template
    return ecg

ecg_synth = build_ecg_from_rr(synthetic_rr_ms, fs=250)
plt.figure(figsize=(8,2))
plt.plot(ecg_synth[:2000])
plt.title("Synthetic ECG from SV-SDG (first 8s @250Hz)")
plt.show()

import numpy as np
import pandas as pd
import neurokit2 as nk

# --- Helper functions (unchanged) ---
def compute_rr(ecg_signal, fs):
    ecg_clean = nk.ecg_clean(ecg_signal, sampling_rate=fs)
    _, info = nk.ecg_peaks(ecg_clean, sampling_rate=fs)
    rpeaks = info["ECG_R_Peaks"]
    return np.diff(rpeaks) / fs  # in seconds

def poincare_sd1_sd2(rr, window_beats=10):
    sd1, sd2 = [], []
    for i in range(len(rr) - window_beats + 1):
        seg = rr[i:i+window_beats]
        diff = np.diff(seg)
        sd1.append(np.sqrt(np.var(diff)/2))
        sd2.append(np.sqrt(2*np.var(seg) - np.var(diff)/2))
    return np.array(sd1), np.array(sd2)

def svsdg_modulation(sd1, sd2, fs_mod=4):
    t = np.arange(len(sd1)) / fs_mod
    wS, wV = 2*np.pi*0.1, 2*np.pi*0.25
    return sd2 * np.sin(wS*t) + sd1 * np.sin(wV*t)

def generate_synthetic_rr(duration_s, m, fs_mod, hr_bpm):
    dt = 1.0/fs_mod
    mu_hr = hr_bpm/60.0
    integral, t, idx = 0.0, 0.0, 0
    rr_times = []
    while t < duration_s:
        integral += (mu_hr + (m[idx] if idx < len(m) else 0)) * dt
        if integral >= 1.0:
            rr_times.append(t)
            integral -= 1.0
        t += dt; idx += 1
    return np.diff(rr_times)  # in seconds

def compute_hrv_metrics(rr_s):
    rr_ms = rr_s * 1000
    meanNN = rr_ms.mean()
    SDNN   = rr_ms.std(ddof=1)
    RMSSD  = np.sqrt(np.mean(np.diff(rr_ms)**2))
    SD1    = np.sqrt(np.var(np.diff(rr_ms))/2)
    SD2    = np.sqrt(2*np.var(rr_ms) - np.var(np.diff(rr_ms))/2)
    return meanNN, SDNN, RMSSD, SD1, SD2

# --- Parameters ---
fs           = 1000
window_beats = 10   # shorter window
fs_mod       = 4

records = []

# Unpack your data
ecg_array = ecg_trials['data']  # shape: (subjects, cond, subcond, trials, samples)

n_subj, n_cond, n_subc, n_trial, _ = ecg_array.shape

for subj in range(n_subj):
    for cond in range(n_cond):
        for subc in range(n_subc):
            for t_idx in range(n_trial):
                trial = ecg_array[subj, cond, subc, t_idx, :].flatten()
                valid = trial[~np.isnan(trial)]
                if len(valid) < fs:  # skip if <1 second valid
                    continue

                rr = compute_rr(valid, fs)
                if len(rr) < window_beats + 1:
                    continue

                sd1, sd2 = poincare_sd1_sd2(rr, window_beats)
                m       = svsdg_modulation(sd1, sd2, fs_mod)

                duration = len(valid)/fs
                hr_bpm   = 60 / np.mean(rr)

                synth_rr = generate_synthetic_rr(duration, m, fs_mod, hr_bpm)
                if len(synth_rr) < 1:
                    continue

                metrics = compute_hrv_metrics(synth_rr)
                records.append({
                    "subject": subj,
                    "condition": cond,
                    "subcondition": subc,
                    "trial": t_idx,
                    "meanNN": metrics[0],
                    "SDNN":   metrics[1],
                    "RMSSD":  metrics[2],
                    "SD1":    metrics[3],
                    "SD2":    metrics[4]
                })

df_synth_hrv = pd.DataFrame.from_records(records)
print("Generated synthetic HRV for", len(df_synth_hrv), "trials")

# Save
df_synth_hrv.to_pickle("synthetic_hrv_dataset.pkl")



# Load synthetic HRV
df_synth = pd.read_pickle("/content/synthetic_hrv_dataset.pkl")
print(df_synth.shape)
df_synth.head()

