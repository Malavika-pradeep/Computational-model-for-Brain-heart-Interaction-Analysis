
# ✅ Install required packages (if not already installed)  
!pip install mne neurokit2 matplotlib tqdm --quiet  
!pip install pymatreader h5py mne neurokit2  

# ✅ Import libraries  
import os  
import mne  
import neurokit2 as nk  
import numpy as np  
import matplotlib.pyplot as plt  
import pickle  
import gc  

SAVE_PLOTS = False  # Set to True to save EEG plots  
dataset_path = "/content/drive/MyDrive/EntireDataset_240GB"  
output_root = "/content/output_eeg_data"  
output_images_dir = "/content/output_eeg_images"  
os.makedirs(output_root, exist_ok=True)  
if SAVE_PLOTS:  
    os.makedirs(output_images_dir, exist_ok=True)  


num_conditions = 3  
num_subconditions = 3  
num_trials = 54  
max_time = 26000  

# ✅ Event dictionary (same as before)  
event_dicts = {  
    "five": {  
        "justlisten/five": 1,  
        "memory/correct/five": 29,  
        "memory/incorrect/five": 28  
    },  
    "nine": {  
        "justlisten/nine": 2,  
        "memory/correct/nine": 31,  
        "memory/incorrect/nine": 30  
    },  
    "thirteen": {  
        "justlisten/thirteen": 3,  
        "memory/correct/thirteen": 33,  
        "memory/incorrect/thirteen": 32  
    }  
}  

epoch_durations = {"five": 10, "nine": 18, "thirteen": 26}  
reject_criteria = dict(eeg=500e-6)  # More lenient rejection threshold for EEG  

# ✅ EEG-specific parameters  
eeg_channels = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T7', 'C3', 'Cz',   
                'C4', 'T8', 'P7', 'P3', 'Pz', 'P4', 'P8', 'O1', 'O2']  # Standard 10-20 system  

# ✅ Load the subject's EEG data  
try:  
    print(f"\n🔍 Processing {subject_id}...")  
    raw = mne.io.read_raw_eeglab(file_path, preload=True)  
    
    # Set channel types and select EEG channels  
    raw.set_channel_types({ch: 'eeg' for ch in eeg_channels if ch in raw.ch_names})  
    raw.pick_types(eeg=True)  # Only keep EEG channels  

    # Apply bandpass filter (1-40 Hz typical for EEG)  
    raw.filter(1, 40, fir_design='firwin')  
    
    # Apply common average reference  
    raw.set_eeg_reference('average')  
    
    events, _ = mne.events_from_annotations(raw)  

    # Preallocate subject 5D array (condition × subcondition × trial × channel × time)  
    subject_data = np.full((num_conditions, num_subconditions, num_trials, len(eeg_channels), max_time),   
                          np.nan, dtype=np.float32)  

    # Loop through each condition  
    for condition_idx, (condition, event_dict) in enumerate(event_dicts.items()):  
        available_event_ids = set(events[:, 2])  
        filtered_event_dict = {k: v for k, v in event_dict.items() if v in available_event_ids}  
        if not filtered_event_dict:  
            print(f"⚠️ Skipping condition '{condition}' for {subject_id}: No matching events.")  
            continue  

        try:  
            epochs = mne.Epochs(  
                raw, events, tmin=-3, tmax=epoch_durations[condition],  
                event_id=filtered_event_dict, preload=True,  
                reject=reject_criteria, baseline=(-3, 0)  # Added baseline correction  
            )  
        except Exception as e:  
            print(f"❌ [{subject_id}] Epoching failed: {e}")  
            continue  

        # Process trials  
        for trial_idx in range(min(num_trials, len(epochs))):  
            try:  
                selected_epoch = epochs[trial_idx]  
                eeg_data = selected_epoch.get_data()  # Shape: (channels, time)  
                
                # Skip if data is mostly NaN or flat  
                if np.isnan(eeg_data).all() or np.all(np.std(eeg_data, axis=1) < 1e-6):  
                    continue  

                # Event and subcondition mapping  
                event_code = epochs.events[trial_idx, 2]  
                if event_code in event_dict.values():  
                    event_label = list(event_dict.keys())[list(event_dict.values()).index(event_code)]  
                    subcondition_idx = 0 if "justlisten" in event_label else (1 if "memory/correct" in event_label else 2)  
                else:  
                    continue  

                # Handle variable time lengths  
                time_length = eeg_data.shape[1]  
                if time_length > max_time:  
                    eeg_data_trimmed = eeg_data[:, :max_time]  
                else:  
                    padding = [(0, 0), (0, max_time - time_length)]  
                    eeg_data_trimmed = np.pad(eeg_data, padding, constant_values=np.nan)  

                subject_data[condition_idx, subcondition_idx, trial_idx, :, :] = eeg_data_trimmed.astype(np.float32)  

                # Save EEG plot (topomap example)  
                if SAVE_PLOTS and trial_idx % 10 == 0:  # Save every 10th trial to reduce files  
                    fig = plt.figure(figsize=(12, 6))  
                    plt.plot(eeg_data.T)  
                    plt.title(f"{subject_id} {condition} Trial {trial_idx}")  
                    plot_path = os.path.join(output_images_dir, f"{subject_id}_{condition}_{trial_idx}.png")  
                    fig.savefig(plot_path)  
                    plt.close(fig)  

            except Exception as e:  
                print(f"⚠️ [{subject_id}] Trial {trial_idx} processing failed: {e}")  
                continue  

    # Clean up memory  
    del raw, epochs  
    gc.collect()  

    # ✅ Final 5D data array (condition × subcondition × trial × channel × time)  
    final_data_array = subject_data  

    # Save as pickle  
    pickle_filename = os.path.join(output_root, f"processed_{subject_id}_eeg_data.pkl")  
    with open(pickle_filename, "wb") as f:  
        pickle.dump({  
            "subject": subject_id,   
            "data": final_data_array,  
            "channels": eeg_channels  
        }, f)  

    print("\nEEG Processing Complete!")  
    print(f"Data shape: {final_data_array.shape}")  
    print(f"Saved to: {pickle_filename}")  

except Exception as e:  
    print(f"❌ Failed to process {subject_id}: {e}")  