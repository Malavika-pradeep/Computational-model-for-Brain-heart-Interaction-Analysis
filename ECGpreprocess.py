import os
import mne
import neurokit2 as nk
import numpy as np
import pickle

# Define paths
dataset_path = "path_to_your_ecg_dataset"  # Replace with your dataset path
output_root = "output_ecg_data"
output_images_dir = "output_ecg_images"
os.makedirs(output_root, exist_ok=True)
os.makedirs(output_images_dir, exist_ok=True)

# Define processing parameters
num_conditions = 3  # Five, Nine, Thirteen
num_subconditions = 3  # Just Listen, Memory Correct, Memory Incorrect
num_trials = 54  # Per condition
max_time = 26000  # Max duration in ms (26s * 1000Hz)

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



# Epoch rejection criteria
reject_criteria = dict(ecg=1000000e-6)
epoch_durations = {"five": 10, "nine": 18, "thirteen": 26} # Duration of each condition's epochs (in seconds)

# Subjects to exclude
excluded_subjects = {"sub-013", "sub-014", "sub-015", "sub-016","sub-017", "sub-018", "sub-019", "sub-020", "sub-021", "sub-022", "sub-023", "sub-024", "sub-025", "sub-026", "sub-027", "sub-028", "sub-029", "sub-030", "sub-031", "sub-037", "sub-066"}

# Find all .set files in the dataset folder
def find_set_files(root_dir):
    set_files = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".set") and "task-memory_ecg" in file:
                set_files.append(os.path.join(root, file))
    return set_files


set_files = find_set_files(dataset_path) # Get list of all ECG .set files

# Initialize storage for processed data
subject_list = []
data_array = []

# Process each .set file
for file_path in set_files:
    subject_id = os.path.basename(file_path).split("_")[0]
    if subject_id in excluded_subjects:
        continue

    print(f"Processing {subject_id}...")
    raw = mne.io.read_raw_eeglab(file_path, preload=True)
    raw.set_channel_types({"ECG": "ecg"})
    events, _ = mne.events_from_annotations(raw)

    # Create subject's data array
    subject_data = np.full((num_conditions, num_subconditions, num_trials, max_time), np.nan)

    for condition_idx, (condition, event_dict) in enumerate(event_dicts.items()):
        events, event_id_map = mne.events_from_annotations(raw)

# Get available event codes for this subject (as integers)
        available_event_ids = set(events[:, 2])

# Filter out only the event codes in this condition that are actually present
        filtered_event_dict = {k: v for k, v in event_dict.items() if v in available_event_ids}

# Skip if none of the condition's event codes are in the data
        if not filtered_event_dict:
            print(f"Skipping condition '{condition}' for {subject_id}: No matching events found.")
            continue
# Filter out only the event codes in this condition that are actually present
       
        epochs = mne.Epochs(raw, events, tmin=-3, tmax=epoch_durations[condition],
                             event_id=event_dict, preload=True, reject=reject_criteria, baseline= None)

        for trial_idx in range(min(num_trials, len(epochs))):
            selected_epoch = epochs[trial_idx]
            ecg_signal = selected_epoch.get_data(picks="ECG")[0, 0, :]
            
            # Skip if the signal is too short, empty, or full of NaNs
            if np.isnan(ecg_signal).all() or len(ecg_signal) < 1000 or np.std(ecg_signal) < 1e-6:
                print(f"Skipping trial {trial_idx}: ECG data invalid or flat for {subject_id} - {condition}")
                continue
            # Process ECG data
            ecg_data = np.squeeze(ecg_signal)
            try:
                if ecg_data is None or np.isnan(ecg_data).any() or np.std(ecg_data) < 1e-5:
                    print(f"[{subject_id}] ECG data invalid or flat. Skipping.")
                    continue

                ecg_cleaned = nk.ecg_clean(ecg_data, sampling_rate=1000)

    # ✅ Proper unpacking
                _, rpeaks_dict = nk.ecg_peaks(ecg_cleaned, sampling_rate=1000)
                rpeaks = rpeaks_dict["ECG_R_Peaks"]

                if len(rpeaks) < 3:
                    print(f"[{subject_id}] Too few R-peaks detected ({len(rpeaks)}). Skipping.")
                    continue

                signals, info = nk.ecg_process(ecg_data, sampling_rate=1000)
                print(f"[{subject_id}] ECG processed successfully!")

            except Exception as e:
                print(f"[{subject_id}] Error during ECG processing: {e}")
                continue


            # Identify subcondition index
            event_code = epochs.events[trial_idx, 2]
            if event_code in event_dict.values():
                event_label = list(event_dict.keys())[list(event_dict.values()).index(event_code)]
                subcondition_idx = 0 if "justlisten" in event_label else (1 if "memory/correct" in event_label else 2)
            else:
                print(f"Skipping trial {trial_idx}: Unexpected event code {event_code} for {subject_id} - {condition}")
                continue

            # Ensure uniform shape
            time_length = len(ecg_data)
            if time_length > max_time:
                ecg_data = ecg_data[:max_time]
            else:
                ecg_data = np.pad(ecg_data, (0, max_time - time_length), constant_values=np.nan)

            subject_data[condition_idx, subcondition_idx, trial_idx, :] = ecg_data

            # Save ECG plots using NeuroKit2, ensuring the plot is valid
            plot = nk.ecg_plot(signals, info)
            if plot:
                plot.savefig(os.path.join(output_images_dir, f"{subject_id}_{condition}_{trial_idx}.png"))
                plot.close()
            else:
                print(f"Skipping plot saving for {subject_id} - {condition} - Trial {trial_idx} (No valid plot generated)")

    # Store processed data
    subject_list.append(subject_id)
    data_array.append(subject_data)

# Convert to NumPy array
final_data_array = np.array(data_array)

# Save as pickle file
pickle_filename = os.path.join(output_root, "processed_ecg_data.pkl")
with open(pickle_filename, "wb") as f:
    pickle.dump({"subjects": subject_list, "data": final_data_array}, f)

print("Processing complete. ECG data stored in 'processed_ecg_data.pkl'.")