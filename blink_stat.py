# MIT License
# 
# Copyright (c) 2024 Yu Bao, Harbin Institute of Technology
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os
import numpy as np
import matplotlib.pyplot as plt
import mne
from mne.preprocessing import ICA
from mne_icalabel import label_components
from scipy.signal import find_peaks

montage_file = 'montage.csv'

def load_edf_files(directory):
    raw_list = []
    file_names = []
    
    custom_montage = mne.channels.read_custom_montage(montage_file)
    for filename in os.listdir(directory):
        if filename.endswith(".edf"):
            file_path = os.path.join(directory, filename)
            raw = mne.io.read_raw_edf(file_path, preload=True)
            raw.set_montage(custom_montage)
            raw = raw.set_eeg_reference("average")
            raw_list.append(raw)
            if len(raw_list) >= 10086:
                break
            file_names.append(filename)
    return raw_list, file_names

def preprocess_data(raw):
    raw.filter(l_freq=1, h_freq=100)
    
    ica = ICA(n_components=30, random_state=97, max_iter="auto",method='infomax', fit_params=dict(extended=True))
    ica.fit(raw)
    
    ic_labels = label_components(raw, ica, method="iclabel")
    
    eyeblink_indices = [idx for idx, label in enumerate(ic_labels["labels"]) if label == "eye blink"]
    
    if not eyeblink_indices:
        print("No eyeblink component found. Using the component with highest eyeblink probability.")
        #eyeblink_indices = [np.argmax(ic_labels["y_pred_proba"][:, ic_labels["labels_unique"].index("eyeblink")])]
        print(ic_labels["y_pred_proba"]) # debug
        print(ic_labels["labels"])
    if "eye blink" in ic_labels["labels"]:
        # eyeblink_indices = [np.argmax(ic_labels["y_pred_proba"][:, ic_labels["labels"].index("eye blink")])]
        eyeblink_index = ic_labels["labels"].index("eye blink")
        eyeblink_proba = ic_labels["y_pred_proba"][eyeblink_index]
        eyeblink_indices = [np.argmax(eyeblink_proba)]
        eyeblink_components = ica.get_sources(raw)._data[eyeblink_indices]
    
        # If multiple eyeblink components, sum them
        if len(eyeblink_indices) > 1:
            eyeblink_component = np.sum(eyeblink_components, axis=0)
        else:
            eyeblink_component = eyeblink_components[0]
        
        return eyeblink_component
    else:
        print("no eyeblink IC detected")
        return 0

def detect_blinks(eyeblink_component, sfreq, threshold=1.0):
    peaks, _ = find_peaks(eyeblink_component, height=threshold, distance=int(0.5 * sfreq), prominence= 0.5, width=int(0.1 * sfreq))
    return peaks

def calculate_blink_frequency(blinks, duration):
    # Calculate blinks per minute
    blinks_per_minute = len(blinks) / (duration / 60)
    return blinks_per_minute

def analyze_blink_frequency_by_phase(raw, file_name):
    eyeblink_component = preprocess_data(raw)
    if eyeblink_component is None:
        return None

    events = mne.find_events(raw, stim_channel='Trigger')
    read_blinks, imagine_blinks, rest_blinks = [], [], []

    for event in events:
        if event[2] == 65380:  # Start of reading phase
            start_time = event[0] / raw.info['sfreq']
            read_end = start_time + 5
            imagine_end = read_end + 3.3
            rest_end = imagine_end + 1.8

            read_blinks.extend(detect_blinks(eyeblink_component[int(start_time*raw.info['sfreq']):int(read_end*raw.info['sfreq'])], raw.info['sfreq']))
            imagine_blinks.extend(detect_blinks(eyeblink_component[int(read_end*raw.info['sfreq']):int(imagine_end*raw.info['sfreq'])], raw.info['sfreq']))
            rest_blinks.extend(detect_blinks(eyeblink_component[int(imagine_end*raw.info['sfreq']):int(rest_end*raw.info['sfreq'])], raw.info['sfreq']))

    total_read_time = len(events) * 5 / 60  
    total_imagine_time = len(events) * 3.3 / 60
    total_rest_time = len(events) * 1.8 / 60

    read_frequency = len(read_blinks) / total_read_time if total_read_time > 0 else 0
    imagine_frequency = len(imagine_blinks) / total_imagine_time if total_imagine_time > 0 else 0
    rest_frequency = len(rest_blinks) / total_rest_time if total_rest_time > 0 else 0

    print(f"File: {file_name}")
    print(f"Read Phase: {read_frequency:.2f} blinks/minute")
    print(f"Imagine Phase: {imagine_frequency:.2f} blinks/minute")
    print(f"Rest Phase: {rest_frequency:.2f} blinks/minute")

    return read_frequency, imagine_frequency, rest_frequency

def analyze_all_files(raw_list, file_names):
    all_frequencies = []
    for raw, file_name in zip(raw_list, file_names):
        frequencies = analyze_blink_frequency_by_phase(raw, file_name)
        if frequencies:
            all_frequencies.append(frequencies)

    return all_frequencies

def calculate_average_frequencies(all_frequencies):
    if not all_frequencies:
        return None

    avg_read = np.mean([f[0] for f in all_frequencies])
    avg_imagine = np.mean([f[1] for f in all_frequencies])
    avg_rest = np.mean([f[2] for f in all_frequencies])

    print("\\nOverall Average Frequencies:")
    print(f"Read Phase: {avg_read:.2f} blinks/minute")
    print(f"Imagine Phase: {avg_imagine:.2f} blinks/minute")
    print(f"Rest Phase: {avg_rest:.2f} blinks/minute")

    return avg_read, avg_imagine, avg_rest

def visualize_average_frequencies(avg_frequencies):
    phases = ['Read', 'Imagine', 'Rest']
    plt.figure(figsize=(10, 6))
    plt.bar(phases, avg_frequencies)
    plt.ylabel('Average Blinks per Minute')
    plt.title('Average Eyeblink Frequency by Phase')
    for i, v in enumerate(avg_frequencies):
        plt.text(i, v, f'{v:.2f}', ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig("average_eyeblink_frequency_by_phase.png")
    plt.show()

# Main execution
if __name__ == "__main__":
    edf_directory = "edf_dir"
    raw_list, file_names = load_edf_files(edf_directory)
    all_frequencies = analyze_all_files(raw_list, file_names)
    avg_frequencies = calculate_average_frequencies(all_frequencies)
    if avg_frequencies:
        visualize_average_frequencies(avg_frequencies)

