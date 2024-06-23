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
import mne
import numpy as np
import pickle
from pyprep.find_noisy_channels import NoisyChannels
from pyprep.prep_pipeline import PrepPipeline
from mne_icalabel import label_components
import pandas as pd 
import argparse

import argparse

# If you DO NOT use any of the following arguments, the preprocessing will run with default settings.
# usage: preprocessing.py [-h] [--id ID] [--test] [--not_prep] [--not_reject] [--count_limit COUNT_LIMIT] [--method_str METHOD_STR] [--ica] [--ransac] [--step]
# options:
#   -h, --help            show this help message and exit
#   --id ID, -i ID        id attribute, default is A
#   --test, -t            Run test, only process part of the code
#   --not_prep, -p        Do not run prep
#   --not_reject, -a      Do not perform bad channel rejection
#   --count_limit COUNT_LIMIT, -c COUNT_LIMIT
#                         Maximum number of files to process before stopping, default is 10086
#   --method_str METHOD_STR, -m METHOD_STR
#                         Identifier for distinguishing different processing settings, default is undefined
#   --ica, -I             Run ICA
#   --ransac, -R          Use RANSAC in PREP
#   --step, -s            Save processed data after each step


parser = argparse.ArgumentParser(description='Options for preprocessing', formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('--id', '-i', help='id attribute, default is A', default='A')
parser.add_argument('--test', '-t', help='Run test, only process part of the code', action='store_true')
parser.add_argument('--not_prep', '-p', help='Do not run prep', action='store_false')
parser.add_argument('--not_reject', '-a', help='Do not perform bad channel rejection', action='store_false')
parser.add_argument('--count_limit', '-c', help='Maximum number of files to process before stopping, default is 10086', default=10086, type=int)
parser.add_argument('--method_str', '-m', help='Identifier for distinguishing different processing settings, default is undefined', default='undefined')
parser.add_argument('--ica', '-I', help='Run ICA', action='store_true')
parser.add_argument('--ransac', '-R', help='Use RANSAC in PREP', action='store_true')
parser.add_argument('--step', '-s', help='Save processed data after each step', action='store_true')
args = parser.parse_args()
print(args)

try:
    root_folder = '/' # default to “/”, you can change it to the path of the root folder of the dataset
    word_list_folder = os.path.join(root_folder, 'Chisco/textdataset')
    montage_file = 'montage.csv'
    IC_NUM = 30
    PATCH = False 


    subject_id = args.id 
    PREP = args.not_prep
    AUTO_REJECT = args.not_reject 
    method_str = args.method_str 
    ICA = args.ica 
    RANSAC = args.ransac 
    count_limit = args.count_limit 
    TEST = args.test 
    STEP = args.step
except Exception as e:
    print(e)
    
# other settings

useless_channels = ['11','110','EKG','EMG','84','85','10','111']
eog_channels = ['VEO','HEO']
sample_rate = 500
output_folder = os.path.join('preprocess_output', method_str + '_' + subject_id)

# select data folder by ID
if subject_id == 'A':
    data_folder = os.path.join(root_folder,'Chisco/sub-01')
elif subject_id == 'B':
    data_folder = os.path.join(root_folder,'Chisco/sub-02')
elif subject_id == 'C':
    data_folder = os.path.join(root_folder,'Chisco/sub-03')
else:
    print("Invalid subject id")
    exit()



if not os.path.exists(output_folder):
    os.makedirs(output_folder)
if not os.path.exists(os.path.join(output_folder,'pkl')):
    os.makedirs(os.path.join(output_folder,'pkl'))
if not os.path.exists(os.path.join(output_folder,'fif-epo')):
    os.makedirs(os.path.join(output_folder,'fif-epo'))
if not os.path.exists(os.path.join(output_folder,'fif')):
    os.makedirs(os.path.join(output_folder,'fif'))
if not os.path.exists(os.path.join(output_folder,'log')):
    os.makedirs(os.path.join(output_folder,'log'))


def process_edf_file(edf_file, montage_file, useless_channels, output_folder, word_list_folder):

    raw = mne.io.read_raw_edf(edf_file, preload=True)

    raw.resample(sample_rate)

    if TEST:
        raw.crop(tmin=300, tmax=600)

    custom_montage = mne.channels.read_custom_montage(montage_file) # import custom montage

    raw.set_montage(custom_montage)    
    
    raw.drop_channels(useless_channels)    # delete useless channels

    raw.set_channel_types({'VEO': 'eog', 'HEO': 'eog'})

    if PREP:
        # run pyprep
        print("Running pyprep")
        prep_params = {
            "ref_chs": "eeg",
            "reref_chs": "eeg",
            "line_freqs": np.arange(50, sample_rate / 2, 50),
        }

        prep = PrepPipeline(raw, prep_params, custom_montage, ransac=RANSAC)
        prep.fit()

        raw_new = prep.raw 

    else:
        print("Not running pyprep")
        raw_new = raw.copy()
    
    print("Still bad channels: ", raw_new.info['bads'])

    # High-pass filter
    raw_new.filter(l_freq=1, h_freq=None)

    if STEP:
        # save unsegmented processed data
        raw_new.save(os.path.join(output_folder,'fif' ,f"{os.path.basename(edf_file).replace('.edf', '')}_{subject_id}_{method_str}-raw.fif"), overwrite=True)

    # event detection
    events = mne.find_events(raw_new, stim_channel='Trigger')

    # keep events of id 65380
    events = events[events[:, 2] == 65380]

    # load CSV as metadata

    # find corresponding word list csv
    import re
    current_file = os.path.basename(edf_file)
    match = re.search(r'run-(\d+)', current_file)
    if match:
        run_number = int(match.group(1))
        print("Run number:", run_number)
    else:
        print("Error, Run number not found")

    csv_file = os.path.join(word_list_folder, f"split_data_{run_number}.xlsx")
    words_df = pd.read_excel(csv_file)
    words_list = words_df.iloc[:, 0].tolist() 
    print('Text list read, total:', len(words_list), "items")

    n_events = len(events)
    # If the word list is longer than the number of events, truncate it
    if len(words_list) > n_events:
        words_list = words_list[:n_events]
        print("Warning: The word list is longer than the number of events, truncating the word list")
    metadata = pd.DataFrame({'Word': words_list})

    # epoching, add metadata
    epochs_r = mne.Epochs(raw_new, events, tmin=0, tmax=5, baseline=None, preload=True, metadata=metadata)
    epochs_i = mne.Epochs(raw_new, events, tmin=5, tmax=8.3, baseline=None, preload=True, metadata=metadata)

    if AUTO_REJECT:
        from autoreject import AutoReject
        ar = AutoReject()
        epochs_clean_r, reject_log_r = ar.fit_transform(epochs_r, return_log=True)
        epochs_clean_i, reject_log_i = ar.fit_transform(epochs_i, return_log=True)
    else:
        print("Not running autoreject")
        epochs_clean_r = epochs_r
        epochs_clean_i = epochs_i
    
    if STEP:
        # Save the data after bad segment rejection as FIF files
        for (epoch, type_str) in [(epochs_clean_r, 'read'), (epochs_clean_i, 'imagine')]:
            save_epochs_to_fif(epoch, edf_file, type_str, '_rej')

        # Save the data after bad segment rejection as pickle files
        for (epoch, type_str) in [(epochs_clean_r, 'read'), (epochs_clean_i, 'imagine')]:
            save_epochs_to_pickle(epoch, edf_file, type_str, '_rej')

    # Start ICA artifact removal

    if ICA:
        ica_r = mne.preprocessing.ICA(n_components=30, random_state=97, max_iter="auto",method='infomax', fit_params=dict(extended=True)) # 使用extended-infomax算法
        ica_r.fit(epochs_clean_r)
        ic_labels_r = label_components(epochs_clean_r, ica_r, method="iclabel")
        labels_r = ic_labels_r["labels"]
        exclude_idx_r = [
            idx for idx, label in enumerate(labels_r) if label not in ["brain", "other"]
        ]
        print(f"Reading Epochs Excluding these ICA components: {exclude_idx_r}")
        epochs_clean_r_reconstructed = epochs_clean_r.copy()
        ica_r.apply(epochs_clean_r_reconstructed, exclude=exclude_idx_r)

        ica_i = mne.preprocessing.ICA(n_components=30, random_state=97, max_iter="auto",method='infomax', fit_params=dict(extended=True))
        ica_i.fit(epochs_clean_i)
        ic_labels_i = label_components(epochs_clean_i, ica_i, method="iclabel")
        labels_i = ic_labels_i["labels"]
        exclude_idx_i = [
            idx for idx, label in enumerate(labels_i) if label not in ["brain", "other"]
        ]
        print(f"Imagine Epochs Excluding these ICA components: {exclude_idx_i}")
        epochs_clean_i_reconstructed = epochs_clean_i.copy()
        ica_i.apply(epochs_clean_i_reconstructed, exclude=exclude_idx_i)
        
        if STEP:
            # Save the data after artifact removal as FIF files
            for (epoch, type_str) in [(epochs_clean_r_reconstructed, 'read'), (epochs_clean_i_reconstructed, 'imagine')]:
                save_epochs_to_fif(epoch, edf_file, type_str, '_rej_ica')

            # Save the data after artifact removal as pickle files
            for (epoch, type_str) in [(epochs_clean_r_reconstructed, 'read'), (epochs_clean_i_reconstructed, 'imagine')]:
                save_epochs_to_pickle(epoch, edf_file, type_str, '_rej_ica')

        print("ICA applied on", edf_file)
        
def save_epochs_to_fif(epochs, edf_file, type_str, extra_str=''):

    if not os.path.exists(os.path.join(output_folder,f'fif-epo{extra_str}')):
        os.makedirs(os.path.join(output_folder,f'fif-epo{extra_str}'))

    epochs.save(os.path.join(output_folder,f'fif-epo{extra_str}',f"{os.path.basename(edf_file).replace('.edf', '')}_{subject_id}_{type_str}_{method_str}{extra_str}-epo.fif"), overwrite=True)

def save_epochs_to_pickle(epochs, edf_file, type_str, extra_str=''):

    data_to_save = []
    for epoch_idx in range(len(epochs)):

        sentence = epochs.metadata.iloc[epoch_idx].iloc[0] if epochs.metadata is not None else None

        data = epochs[epoch_idx].get_data(copy=False)

        data_to_save.append({"text": sentence, "input_features": data})


    if not os.path.exists(os.path.join(output_folder,f'pkl{extra_str}')):
        os.makedirs(os.path.join(output_folder,f'pkl{extra_str}'))

    with open(os.path.join(output_folder,f'pkl{extra_str}',f"{os.path.basename(edf_file).replace('.edf', '')}_{subject_id}_epochs_{type_str}_{method_str}{extra_str}.pkl"), 'wb') as file:
        pickle.dump(data_to_save, file)


# Traverse all .edf files in the folder
if os.listdir(data_folder) == [] or not os.path.exists(data_folder):
    print("No files in the folder or the folder does not exist")
    exit()

count = 0  # Limit the number of files processed

for root, dirs, files in os.walk(data_folder):
    # Sort files by leading number in the file name
    sorted_files = sorted(files, key=lambda x: int(x.split(' ')[0]) if x.split(' ')[0].isdigit() else float('inf'))
    
    for file in sorted_files:
        count += 1
        if count > count_limit:
            break
        if PATCH and os.path.exists(os.path.join(output_folder, 'pkl', f"{os.path.basename(file).replace('.edf', '')}_{subject_id}_epochs_read_{method_str}.pkl")):
            print(os.path.basename(file), "already processed, skipping")
            continue
        if file.endswith('.edf'):
            print("********Processing file: ", file, "********")
            edf_file_path = os.path.join(root, file)
            process_edf_file(edf_file_path, montage_file, useless_channels, output_folder, word_list_folder)
