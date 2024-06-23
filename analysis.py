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

import mne
import os
import numpy as np
import math
import matplotlib.pyplot as plt

root_folder = '/' # default to “/”, you can change it to the path of the root folder of the dataset
folder_path_A = os.path.join(root_folder, 'Chisco/derivatives/preprocessed_fif/sub-01')
folder_path_B = os.path.join(root_folder, 'Chisco/derivatives/preprocessed_fif/sub-02')
folder_path_C = os.path.join(root_folder, 'Chisco/derivatives/preprocessed_fif/sub-03')

A_exclude_list = []
B_exclude_list = ['run-45']
C_exclude_list = ["run-09", "run-041"]

# Get all .fif files from the three subjects' corresponding folders
fif_files_A = []
for root, dirs, files in os.walk(folder_path_A):
    for file in files:
        # If the file ends with .fif and the filename does not start with any keyword in the exclusion list
        if file.endswith(".fif") and not file.startswith(tuple(A_exclude_list)):
            fif_files_A.append(os.path.join(root, file))
        # Print a message if a file is excluded
        elif file.endswith(".fif"):
            print("Exclude file: ", file)
        
fif_files_B = []
for root, dirs, files in os.walk(folder_path_B):
    for file in files:
        if file.endswith(".fif") and not file.startswith(tuple(B_exclude_list)):
            fif_files_B.append(os.path.join(root, file))
        elif file.endswith(".fif"):
            print("Exclude file: ", file)
fif_files_C = []
for root, dirs, files in os.walk(folder_path_C):
    for file in files:
        if file.endswith(".fif") and not file.startswith(tuple(C_exclude_list)):
            fif_files_C.append(os.path.join(root, file))
        elif file.endswith(".fif"):
            print("Exclude file: ", file)

# Retain only files with "imagine" in the filename, i.e., imagination segments
fif_files_A_imagine = [f for f in fif_files_A if "imagine" in f]
fif_files_B_imagine = [f for f in fif_files_B if "imagine" in f]
fif_files_C_imagine = [f for f in fif_files_C if "imagine" in f]
# Retain only files with "read" in the filename, i.e., reading segments
fif_files_A_read = [f for f in fif_files_A if "read" in f]
fif_files_B_read = [f for f in fif_files_B if "read" in f]
fif_files_C_read = [f for f in fif_files_C if "read" in f]

# Read all imagination Epochs objects
epochs_list_A_imagine = [mne.read_epochs(fif_file, verbose = 'ERROR') for fif_file in fif_files_A_imagine]
epochs_list_B_imagine = [mne.read_epochs(fif_file, verbose = 'ERROR') for fif_file in fif_files_B_imagine]
epochs_list_C_imagine = [mne.read_epochs(fif_file, verbose = 'ERROR') for fif_file in fif_files_C_imagine]

# Read all reading Epochs objects
epochs_list_A_read = [mne.read_epochs(fif_file, verbose = 'ERROR') for fif_file in fif_files_A_read]
epochs_list_B_read = [mne.read_epochs(fif_file, verbose = 'ERROR') for fif_file in fif_files_B_read]
epochs_list_C_read = [mne.read_epochs(fif_file, verbose = 'ERROR') for fif_file in fif_files_C_read]

# Get all bad channels (bads) and unify them
all_bads_A_imagine = set()
for epochs in epochs_list_A_imagine:
    all_bads_A_imagine.update(epochs.info['bads'])
    print(epochs.info['bads'],"from A",epochs)
# Get all bad channels (bads) and unify them for reading segments
all_bads_A_read = set()
for epochs in epochs_list_A_read:
    all_bads_A_read.update(epochs.info['bads'])
    print(epochs.info['bads'],"from A",epochs)

all_bads_B_imagine = set()
for epochs in epochs_list_B_imagine:
    all_bads_B_imagine.update(epochs.info['bads'])
    print(epochs.info['bads'],"from B",epochs)
all_bads_B_read = set()
for epochs in epochs_list_B_read:
    all_bads_B_read.update(epochs.info['bads'])
    print(epochs.info['bads'],"from B",epochs)

all_bads_C_imagine = set()
for epochs in epochs_list_C_imagine:
    all_bads_C_imagine.update(epochs.info['bads'])
    print(epochs.info['bads'],"from C",epochs)

all_bads_C_read = set()
for epochs in epochs_list_C_read:
    all_bads_C_read.update(epochs.info['bads'])
    print(epochs.info['bads'],"from C",epochs)

# Parietal channel list
C_channels = ['Cz','CCP2h','CP2','CPPz','CP1','CCP1h','FCC1h','FCCz','FCC2h']

# Unify the bads list for all Epochs objects

# Remove parietal channels from the bads list
for epochs in epochs_list_A_imagine:
    epochs.info['bads'] = list(all_bads_A_imagine - set(C_channels))
for epochs in epochs_list_A_read:
    epochs.info['bads'] = list(all_bads_A_read - set(C_channels))

for epochs in epochs_list_B_imagine:
    epochs.info['bads'] = list(all_bads_B_imagine - set(C_channels))
for epochs in epochs_list_B_read:
    epochs.info['bads'] = list(all_bads_B_read - set(C_channels))

for epochs in epochs_list_C_imagine:
    epochs.info['bads'] = list(all_bads_C_imagine - set(C_channels))
for epochs in epochs_list_C_read:
    epochs.info['bads'] = list(all_bads_C_read - set(C_channels))

print(f"Bad channels list for Subject A imagination segments: {all_bads_A_imagine}")
print(f"Bad channels list for Subject A reading segments: {all_bads_A_read}")
print(f"Bad channels list for Subject B imagination segments: {all_bads_B_imagine}")
print(f"Bad channels list for Subject B reading segments: {all_bads_B_read}")
print(f"Bad channels list for Subject C imagination segments: {all_bads_C_imagine}")
print(f"Bad channels list for Subject C reading segments: {all_bads_C_read}")

# Concatenate all Epochs objects
for subject in ['A', 'B', 'C']:
    exec(f'epochs_list_{subject}_imagine_all = mne.concatenate_epochs(epochs_list_{subject}_imagine)')
    exec(f'epochs_list_{subject}_read_all = mne.concatenate_epochs(epochs_list_{subject}_read)')

A_imagine_evoked = epochs_list_A_imagine_all.average()
B_imagine_evoked = epochs_list_B_imagine_all.average()
C_imagine_evoked = epochs_list_C_imagine_all.average()

# Concatenate two time segments
times1 = np.arange(5.1,5.7,0.1)
times2 = np.arange(5.8,8.1,0.6)
times = np.concatenate((times1,times2),axis=0)

A_time_topomap = A_imagine_evoked.plot_topomap(times, contours = 10)
B_time_topomap = B_imagine_evoked.plot_topomap(times, contours = 10)
C_time_topomap = C_imagine_evoked.plot_topomap(times, contours = 10)

A_time_topomap.savefig('figs/A_time_topomap.png')
B_time_topomap.savefig('figs/B_time_topomap.png')
C_time_topomap.savefig('figs/C_time_topomap.png')
erpA_fig = A_imagine_evoked.plot()
erpB_fig = B_imagine_evoked.plot()
erpC_fig = C_imagine_evoked.plot()
erpA_fig.savefig('figs/erpA.png')
erpB_fig.savefig('figs/erpB.png')
erpC_fig.savefig('figs/erpC.png')
