# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 08:23:18 2024

@author: SKV Hähnlein
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 10:05:26 2024

@author: ankadilfer
"""
import pyxdf
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import  filtfilt, welch, freqz, butter 
import numpy.polynomial.polynomial as poly
import mne 
mne.set_log_level('CRITICAL')
from pyprep import NoisyChannels
import pandas as pd
from mne_icalabel import label_components
import csv
import os

def butter_bandpass_filter(data, lowcut, highcut, fs, order):
    b, a = butter(order, [lowcut, highcut], btype='bandpass',fs=fs)
    filtered_data =  filtfilt(b, a, data, axis=0)
    
    # Compute the frequency response
    #w, h = freqz(b,a, worN=2048)
    
    # Convert frequencies to Hz for plotting
    #freqs = w * fs / (2 * np.pi)
    
    # Plot the magnitude response
    # plt.figure(figsize=(12, 6))
    # plt.plot(freqs, 20 * np.log10(abs(h)), 'b')
    # plt.title("Bandpass Filter Frequency Response")
    # plt.xlabel("Frequency (Hz)")
    # plt.ylabel("Gain (dB)")
    # plt.grid()
    # plt.show()
    
    return filtered_data

def bandstop_filter(data, fs, lowcut, highcut, order):
    b,a = butter(order, [lowcut, highcut], btype='bandstop', fs=fs)
    filtered_data =  filtfilt(b, a, data, axis=0)
    
    # Compute the frequency response
    #w, h = freqz(b,a)
    
    # Convert frequencies to Hz for plotting
    #freqs = w * fs / (2 * np.pi)
    
    # Plot the magnitude response
    # plt.figure(figsize=(12, 6))
    # plt.plot(freqs, 20 * np.log10(abs(h)), 'b')
    # plt.title("Bandstop Filter Frequency Response")
    # plt.xlabel("Frequency (Hz)")
    # plt.ylabel("Gain (dB)")
    # plt.grid()
    # plt.show()
    
    return filtered_data



def plot_PSD(data, fs, stage):

    n_samples, n_channels = data.shape
    plt.figure(figsize=(12, 8))

    # Define font size for the plot
    font_size = 16

    # Channel names
    channel_names = ["Fp1", "Fp2", "F3", "F4", "Fz", "C3", "C4", "Cz", "P3", "P4", "Pz", "O1", "O2", "T5", "T6", "T3"]

    # Define a color palette for better aesthetics
    color_palette = plt.cm.viridis(np.linspace(0, 1, n_channels))

    # Loop through each channel
    for channel in range(n_channels):
        # Calculate the PSD using Welch's method
        freqs, psd = welch(data[:, channel], fs=fs, nperseg=512)

        # Plot the PSD for the current channel
        if channel < len(channel_names):
            plt.semilogy(freqs, psd, label=channel_names[channel], color=color_palette[channel])
        else:
            plt.semilogy(freqs, psd, label=f'Channel {channel + 1}', color=color_palette[channel])

    # Set plot title and labels with larger font size
    plt.title(f"PSD of {stage} Data", fontsize=font_size, weight='bold')
    plt.xlabel("Frequency (Hz)", fontsize=font_size-1, weight='bold')
    plt.ylabel("Power/Frequency (dB/Hz)", fontsize=font_size-1, weight='bold')

    # Customize tick parameters for better readability
    plt.xticks(fontsize=font_size - 2)
    plt.yticks(fontsize=font_size - 2)

    # Add grid with minor and major lines for precision
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

    # Add legend with transparency for a professional look
    plt.legend(fontsize=font_size - 2, loc='upper right', frameon=True, framealpha=0.8)

    # Adjust layout for better spacing
    plt.tight_layout()

    # Add horizontal and vertical lines to denote key regions (optional)
    plt.axhline(y=0, color='black', linewidth=0.8, linestyle='-')
    plt.axvline(x=0, color='black', linewidth=0.8, linestyle='-')

    plt.show()



    

def plot_signal(data, time, fs, stage):
    
    n_samples, n_channels = data.shape
    plt.figure(figsize=(15, 3))  # Adjust figure size for a single channel

    # Define font size for the plot
    font_size = 16

    # Channel names (for up to 16 channels)
    channel_names = ["Fp1", "Fp2", "F3", "F4", "Fz", "C3", "C4", "Cz", 
                     "P3", "P4", "Pz", "O1", "O2", "T5", "T6", "T3"]

    # Plot only the first channel
    plt.plot(time[:-2875], data[2500:-375, 0], label="Channel 1", color="navy", linewidth=1.8)  # Use a clean color and line width

    # Set plot title and labels with larger font size
    channel_name = channel_names[0] if n_channels > 0 else "Channel 1"
    plt.title(f"Time-Domain Signal - {channel_name} ({stage} Data)", fontsize=font_size, weight='bold')
    plt.xlabel("Time (s)", fontsize=font_size - 1, weight='bold')
    plt.ylabel("Amplitude (bit values)", fontsize=font_size - 1, weight='bold')  # Added µV for amplitude representation

    # Customize tick parameters for better readability
    plt.xticks(fontsize=font_size - 2)
    plt.yticks(fontsize=font_size - 2)

    # Add minor ticks for a scientific appearance
    plt.minorticks_on()
    plt.grid(True, which='major', linestyle='--', linewidth=0.8, alpha=0.7)
    plt.grid(True, which='minor', linestyle=':', linewidth=0.5, alpha=0.5)

    # Highlight zero line for better reference
    plt.axhline(y=0, color='gray', linestyle='-', linewidth=0.8, alpha=0.7)

    # Add a legend
    plt.legend(loc='upper right', fontsize=font_size - 2, frameon=True, framealpha=0.9)

    # Adjust layout for better spacing
    plt.tight_layout()
    plt.show()





# Extract marker indices based on value
def get_marker_indices(marker_data, marker_value, marker_timestamps):
    time_indices=[]
    marker_indices = [index for index, value in enumerate(marker_data) if value == marker_value]
    for i in marker_indices:
        if i < len(marker_timestamps):
            time_indices.append(marker_timestamps[i])
    return time_indices

# Find closest indices in the data timestamps
def find_closest_indices(target_list, reference_list):
    indices = []
    for target in target_list:
        closest_index = np.argmin(np.abs(reference_list - target))
        indices.append(closest_index)
    return indices

# Extract intervals for a given start and end marker
def extract_intervals(marker, raw_data, avg_length):
    # Initialize list to store intervals for each channel
    channel_intervals = []
    
    # Loop over each channel in reconstructed_raw_data
    for channel_idx in range(raw_data.shape[0]):
        channel_data = []  # This will store both data intervals and their expanded indices for each interval
        
        # Extract intervals for the current channel using the average length
        for start in marker:
            # Calculate the end as start + average_length
            end = start + avg_length
            
            # Ensure start and end are within bounds
            if start < raw_data.shape[1] and end <= raw_data.shape[1]:
                # Get data interval
                interval = raw_data[channel_idx, start:end]
                
                # Generate the array of indices from start to end (exclusive)
                indices_array = np.arange(start, end)
                
                # Append both the interval data and the indices array as a pair
                channel_data.append([interval, indices_array])
                
            else:
                print(f"Out of bounds for channel {channel_idx}: start={start}, end={end}")
        
        # Append the channel data for this channel to the main list
        channel_intervals.append(channel_data)
    return channel_intervals


def reshape_data(data,average_length):
    num_channels = len(data)  # Should be 16
    num_questions = len(data[0])  # Should be 20
    segment_length =  average_length # Use the calculated average length
      # Initialize an empty numpy array to store the reshaped data
    reshaped_data = np.zeros((num_channels, num_questions, segment_length))

      # Fill the reshaped_data array with interval data for each channel and question
    for channel_idx in range(num_channels):
              for question_idx in range(num_questions):
                  # Extract the interval data for the current channel and question
                  interval_data = data[channel_idx][question_idx][0]
                  
                  # Ensure the interval data length matches the segment_length
                  if len(interval_data) == segment_length:
                      reshaped_data[channel_idx, question_idx, :] = interval_data
                  else:
                      print(f"Interval length mismatch for channel {channel_idx}, question {question_idx}")
    reshaped_data = np.transpose(reshaped_data, (1, 0, 2)) 
    return reshaped_data

    

# File paths and channel names

file_paths = [
               'C:/Users/MDilf/Desktop/Anka Datenbearbeitung/Datenbank/3-Hindi Trials/Hindi4_Viraj1.xdf'
    ]
channel_names = ["Fp1", "Fp2", "F3", "F4", "Fz", "C3", "C4", "Cz", "P3", "P4", "Pz", "O1", "O2", "T5", "T6", "T3"]
counter=0
# Process each file
for file_path in file_paths:
    counter+=1
    print(f'The counter is now {counter}')
    streams, header = pyxdf.load_xdf(file_path)
    
    obci_eeg_stream = None
    Marker = None
    for stream in streams:
        if 'obci_eeg2' in stream['info']['name'][0]:
            obci_eeg_stream = stream
        elif 'Markers' in stream['info']['name'][0]:
            Marker = stream
            
    
    #Set sampling frequency, load data and create time array
    sfreq = float(obci_eeg_stream["info"]["nominal_srate"][0])
    data = obci_eeg_stream["time_series"]
    time=np.arange(0,len(data),1)/sfreq
    
    
    
    
    # Check the shape of the data and the sampling frequency 
    frequency= data.shape[0]/time[-1]
    print(f'The actual sampling frequency is {frequency} ')
    print(f'The actual shape of the data is {data.shape} ')
    
    
    
    #Plot raw data 
    plot_PSD(data, sfreq, "raw")
    plot_signal(data, time, sfreq, "raw")
    
    
    
    # Bandpass and Bandstop filtering 
    filtered_data= butter_bandpass_filter(data, lowcut=1,highcut=40,fs= sfreq, order=4)
    filtered_data= bandstop_filter(filtered_data, sfreq, 24, 26,3)
    filtered_data= bandstop_filter(filtered_data, sfreq, 49, 51,4)
    
    
    #Subtracting DC offset
    overall_means = np.mean(filtered_data, axis=0)
    corrected_eeg_data = filtered_data - overall_means[np.newaxis,:]  
    n_samples, n_channels = corrected_eeg_data.shape

    
  
    #Plot corrected and filtered signal 
    plot_PSD(corrected_eeg_data, sfreq, "filtered and offset corrected")
    plot_signal(corrected_eeg_data, time, sfreq,"filtered and offset corrected")

    
    
    #Create mne object 
    info = mne.create_info(corrected_eeg_data.shape[1], sfreq, ["eeg"] * corrected_eeg_data.shape[1])
    preprocessed_raw = mne.io.RawArray(corrected_eeg_data.T, info)
    
    # Rename channels, set channel locations
    preprocessed_raw.rename_channels({
        '0': 'Fp1', '1': 'Fp2', '2': 'F3', '3': 'F4', 
        '4': 'Fz', '5': 'C3', '6': 'C4', '7': 'Cz',
        '8': 'P3', '9': 'P4', '10': 'Pz', '11': 'O1',
        '12': 'O2', '13': 'T5', '14': 'T6', '15': 'T3'
    })
    preprocessed_raw.set_montage("standard_1020")
  
    
    
    
    #Bad channel detection and interpolation
    noisy_data = NoisyChannels(preprocessed_raw, random_state=1337)
    noisy_data.find_all_bads(ransac=False)
    all_bad_channels = noisy_data.get_bads()
    preprocessed_raw.info['bads'] = all_bad_channels
    preprocessed_raw.interpolate_bads()
    print("All bad channels:", all_bad_channels)
    
    
    #Rereference Signal
    preprocessed_raw.set_eeg_reference('average', projection=False)
    
    

    
    #Plot interpolated signal 
    plot_PSD(preprocessed_raw.get_data().T, sfreq, "re-referrenced")
    plot_signal(preprocessed_raw.get_data().T, time, sfreq,  "re-referrenced")
    
    
    #Fit ICA
    ica = mne.preprocessing.ICA(n_components=10, random_state=42, method='fastica', verbose=True)
    ica.fit(preprocessed_raw)  
    
    
    #Label ICs and filter out the ones not related to brain signals
    ica_labels = label_components(preprocessed_raw, ica, method="iclabel")
    df = pd.DataFrame(ica_labels)
    low_certainty_components = df[(df['y_pred_proba'] < 0.60) | (df['labels'] != 'brain')].index.tolist()
    print("Low-certainty or non-brain components to exclude:", low_certainty_components)
    print("The Labels have been classified as:",df)
    


    # Apply ICA to reconstruct the signal without low-certainty components
    ica.exclude = low_certainty_components
    raw_cleaned = ica.apply(preprocessed_raw.copy())
    
    
    #Plot reconstructed signal 
    plot_PSD(raw_cleaned.get_data().T, sfreq, "after ICA")
    plot_signal(raw_cleaned.get_data().T, time, sfreq, "after ICA")
    
    
    
    
    
    #Epoching 
    marker_data = Marker['time_series']  # marker values
    marker_timestamps = Marker['time_stamps']  # corresponding timestamps
    data_timestamps = np.array(obci_eeg_stream['time_stamps'])  # Convert to numpy for efficient calculation
    print('Debug data shape control', raw_cleaned.get_data().shape)
    
    
    # Extract marker timestamps for values 2, 3, and 4
    time_indices_2 = get_marker_indices(marker_data, 2, marker_timestamps)
    time_indices_3 = get_marker_indices(marker_data, 3, marker_timestamps)
    time_indices_4 = get_marker_indices(marker_data, 4, marker_timestamps)

    

   # # Find closest data timestamps for each marker set
    indices_2 = find_closest_indices(time_indices_2, data_timestamps)
    indices_3 = find_closest_indices(time_indices_3, data_timestamps)
    indices_4 = find_closest_indices(time_indices_4, data_timestamps)
  
    
   

    # Calculate average interval lengths for each marker pair
    interval_lengths_2_to_3 = [end - start for start, end in zip(indices_2, indices_3)]
    average_length_2_to_3 = int(np.mean(interval_lengths_2_to_3))

   
    #Get the start  1s after the beginn of the relaxing period
    updated_indices_4 = [value + 125 for value in indices_4]

    # Extract intervals for each channel for markers 2-to-3 and 3-to-4
    raw_cleaned_data = raw_cleaned.get_data()  # Shape: (n_channels, n_times)
    channel_intervals_2_to_3 = extract_intervals(indices_2, raw_cleaned_data, average_length_2_to_3)
    channel_intervals_4_to_1 = extract_intervals(updated_indices_4, raw_cleaned_data, int(sfreq))
 


    
    
    # Define font size for the plot
    font_size = 16
    
    # Plot chosen epochs
    time_axis = data_timestamps - data_timestamps[0]  # Shift the time axis to start at 0
    adjusted_marker_times = marker_timestamps - data_timestamps[0]
    
    plt.figure(figsize=(15, 6))  # Adjust figure size for clarity

    # Plot the markers as vertical lines
    for i, marker_time in enumerate(adjusted_marker_times):
        plt.axvline(x=marker_time, color='red', linestyle='--', label="Beep Tone" if i == 0 else "")

    # Imagined speech intervals
    channel_1_intervals_imagined_speech = channel_intervals_2_to_3[0]  # Assuming channel 1 is the first channel (index 0)
    for interval_data, indices in channel_1_intervals_imagined_speech:
        interval_times = time_axis[indices]  # Convert indices to corresponding time values
        plt.plot(interval_times, interval_data, color="orange", linewidth=2,
                 label="Imagined Speech" if interval_data is channel_1_intervals_imagined_speech[0][0] else "")

    # White noise intervals
    channel_1_intervals_white_noise = channel_intervals_4_to_1[0]  # Assuming channel 1 is the first channel (index 0)
    for interval_data, indices in channel_1_intervals_white_noise:
        interval_times_1 = time_axis[indices]  # Convert indices to corresponding time values
        plt.plot(interval_times_1, interval_data, color="blue", linewidth=2, alpha=0.7,
                 label="White Noise" if interval_data is channel_1_intervals_white_noise[0][0] else "")

    # Add labels, legend, and title with uniform font sizes
    plt.xlabel("Time (s)", fontsize=font_size - 1, weight='bold')
    plt.ylabel("Amplitude (bit values)", fontsize=font_size - 1, weight='bold')
    plt.title("Fp1 Channel with Highlighted Intervals on Imagined Speech", fontsize=font_size, weight='bold')

    # Customize tick parameters for better readability
    plt.xticks(fontsize=font_size - 2)
    plt.yticks(fontsize=font_size - 2)

    # Add a legend with a transparent background
    plt.legend(fontsize=font_size - 2, frameon=True, framealpha=0.9)

    # Enhance grid appearance
    plt.grid(True, which='major', linestyle='--', linewidth=0.8, alpha=0.7)
    plt.grid(True, which='minor', linestyle=':', linewidth=0.5, alpha=0.5)
    plt.minorticks_on()

    # Highlight zero line
    plt.axhline(y=0, color='gray', linestyle='-', linewidth=0.8, alpha=0.7)

    # Adjust layout for better spacing
    plt.tight_layout()
    plt.show()
        
        
        
        
    #Baseline correction of imaginedd Speech epochs based on white noise using absolute baseline corerrection approach
    imagined_speech_data=reshape_data(channel_intervals_2_to_3,average_length_2_to_3)
    white_noise_data=reshape_data(channel_intervals_4_to_1,int(sfreq))
    baseline_white_noise = np.mean(white_noise_data, axis=2, keepdims=True) 
    baseline_expanded = np.tile(baseline_white_noise, (1, 1, imagined_speech_data.shape[2]))
    baseline_corrected_epochs = imagined_speech_data - baseline_expanded  

    

    
    
    #Z-score normalization
    mean_epochs = np.mean(baseline_corrected_epochs, axis=2, keepdims=True)
    std_epochs = np.std(baseline_corrected_epochs, axis=2, keepdims=True)
    zscore_normalized_epochs = (baseline_corrected_epochs - mean_epochs) / std_epochs
    
    
    
    #Plot relative z-scored normalized signal and its psd of channel 1

    channel_1_data = zscore_normalized_epochs[:, 0, :]
    time_axis = np.arange(0, zscore_normalized_epochs.shape[2]-126)/sfreq
    
    # Plot each trial of channel 1
    plt.figure(figsize=(12, 8))
    for trial in range(20):
        plt.plot(time_axis,channel_1_data[trial][63:-63], label=f'Trial {trial + 1}')
    
    plt.title("Z-Score Normalized and Baseline Corrected Epochs for Channel 1 (20 Trials)")
    plt.xlabel("Time (s)")
    plt.ylabel("Z-Score")
    plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
    plt.show()

    
    # Plot PSD for each trial
    plt.figure(figsize=(12, 8))
    for trial in range(20):
        freqs, psd = welch(channel_1_data[trial][63:-63], fs=sfreq)
        plt.plot(freqs, psd, label=f'Trial {trial + 1}')
    
    plt.title("Power Spectral Density (PSD) for Channel 1 (20 Trials)")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power/Frequency (dB/Hz)")
    plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
    plt.show()
    
    
    
    
    
    #Save preprocessed data as csv file 
    num_channels = zscore_normalized_epochs.shape[1]  # Should be 16
    num_questions = zscore_normalized_epochs.shape[0] # Should be 20
    segment_length = zscore_normalized_epochs.shape[2]   # Should be 1143
    print(zscore_normalized_epochs.shape)
    save_data = np.zeros((num_channels, num_questions, segment_length))
    
    # Fill the save_data array with interval data for each channel and question
    for question_idx in range(num_questions):
        for channel_idx in range(num_channels):
            # Extract the interval data for the current question and channel
            interval_data = zscore_normalized_epochs[question_idx, channel_idx, :]
    
            # Ensure the interval data length matches the segment_length
            if len(interval_data) == segment_length:
                save_data[channel_idx, question_idx, :] = interval_data
            else:
                print(f"Interval length mismatch for question {question_idx}, channel {channel_idx}")



    
    # Define the path to save the CSV file
    save_directory = 'C:/Users/MDilf/Desktop/Anka Datenbearbeitung/Datenbank/5-Preprocessed Japanese'#  directory where I want to save
    csv_filename = os.path.join(save_directory,  f'Preprocessed_Data_Japanese{counter}.csv')
    
    # Ensure the save directory exists
    os.makedirs(save_directory, exist_ok=True)
    
    # Flatten the data to save it into a CSV format: each row will contain channel, question, and data points
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        
        # Write header (optional, to identify rows and columns)
        header = ['Channel', 'Question'] + ['Time_Point_' + str(i + 1) for i in range(segment_length)]
        writer.writerow(header)
        
        # Write each row of data (channel, question, time points)
        for channel_idx in range(num_channels):
            for question_idx in range(num_questions):
                # Extract the interval data for the current channel and question
                interval_data = save_data[channel_idx, question_idx, :]
                # Create a row with channel number, question number, and the corresponding data points
                row = [channel_idx + 1, question_idx + 1] + interval_data.tolist()
                writer.writerow(row)
    
    print(f"Preprocessed EEG data saved successfully to {csv_filename}!")
        
    
    
    
    
        
