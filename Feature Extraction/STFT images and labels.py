# -*- coding: utf-8 -*-
"""
Spyder-Editor

Dies ist eine temporäre Skriptdatei.
"""
import pandas as pd
import numpy as np
from scipy.signal import stft
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
from PIL import Image
import io


# Read the CSV file using pandas
csv_filename = [ 'C:/Users/MDilf/Desktop/Anka Datenbearbeitung/Datenbank/5-Preprocessed Japanese/Preprocessed_Data_Japanese1.csv',
                'C:/Users/MDilf/Desktop/Anka Datenbearbeitung/Datenbank/5-Preprocessed Japanese/Preprocessed_Data_Japanese2.csv',
               'C:/Users/MDilf/Desktop/Anka Datenbearbeitung/Datenbank/5-Preprocessed Japanese/Preprocessed_Data_Japanese3.csv',
               'C:/Users/MDilf/Desktop/Anka Datenbearbeitung/Datenbank/5-Preprocessed Japanese/Preprocessed_Data_Japanese4.csv',
               'C:/Users/MDilf/Desktop/Anka Datenbearbeitung/Datenbank/5-Preprocessed Japanese/Preprocessed_Data_Japanese5.csv',
               'C:/Users/MDilf/Desktop/Anka Datenbearbeitung/Datenbank/5-Preprocessed Japanese/Preprocessed_Data_Japanese6.csv',
               'C:/Users/MDilf/Desktop/Anka Datenbearbeitung/Datenbank/5-Preprocessed Japanese/Preprocessed_Data_Japanese7.csv',
               'C:/Users/MDilf/Desktop/Anka Datenbearbeitung/Datenbank/5-Preprocessed Japanese/Preprocessed_Data_Japanese8.csv',
               'C:/Users/MDilf/Desktop/Anka Datenbearbeitung/Datenbank/5-Preprocessed Japanese/Preprocessed_Data_Japanese9.csv',
               'C:/Users/MDilf/Desktop/Anka Datenbearbeitung/Datenbank/5-Preprocessed Japanese/Preprocessed_Data_Japanese10.csv'

               ]
file_counter=0
for file in csv_filename:
    df = pd.read_csv(file)
    file_counter+=1
    # Extract the number of channels, number of questions, and segment length
    num_channels = df['Channel'].nunique()
    num_questions = df['Question'].nunique()
    segment_length = len(df.columns) - 2  # Subtracting 'Channel' and 'Question' columns
    
    
    # Initialize an empty numpy array to store the data
    data= np.zeros((num_channels, num_questions, segment_length))
    print(data.shape)
    
    # Iterate through the DataFrame to fill the reshaped_data array
    for index, row in df.iterrows():
        try:
            channel_idx = int(row['Channel']) - 1  # Convert channel number to zero-based index
            question_idx = int(row['Question']) - 1  # Convert question number to zero-based index
            
          
            # Get the interval data from the current row
            interval_data = row.iloc[2:].values  # Skip 'Channel' and 'Question' columns
    
    
            # Assign the data to the appropriate position in the reshaped_data array
            data[channel_idx, question_idx, :] = interval_data
        
    
        except Exception as e:
            print(f"Error processing row {index}: {e}")
    
    print("Data reshaped successfully from CSV!")
    
    
    
    # Define the directory where you want to save the images
    output_dir = 'C:/Users/MDilf/Desktop/Anka Datenbearbeitung/Feature Extraction/STFT_images/Japanese'
    os.makedirs(output_dir, exist_ok=True)  # Create the directory if it doesn't exist
    
    # Parameters
    fs = 125  # Hz
    nperseg = 256  # Length of each segment
    overlap=nperseg/2
    
    # Initialize a counter for the total number of images
    image_counter = 0
    
    # Iterate over each of the 20 channels for each sample
    for channel_idx in range(data.shape[0]):
        for sample_idx in range(data.shape[1]):
            # Extract the time-series data for this channel
            time_series_data = data[channel_idx, sample_idx, :]
    
            # Compute STFT
            f, t, Zxx = stft(time_series_data, fs=fs, nperseg=nperseg,noverlap=overlap)
    
            # Plot the magnitude of the STFT for visualization
            plt.figure()
            plt.pcolormesh(t, f, np.abs(Zxx), shading='gouraud')
            plt.title(f'STFT Magnitude File {file_counter}- Channel {channel_idx+1}, Question {sample_idx+1}')
            plt.ylabel('Frequency [Hz]')
            plt.xlabel('Time [sec]')
            plt.colorbar(label='Magnitude')
    
            # Save the figure as a .png file
            # Save the plot to an in-memory buffer
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', transparent=False)
            buffer.seek(0)  # Move the cursor to the start of the buffer
    
            # Open the image from the buffer and convert to RGB
            img = Image.open(buffer)
            rgb_img = img.convert('RGB')
    
            # Save the RGB image directly to a file
            filename_pictures = os.path.join(output_dir, f'stft_file_{file_counter}_channel_{channel_idx+1}_Question_{sample_idx+1}.png')
            rgb_img.save(filename_pictures)
    
            # Close the buffer
            buffer.close()
            
            # Close the plot to free up memory
            plt.close()
    
            # Increment the counter after each plot
            image_counter += 1
    
    # Print the total number of images generated
    print(f'Files processed: {file_counter}')
    print(f'Total number of images generated: {image_counter}')
    print("Iamges saved sucessfully")

english_labels=np.array([0,1,0,1,0,0,0,1,1,0,1,0,1,0,1,1,1,0,1,0])

hindi_labels=np.array([1,0,1,0,1,0,1,0,0,1,0,1,0,1,1,0,0,1,0,1])

japanese_labels=np.array([0,1,0,1,0,0,1,0,0,1,0,1,1,1,1,0,0,0,1,1])

arabic_labels=np.array([0,1,0,0,1,1,1,0,0,0,0,0,1,1,1,0,1,1,0,1])

danish_labels=np.array([1,0,0,0,0,0,1,0,1,1,0,1,0,1,1,1,0,1,1,0])

label_dict = {
    "english": english_labels,
    "hindi": hindi_labels,
    "japanese": japanese_labels,
    "arabic": arabic_labels,
    "danish": danish_labels
}

#Choose the right language
chosen_language = "japanese"

# Extract the corresponding labels
chosen_labels = label_dict[chosen_language]

concatenated_labels = np.tile(chosen_labels, 16*20)

# Save the chosen vector as a .npy file
filename_labels = os.path.join(output_dir, f"C:/Users/MDilf/Desktop/Anka Datenbearbeitung/Feature Extraction/STFT_images/Japanese/{chosen_language}_labels.npy")
np.save(filename_labels, concatenated_labels)

print(f"Labels for {chosen_language} saved")
print(f'Total number of labels generated: {len(concatenated_labels)}')
