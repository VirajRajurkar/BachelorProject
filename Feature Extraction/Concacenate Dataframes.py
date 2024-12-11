# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 21:52:39 2024

@author: SKV Hähnlein
"""

import os
import pandas as pd

# Directory containing the CSV files
input_directory = "C:/Users/MDilf/Desktop/Anka Datenbearbeitung/Feature Extraction/Features_Manual_Multiclass"  # Replace with the directory containing your CSV files
output_file = "C:/Users/MDilf/Desktop/Anka Datenbearbeitung/Feature Extraction/Feature_Extraction_Features_Manual_Multiclass.csv"          # Name of the output file

# List all files in the directory
all_files = [os.path.join(input_directory, f) for f in os.listdir(input_directory) if f.endswith('.csv')]

# Initialize an empty list to store dataframes
dataframes = []

# Loop through each file and read it
for file in all_files:
    df = pd.read_csv(file)
    dataframes.append(df)

# Concatenate all dataframes
combined_df = pd.concat(dataframes, ignore_index=True)

# Save the combined dataframe to a CSV file
combined_df.to_csv(output_file, index=False)

print(f"Combined CSV file saved as {output_file}")
