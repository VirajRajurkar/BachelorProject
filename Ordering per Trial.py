# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 20:13:55 2024

@author: SKV Hähnlein
"""

import pandas as pd

# Load the CSV file
file_path = 'C:/Users/MDilf/Desktop/Anka Datenbearbeitung/Feature Extraction/Feature_SNN/Binary/SNN_Ordered_Subject/SNN_ordered_after_subject_Japanese.csv'  # Replace with the actual file path
df = pd.read_csv(file_path)

# Define the chunk size
chunk_size = 160

# Split the DataFrame into chunks
chunks = [df[i:i + chunk_size] for i in range(0, len(df), chunk_size)]

# Rearrange chunks into the desired order
reordered_chunks = []

# Iterate in sets of 4 chunks to apply the reordering pattern
for i in range(0, len(chunks), 4):
    if i < len(chunks):       # Add the 1st chunk
        reordered_chunks.append(chunks[i])
    if i + 2 < len(chunks):   # Add the 3rd chunk (if it exists)
        reordered_chunks.append(chunks[i + 2])
    if i + 1 < len(chunks):   # Add the 2nd chunk (if it exists)
        reordered_chunks.append(chunks[i + 1])
    if i + 3 < len(chunks):   # Add the 4th chunk (if it exists)
        reordered_chunks.append(chunks[i + 3])

# Combine reordered chunks back into a single DataFrame
result_df = pd.concat(reordered_chunks, ignore_index=True)

# Save the reordered DataFrame to a new CSV file
result_file_path = 'C:/Users/MDilf/Desktop/Anka Datenbearbeitung/Feature Extraction/Feature_SNN/Binary/SNN_Ordered_Trial/SNN_ordered_after_Trial_Japanese.csv'
result_df.to_csv(result_file_path, index=False)

print(f"Reordered file saved to: {result_file_path}")
