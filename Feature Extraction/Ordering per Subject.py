# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 12:55:30 2024

@author: SKV Hähnlein
"""

import pandas as pd

# Load your DataFrame (replace 'your_file.csv' with your actual file)
df = pd.read_csv('C:/Users/MDilf/Desktop/Anka Datenbearbeitung/Feature Extraction/Feature_SNN/Binary/SNN_Feature_Matrix_Japanese.csv')

# Step 1: Split the DataFrame into chunks of 320 rows each
chunk_size = 320
chunks = [df[i:i+chunk_size] for i in range(0, len(df), chunk_size)]

# Step 2: Rearrange the chunks in the desired order
reordered_chunks = []
num_chunks = len(chunks)
for i in range(len(chunks) // 2):  # Handle first half and second half
    if i < len(chunks):  # Add first half
        reordered_chunks.append(chunks[i])
    if i + len(chunks) // 2 < num_chunks:  # Add corresponding second half
        reordered_chunks.append(chunks[i + len(chunks) // 2])

# Step 3: Combine the reordered chunks back into a single DataFrame
reordered_df = pd.concat(reordered_chunks)

# Step 4: Save the reordered DataFrame to a new file
reordered_df.to_csv('C:/Users/MDilf/Desktop/Anka Datenbearbeitung/Feature Extraction/Feature_SNN/Binary/SNN_Ordered_Subject/SNN_Ordered_Japanese.csv', index=False)

print("Reordered file saved")
