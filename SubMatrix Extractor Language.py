# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 10:45:02 2024

@author: SKV Hähnlein
"""

import numpy as np
import pandas as pd

# Load the distance matrix and labels
distance_matrix = np.load('C:/Users/MDilf/Desktop/Anka Datenbearbeitung/Feature Extraction/Feature_SNN/distance_matrix.npy')
labels = np.load('C:/Users/MDilf/Desktop/Anka Datenbearbeitung/Feature Extraction/Feature_SNN/labels.npy')

# Define the ranges for each language group
language_ranges = {
    "Ara": slice(0, 1600),        # NAra
    "Dan": slice(1600, 3200),     # NDan
    "Eng": slice(3200, 4800),     # NEng
    "Hin": slice(4800, 6400),     # NHin
    "Jap": slice(6400, 8000),     # NJap
    "YAra": slice(8000, 9600),    # YAra
    "YDan": slice(9600, 11200),   # YDan
    "YEng": slice(11200, 12800),  # YEng
    "YHin": slice(12800, 14400),  # YHin
    "YJap": slice(14400, 16000)   # YJap
}

# Combine groups and save as CSV
language_groups = {
    "Ara": (language_ranges["Ara"], language_ranges["YAra"]),
    "Dan": (language_ranges["Dan"], language_ranges["YDan"]),
    "Eng": (language_ranges["Eng"], language_ranges["YEng"]),
    "Hin": (language_ranges["Hin"], language_ranges["YHin"]),
    "Jap": (language_ranges["Jap"], language_ranges["YJap"])
}

# Iterate through each language group, extract submatrices, and save to CSV
for language, (range1, range2) in language_groups.items():
    # Combine indices of N and Y groups
    combined_indices = np.r_[range1, range2]
    
    # Extract the submatrix
    submatrix = distance_matrix[np.ix_(combined_indices, combined_indices)]
    
    # Create labels for the combined group
    group_labels = ["No"] * 1600 + ["Yes"] * 1600  # First 1600 "No", next 1600 "Yes"
    
    # Add labels as the last column to the matrix
    labeled_matrix = np.zeros((submatrix.shape[0], submatrix.shape[1] + 1), dtype=object)
    labeled_matrix[:, :-1] = submatrix  # Copy the submatrix into all but the last column
    labeled_matrix[:, -1] = group_labels  # Add labels as the last column
    
    # Convert to DataFrame
    labeled_matrix_df = pd.DataFrame(labeled_matrix)
    
    # Save as a CSV file
    filename = f"C:/Users/MDilf/Desktop/Anka Datenbearbeitung/Feature Extraction/Feature_SNN/SNN_Binary/SNN_Unordered/SNN_Feature_Matrix_{language}.csv"
    labeled_matrix_df.to_csv(filename,index=False)
    print(f"Saved {filename}")
