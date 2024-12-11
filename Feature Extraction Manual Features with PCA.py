
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 12:06:08 2024

@author: ankadilfer
"""

import pandas as pd
import glob
import numpy as np
from scipy.signal import welch
import antropy as ant
import pywt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


# Use glob to get all CSV files from a directory
all_files = glob.glob("C:/Users/MDilf/Desktop/Anka Datenbearbeitung/Datenbank/Combined languages/*.csv")

# Create a list to store each file's data
dataframes = []

for filename in all_files:
    df = pd.read_csv(filename, delimiter=',')
    preprocessed_data = df.to_numpy()
    num_rows = preprocessed_data.shape[0]

    # Create an empty feature matrix with rows equal to num_rows and 15 columns
    feature_matrix = np.empty((num_rows, 12))






# Loop over each column in preprocessed_data
    for row_idx in range(num_rows):
        

    
        # Compute the power spectral density (PSD) using Welch's method
        freqs, psd = welch(preprocessed_data[ row_idx,:], 125, noverlap=50)

        # Find the indices corresponding to the band of interest
        idx_band_alpha = np.logical_and(freqs >= 7.5, freqs <= 12)
        idx_band_beta = np.logical_and(freqs >= 12, freqs <= 30)
        idx_band_gamma = np.logical_and(freqs >= 30, freqs <= 55)
        idx_band_delta = np.logical_and(freqs >= 0.5, freqs <= 3.5)
        idx_band_theta= np.logical_and(freqs >= 3.5, freqs <= 7.5)

        # Integrate the power (area under the curve) in the band
        bandpower_alpha = np.trapz(psd[idx_band_alpha], freqs[idx_band_alpha])
        bandpower_beta = np.trapz(psd[idx_band_beta], freqs[idx_band_beta])
        bandpower_gamma= np.trapz(psd[idx_band_gamma], freqs[idx_band_gamma])
        bandpower_delta = np.trapz(psd[idx_band_delta], freqs[idx_band_delta])
        bandpower_theta = np.trapz(psd[idx_band_theta], freqs[idx_band_theta])

        # Store the band powers in the next few rows of feature_matrix
        feature_matrix[row_idx,0] = bandpower_alpha
        feature_matrix[row_idx,1] = bandpower_beta
        feature_matrix[row_idx,2] = bandpower_gamma
        feature_matrix[row_idx,3] = bandpower_delta
        feature_matrix[row_idx,4] = bandpower_theta

  

         #Calculate entropy
        approx_entropy = ant.app_entropy(preprocessed_data[ row_idx,:], order=2, metric='chebyshev')
        feature_matrix[row_idx,5] = approx_entropy
    
    
    
        #Calculate wavelet power
        coeffs = pywt.wavedec(preprocessed_data[ row_idx,:], wavelet='Sym6', level=5)
        energy_detail = [np.sum(np.square(detail)) for detail in coeffs[1:]]  # Detail coefficients at each level
        feature_matrix[row_idx,6] = energy_detail[0] #from level 5
        feature_matrix[row_idx,7] = energy_detail[1] #from level 4
        feature_matrix[row_idx,8] = energy_detail[2] #from level 3
        feature_matrix[row_idx,9] = energy_detail[3] #from level 2
        feature_matrix[row_idx,10] = energy_detail[4] #from level 1
        
        
        
        #Calculate Fractual dimension
        data = np.array(preprocessed_data[row_idx, :], dtype=np.float64)
        data = np.ravel(data)
        higuchi_fd = ant.higuchi_fd(data, kmax=10)   
        feature_matrix[row_idx,11] = higuchi_fd
        
    dataframes.append(feature_matrix)
   

# Convert all feature matrices in dataframes to pandas DataFrames
all_feature_dfs = [pd.DataFrame(matrix) for matrix in dataframes]

# Concatenate them into one big DataFrame
big_dataframe = pd.concat(all_feature_dfs, ignore_index=True)


scaler = StandardScaler()
standardized_features = scaler.fit_transform(big_dataframe)

original_feature_names = [
    "Mean", "Variance", "RMS", "Bandpower Alpha", "Bandpower Beta", 
    "Bandpower Gamma", "Bandpower Delta", "Bandpower Theta", 
    "Approx Entropy", "Energy Level 5", "Energy Level 4", "Energy Level 3", 
    "Energy Level 2", "Energy Level 1", "Fractal Dimension"
]


# Apply PCA
# Retain components that explain 95% of the variance
pca = PCA(n_components=0.95)  # Retain enough components to explain 95% variance
principal_components = pca.fit_transform(standardized_features)

# Loadings: Contributions of each feature to the components
loadings = pd.DataFrame(pca.components_,
                        columns=original_feature_names,  # Replace with your feature names
                        index=[f'PC{i+1}' for i in range(len(pca.components_))])

# Display the loadings
print("PCA Loadings:")
print(loadings)




# Check how much variance is explained by each principal component
explained_variance = pca.explained_variance_ratio_
print(f"Explained variance ratio: {explained_variance}")

# Cumulative variance explained
cumulative_variance = np.cumsum(explained_variance)
print(f"Cumulative variance: {cumulative_variance}")

# Visualize the explained variance ratio (scree plot)
plt.figure(figsize=(8, 5))
plt.plot(cumulative_variance, marker='o', linestyle='--')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Explained Variance vs Number of Principal Components')
plt.grid(True)
plt.show()

 # Define labels for different languages
english_labels = np.array(['YEng', 'NEng', 'YEng', 'NEng', 'YEng', 'YEng', 'YEng', 'NEng', 'NEng', 'YEng', 'NEng', 'YEng', 'NEng', 'YEng', 'NEng', 'NEng', 'NEng', 'YEng', 'NEng', 'YEng'])
hindi_labels = np.array(['NHin', 'YHin', 'NHin', 'YHin', 'NHin', 'YHin', 'NHin', 'YHin', 'YHin', 'NHin', 'YHin', 'NHin', 'YHin', 'NHin', 'NHin', 'YHin', 'YHin', 'NHin', 'YHin', 'NHin'])
japanese_labels = np.array(['YJap', 'NJap', 'YJap', 'NJap', 'YJap', 'YJap', 'NJap', 'YJap', 'YJap', 'NJap', 'YJap', 'NJap', 'NJap', 'NJap', 'NJap', 'YJap', 'YJap', 'YJap', 'NJap', 'NJap'])
arabic_labels = np.array(['YAra', 'NAra', 'YAra', 'YAra', 'NAra', 'NAra', 'NAra', 'YAra', 'YAra', 'YAra', 'YAra', 'YAra', 'NAra', 'NAra', 'NAra', 'YAra', 'NAra', 'NAra', 'YAra', 'NAra'])
danish_labels = np.array(['NDan', 'YDan', 'YDan', 'YDan', 'YDan', 'YDan', 'NDan', 'YDan', 'NDan', 'NDan', 'YDan', 'NDan', 'YDan', 'NDan', 'NDan', 'NDan', 'YDan', 'NDan', 'NDan', 'YDan'])

label_dict = {
    "english": english_labels,
    "hindi": hindi_labels,
    "japanese": japanese_labels,
    "arabic": arabic_labels,
    "danish": danish_labels
}

# List of languages corresponding to the splits
languages = ["arabic", "danish", "english", "hindi", "japanese"]


# Convert PCA results to a DataFrame
df_pca = pd.DataFrame(principal_components)

# Parameters for splitting
rows_per_split = 3200
n_splits = len(languages)

# Validate that the PCA results can be evenly split
if df_pca.shape[0] != rows_per_split * n_splits:
    raise ValueError("Error: PCA result does not match the expected total rows for splitting.")

# Process each split
for i, language in enumerate(languages):
    # Get labels for the current language
    labels = label_dict.get(language.lower(), None)
    if labels is None:
        raise ValueError(f"Error: Labels for language '{language}' not found.")
    
    # Repeat labels to match rows_per_split
    tiled_labels = np.tile(labels, rows_per_split // len(labels))
    if len(tiled_labels) != rows_per_split:
        raise ValueError(f"Error: Labels for '{language}' do not match the expected number of rows per split.")
    
    # Slice the PCA DataFrame for the current split
    split_df = df_pca.iloc[i * rows_per_split: (i + 1) * rows_per_split].reset_index(drop=True)
    
    # Add labels to the split DataFrame
    split_df['Labels'] = tiled_labels
    
    # Define save path for the split DataFrame
    save_path = f"C:/Users/MDilf/Desktop/Anka Datenbearbeitung/Feature Extraction/Features_Manual_Multiclass/Feature_Matrix_Multiclass_{language.capitalize()}_Split_{i + 1}.csv"
    
    # Save the split DataFrame to a CSV file
    split_df.to_csv(save_path, index=False)
    print(f"Saved: {save_path}")
        
        
        