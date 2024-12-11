# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 19:59:53 2024

@author: SKV Hähnlein
"""
import pandas as pd
import numpy as np

# Load the CSV file
data = pd.read_csv('C:/Users/MDilf/Desktop/Anka Datenbearbeitung/Feature Extraction/Feature_SNN/Multiclass/Feature_Matrix_SNN.csv')

# Number of rows in each array
num_rows = 1600

# Split the DataFrame into 10 parts
split_dfs = [data.iloc[i * num_rows:(i + 1) * num_rows] for i in range(10)]

# Combine specified pairs
NAra_YAra = pd.concat([split_dfs[0], split_dfs[5]], ignore_index=True)  # NAra & YAra
NDan_YDan = pd.concat([split_dfs[1], split_dfs[6]], ignore_index=True)  # NDan & YDan
NEng_YEng = pd.concat([split_dfs[2], split_dfs[7]], ignore_index=True)  # NEng & YEng
NHin_YHin = pd.concat([split_dfs[3], split_dfs[8]], ignore_index=True)  # NHin & YHin
NJap_YJap = pd.concat([split_dfs[4], split_dfs[9]], ignore_index=True)  # NJap & YJap

# Save the combined DataFrames as new CSV files
NAra_YAra.to_csv('C:/Users/MDilf/Desktop/Anka Datenbearbeitung/Feature Extraction/Feature_SNN/Binary/SNN_Feature_Matrix_Arabic.csv', index=False)
NDan_YDan.to_csv('C:/Users/MDilf/Desktop/Anka Datenbearbeitung/Feature Extraction/Feature_SNN/Binary/SNN_Feature_Matrix_Danish.csv', index=False)
NEng_YEng.to_csv('C:/Users/MDilf/Desktop/Anka Datenbearbeitung/Feature Extraction/Feature_SNN/Binary/SNN_Feature_Matrix_English.csv', index=False)
NHin_YHin.to_csv('C:/Users/MDilf/Desktop/Anka Datenbearbeitung/Feature Extraction/Feature_SNN/Binary/SNN_Feature_Matrix_Hindi.csv', index=False)
NJap_YJap.to_csv('C:/Users/MDilf/Desktop/Anka Datenbearbeitung/Feature Extraction/Feature_SNN/Binary/SNN_Feature_Matrix_Japanese.csv', index=False)
print('Done!')
