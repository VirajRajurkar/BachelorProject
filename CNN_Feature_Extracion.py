# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 15:16:37 2024

@author: SKV Hähnlein
"""

from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import numpy as np
import pandas as pd

# Path to your data
base_dir = r'C:/Users/MDilf/Desktop/Anka Datenbearbeitung/Feature Extraction/STFT_images/All languages'

# Data generator
datagen = ImageDataGenerator(rescale=1./255)  # Normalize pixel values to [0, 1]
train_generator = datagen.flow_from_directory(
    base_dir,
    target_size=(224, 224),
    batch_size=32,
    color_mode='rgb',
    class_mode=None,  # No labels
    shuffle=False
)

# Load DenseNet121 model without the classification head
base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze all layers
for layer in base_model.layers:
    layer.trainable = False

# Add a global average pooling layer
x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)

# Feature extraction model
feature_extraction_model = tf.keras.models.Model(inputs=base_model.input, outputs=x)

# Initialize an empty list to store features
all_features = []

# Iterate over each batch in the generator
for i, batch in enumerate(train_generator):
    print(f"Processing batch {i + 1}/{len(train_generator)}")
    
    # Extract features for the current batch
    features = feature_extraction_model.predict(batch)
    
    # Print the shape of the features for this batch
    print(f"Batch {i + 1} features shape: {features.shape}")
    
    # Optionally, store the features for later use
    all_features.append(features)
    
    # Stop iteration when all batches are processed
    if i + 1 == len(train_generator):  # Total number of batches in the generator
        break

# Concatenate all features if needed
all_features = np.vstack(all_features)


# Convert the features to a DataFrame
features_df = pd.DataFrame(all_features)

#Create labels
NAra = np.full(1600, 'NAra')
NDan = np.full(1600, 'NDan')
NEng = np.full(1600, 'NEng')
NHin = np.full(1600, 'NHin')
NJap = np.full(1600, 'NJap')
YAra = np.full(1600, 'YAra')
YDan = np.full(1600, 'YDan')
YEng = np.full(1600, 'YEng')
YHin = np.full(1600, 'YHin')
YJap = np.full(1600, 'YJap')
labels = np.concatenate([NAra, NDan, NEng, NHin, NJap,
                               YAra, YDan, YEng, YHin, YJap])

# Add the labels as a new column in the DataFrame
features_df['labels'] = labels

# Step 5: Save the updated DataFrame
features_df.to_csv('C:/Users/MDilf/Desktop/Anka Datenbearbeitung/Feature Extraction/Feature_CNN/CNN_Feature_Matrix_MultiClass.csv', index=False)

# Display the first few rows of the updated DataFrame
print(features_df.head())































