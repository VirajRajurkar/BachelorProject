# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 16:50:55 2024

@author: SKV Hähnlein
"""

from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import Input, Lambda, Flatten, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import tensorflow as tf
from scipy.spatial.distance import cdist
import numpy as np
import pandas as pd
import random

# Path to your data
base_dir = r'C:/Users/MDilf/Desktop/Anka Datenbearbeitung/Feature Extraction/STFT_images/All languages'

# Data generator
datagen = ImageDataGenerator(rescale=1./255)
train_generator = datagen.flow_from_directory(
    base_dir,
    target_size=(224, 224),
    batch_size=32,
    color_mode='rgb',
    class_mode='categorical',
    shuffle=False
)

# Load the DenseNet121 model
base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze the base model

def create_embedding_model():
    input_layer = Input(shape=(224, 224, 3))
    x = base_model(input_layer)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)  # Replace Flatten with GAP
    x = Dropout(0.5)(x)  # Dropout to prevent overfitting
    return Model(input_layer, x)

embedding_model = create_embedding_model()

# Define the Siamese network
input_1 = Input(shape=(224, 224, 3))
input_2 = Input(shape=(224, 224, 3))
embedding_1 = embedding_model(input_1)
embedding_2 = embedding_model(input_2)

# Euclidean distance
def euclidean_distance(vectors):
    x, y = vectors
    x = tf.keras.backend.l2_normalize(x, axis=-1)
    y = tf.keras.backend.l2_normalize(y, axis=-1)
    return tf.keras.backend.sqrt(tf.keras.backend.sum(tf.keras.backend.square(x - y), axis=1, keepdims=True))

distance = Lambda(euclidean_distance, name="Distance_Layer")([embedding_1, embedding_2])
output = Dense(1, activation='linear', name="Output_Layer")(distance)
siamese_model = Model(inputs=[input_1, input_2], outputs=output)

# Contrastive loss
def contrastive_loss(y_true, y_pred):
    margin = 1
    return tf.keras.backend.mean(y_true * tf.keras.backend.square(y_pred) + 
                                 (1 - y_true) * tf.keras.backend.square(tf.keras.backend.maximum(margin - y_pred, 0)))

siamese_model.compile(optimizer='adam', loss=contrastive_loss, metrics=['mean_squared_error'])
print('Model adjusted and compiled')

# Custom pair generator for tf.data compatibility
def pair_generator(images, labels, batch_size):
    def generator():
        while True:
            pairs, pair_labels = [], []
            for _ in range(batch_size):
                if random.random() > 0.5:  # Similar pair
                    class_idx = random.choice(np.unique(labels))
                    idx1, idx2 = random.sample(list(np.where(labels == class_idx)[0]), 2)
                    pair_labels.append(1)
                else:  # Dissimilar pair
                    idx1 = random.choice(range(len(labels)))
                    idx2 = random.choice([i for i in range(len(labels)) if labels[i] != labels[idx1]])
                    pair_labels.append(0)
                pairs.append((images[idx1], images[idx2]))
            # Convert to tensors
            pair_input_1 = np.array([pair[0] for pair in pairs])
            pair_input_2 = np.array([pair[1] for pair in pairs])
            pair_labels = np.array(pair_labels)
            yield (pair_input_1, pair_input_2), pair_labels

    return generator

# Create a tf.data.Dataset from the generator
def create_tf_dataset(images, labels, batch_size):
    generator = pair_generator(images, labels, batch_size)
    output_signature = (
        (tf.TensorSpec(shape=(None, 224, 224, 3), dtype=tf.float32),
         tf.TensorSpec(shape=(None, 224, 224, 3), dtype=tf.float32)),
        tf.TensorSpec(shape=(None,), dtype=tf.int32)
    )
    dataset = tf.data.Dataset.from_generator(generator, output_signature=output_signature)
    return dataset

# Load images and labels
all_images, all_labels = [], []
for batch, labels in train_generator:
    all_images.append(batch)
    all_labels.extend(np.argmax(labels, axis=1))
    if len(all_labels) >= 16000:  # Stop after 16,000 samples
        break

all_images = np.concatenate(all_images, axis=0)
all_labels = np.array(all_labels)

# Train-validation split
images_train, images_val, labels_train, labels_val = train_test_split(
    all_images, all_labels, test_size=0.2, stratify=all_labels, random_state=42
)


# Create tf.data Datasets
batch_size = 32
train_dataset = create_tf_dataset(images_train, labels_train, batch_size)
val_dataset = create_tf_dataset(images_val, labels_val, batch_size)

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)

# Train the Siamese network
history = siamese_model.fit(
    train_dataset,
    steps_per_epoch=len(images_train) // batch_size,
    validation_data=val_dataset,
    validation_steps=len(images_val) // batch_size,
    epochs=50,
    callbacks=[early_stopping, reduce_lr]
)

print("Training complete!")

# --- Feature Extraction ---
# Step 1: Use the trained embedding model to extract embeddings
embeddings = embedding_model.predict(all_images)  # Shape: (16000, 1024)


distance_matrix = cdist(embeddings, embeddings, metric='euclidean')

# Step 3: Save the distance matrix with labels
np.save('distance_matrix.npy', distance_matrix)
np.save('labels.npy', all_labels)

