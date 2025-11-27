#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 20 09:06:19 2025

@author: areuschel
"""

# Rocks Data Source
#### https://www.kaggle.com/datasets/neelgajare/rocks-dataset?resource=download


# Imports
# import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt



# Load Data

import os
data_dir = os.path.expanduser("~/Desktop/SplitData1020")

## Train, validation, test

train_ds = image_dataset_from_directory(
    os.path.join(data_dir, "train"),
    image_size=(224, 224),
    batch_size=32,
    color_mode = "rgb"
)

val_ds = image_dataset_from_directory(
    os.path.join(data_dir, "valid"),
    image_size=(224, 224),
    batch_size=32,
    color_mode = "rgb"
)

# checking
print(val_ds.class_names)


test_ds = image_dataset_from_directory(
    os.path.join(data_dir, "test"),
    image_size=(224, 224),
    batch_size=32,
    color_mode = "rgb"
)


# pretrained model from efficientNet not accepting RGB weights. try training from scratch

base_model = EfficientNetB0(weights=None, include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False


# classifier
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

## Train on subset

history = model.fit(train_ds, validation_data=val_ds, epochs=10)


# 

model.evaluate(test_ds)

# Visualize Performance

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

# Plot Accuracy
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs, acc, 'b-', label='Training Accuracy')
plt.plot(epochs, val_acc, 'r-', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Plot Loss
plt.subplot(1, 2, 2)
plt.plot(epochs, loss, 'b-', label='Training Loss')
plt.plot(epochs, val_loss, 'r-', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()
