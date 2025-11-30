#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 27 13:11:47 2025

@author: areuschel
"""


# Imports
# import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np



# Load Data

import os


#### ------------ Data from first model, EfficientNet_AR.py ------- ####
## Rock photos obtained from online sources

## data_dir = os.path.expanduser("~/Desktop/SplitData1020")


#### ------------ Data from second model, EfficientNet2_AR.py ----- ####
## Rock photos obtained from lab

## data_dir = os.path.expanduser("~/Desktop/SplitData1127")


#### ------------ New Data, Lab+Online ---------------------------- ####
## Rock photos from both lab and online

data_dir = os.path.expanduser("~/Desktop/SplitData_Full")



## Train, validation, test

train_ds = image_dataset_from_directory(
    os.path.join(data_dir, "train"),
    image_size=(224, 224),
    batch_size=32,
    color_mode = "rgb"
)


# checking
print(train_ds.class_names)


test_ds = image_dataset_from_directory(
    os.path.join(data_dir, "test"),
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



# pretrained model from efficientNet not accepting RGB weights. try training from scratch

base_model = EfficientNetB0(weights=None, include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False


# classifier
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(1, activation='sigmoid')
])

### new- adding function to calc recall for each sample
## why? keras does recall on class 1 and we have fossils as class 0

def fossil_recall(y_true, y_pred):
    y_pred_bin = tf.cast(y_pred > 0.65, tf.float32)

    # fossils, positive class = 0
    true_positives = tf.reduce_sum(tf.cast((y_true == 0) & (y_pred_bin == 0), tf.float32))
    false_negatives = tf.reduce_sum(tf.cast((y_true == 0) & (y_pred_bin == 1), tf.float32))

    return true_positives / (true_positives + false_negatives + 1e-7)



model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', fossil_recall])

## Train

history = model.fit(train_ds, validation_data = val_ds, epochs=10)

# 

test_res = model.evaluate(test_ds)
print(test_res)

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

## more plots!


y_true = np.concatenate([y for x, y in test_ds], axis=0)
y_pred_probs = model.predict(test_ds).ravel()


thresholds = np.linspace(0.4, 0.8, 41)  # 0.4 → 0.8 in steps of 0.01
recalls = []
precisions = []

for t in thresholds:
    preds = (y_pred_probs > t).astype(int)

    TP = np.sum((y_true == 0) & (preds == 0))   # fossils predicted correctly
    FP = np.sum((y_true == 1) & (preds == 0))   # rocks predicted as fossils
    FN = np.sum((y_true == 0) & (preds == 1))   # fossils missed

    recall = TP / (TP + FN + 1e-7)
    precision = TP / (TP + FP + 1e-7)

    recalls.append(recall)
    precisions.append(precision)

plt.figure(figsize=(10, 6))
plt.plot(thresholds, recalls, label="Recall (Fossils)")
plt.plot(thresholds, precisions, label="Precision (Fossils)")
plt.xlabel("Threshold")
plt.ylabel("Score")
plt.title("Precision & Recall vs Threshold (Fossils = class 0)")
plt.legend()
plt.show()
