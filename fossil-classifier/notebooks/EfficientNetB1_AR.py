#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 30 08:46:11 2025

@author: areuschel
"""

# Imports
# import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.applications import EfficientNetB1
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np



# Load Data

import os

data_dir = os.path.expanduser("~/Desktop/SplitData_Full")


## Train, validation, test

train_ds = image_dataset_from_directory(
    os.path.join(data_dir, "train"),
    image_size=(240, 240),
    batch_size=32,
    color_mode = "rgb"
)


test_ds = image_dataset_from_directory(
    os.path.join(data_dir, "test"),
    image_size=(240, 240),
    batch_size=32,
    color_mode = "rgb"
)

val_ds = image_dataset_from_directory(
    os.path.join(data_dir, "valid"),
    image_size=(240, 240),
    batch_size=32,
    color_mode = "rgb"
)


# pretrained model from efficientNet not accepting RGB weights. try training from scratch

base_model = EfficientNetB1(weights=None, include_top=False, input_shape=(240, 240, 3))
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
    y_pred_bin = tf.cast(y_pred > 0.60, tf.float32)

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




# same plot in EfficientNetB03_AR.py


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

