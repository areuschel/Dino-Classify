#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Imports
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
import os

# Training, Validation, and Test Sets
train_dir = "Dino-Classify/fossil-classifier/data/processed/Split_Data_Full/train"
val_dir = "Dino-Classify/fossil-classifier/data/processed/Split_Data_Full/valid"
test_dir = "Dino-Classify/fossil-classifier/data/processed/Split_Data_Full/test"

# Pre-Processing
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(299, 299),
    batch_size=32,
    class_mode='binary'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(299, 299),
    batch_size=32,
    class_mode='binary'
)

test_generator = val_datagen.flow_from_directory(
    test_dir,                
    target_size=(299, 299),
    batch_size=54,
    class_mode='binary',
    shuffle=False             
)

# Inception Model
base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(299, 299, 3))
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
predictions = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Defining Recall (Thanks, Adrian)
def fossil_recall(y_true, y_pred):
    y_pred_bin = tf.cast(y_pred > 0.65, tf.float32)

    # fossils, positive class = 0
    true_positives = tf.reduce_sum(tf.cast((y_true == 0) & (y_pred_bin == 0), tf.float32))
    false_negatives = tf.reduce_sum(tf.cast((y_true == 0) & (y_pred_bin == 1), tf.float32))

    return true_positives / (true_positives + false_negatives + 1e-7)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', fossil_recall])

# Training Model
history = model.fit(
    train_generator,
    epochs=10,
    validation_data=val_generator
)

test_res = model.evaluate(test_generator)
print(test_res)

# Following Adrian's Lead
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

# More Plots!
y_pred_probs = model.predict(test_generator, steps=len(test_generator)).ravel()

y_true = test_generator.classes  # these are already integers (0 or 1)

thresholds = np.linspace(0.2, 0.6, 41)
recalls = []
precisions = []

for t in thresholds:
    preds = (y_pred_probs > t).astype(int)

    TP = np.sum((y_true == 0) & (preds == 0))  # fossils predicted correctly
    FP = np.sum((y_true == 1) & (preds == 0))  # rocks predicted as fossils
    FN = np.sum((y_true == 0) & (preds == 1))  # fossils missed

    recall = TP / (TP + FN + 1e-7)
    precision = TP / (TP + FP + 1e-7)

    recalls.append(recall)
    precisions.append(precision)

plt.figure(figsize=(10, 6))
plt.plot(1-thresholds, recalls, label="Recall (Fossils)")
plt.plot(1-thresholds, precisions, label="Precision (Fossils)")
plt.xlabel("Threshold")
plt.ylabel("Score")
plt.title("Precision & Recall vs Threshold")
plt.legend()
plt.show()






