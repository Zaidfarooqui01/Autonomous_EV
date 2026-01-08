#!/usr/bin/env python
# coding: utf-8

# In[2]:


import tensorflow as tf
import numpy as np
import pickle
import pandas as pd
import random
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical    
import os
os.chdir(r"D:\AEV_Data")


# In[22]:


training_file = r"D:\AEV_Data\AEV_Datasets\train.p"
validation_file = r"D:\AEV_Data\AEV_Datasets\valid.p"
testing_file = r"D:\AEV_Data\AEV_Datasets\test.p"

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)

X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']

print(f"Train: {X_train.shape}, Valid: {X_valid.shape}, Test: {X_test.shape}")

# Normalize images (keeping shape 32x32x3)
X_train = (X_train -128) / 128.0
X_valid = (X_valid -128) / 128.0
X_test = (X_test -128) / 128.0

# One-hot encode labels
y_train_cat = to_categorical(y_train, num_classes=43)
y_valid_cat = to_categorical(y_valid, num_classes=43)
y_test_cat = to_categorical(y_test, num_classes=43)

# Build CNN Model
model = models.Sequential()

# 1st Convolutional Layer
model.add(layers.Conv2D(filters=6, kernel_size=(5,5), strides=(1,1), padding='valid', activation='relu', input_shape=(32,32,3)))
model.add(layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)))    #input is (28,28,6)   output(14,14,6)
# model.add(layers.Dropout(0.25))  # ← Add dropout after pooling

# 2nd Convolutional Layer
model.add(layers.Conv2D(filters=16, kernel_size=(5,5), strides=(1,1), padding='valid', activation='relu')) #input is (14,14,6)  output is(10,10,16)
model.add(layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)))#input is 10,10,16   output is 10,10, 16


# Flatten
model.add(layers.Flatten())

# Fully Connected Layers
model.add(layers.Dense(120, activation='relu'))
# model.add(layers.Dropout(0.5))  # ← Higher dropout for dense layers
model.add(layers.Dense(80, activation='relu'))
# model.add(layers.Dropout(0.5))  # ← Higher dropout for dense layers
model.add(layers.Dense(43, activation='softmax'))  # ← Change to softmax Output layer (43 classes)

# Compile Model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Training Parameters
batch_size = 256
epochs = 10

# Training with Evaluation after Each Epoch
history = model.fit(X_train, y_train_cat,
                    validation_data=(X_valid, y_valid_cat),
                    epochs=epochs,
                    batch_size=batch_size)

# Evaluate on Test Set
test_loss, test_acc = model.evaluate(X_test, y_test_cat)
print(f"Final Test Accuracy: {test_acc*100:.2f}%")

# Load Sign Names CSV
sign_names = pd.read_csv(r"D:\AEV_Data\AEV_Datasets\signnames.csv")



# In[28]:


# Predict Random Test Image
random_idx = random.randint(0, len(X_test)-1)
sample_image = X_test[random_idx]
true_label = y_test[random_idx]

plt.imshow(sample_image)
plt.title("Test Image")
plt.show()

# Expand dims for prediction
sample_input = np.expand_dims(sample_image, axis=0)
pred_probs = model.predict(sample_input)
pred_label = np.argmax(pred_probs)

# Get Class Names
pred_name = sign_names.loc[pred_label]['SignName']
true_name = sign_names.loc[true_label]['SignName']

print(f"Predicted Label: {pred_label} -> {pred_name}")
print(f"True Label: {true_label} -> {true_name}")


# In[ ]:




