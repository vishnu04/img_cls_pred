# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 13:42:19 2022

@author: vishnu
"""

from tensorflow.keras import models, layers, utils
from commons import get_partition_data,cache_prefetch,plot_images, resize_rescale_layer, data_augmentation, predict
from params import BATCH_SIZE, IMAGE_SIZE, SEED_INIT, data_loc, CHANNELS, EPOCHS, SHUFFLE_SIZE
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os

dataset = utils.image_dataset_from_directory(data_loc,
                                              labels='inferred',
                                              label_mode='int',
                                              class_names=None, 
                                              color_mode='rgb',
                                              seed = SEED_INIT,
                                              batch_size=BATCH_SIZE,
                                              shuffle = True,
                                              image_size=(IMAGE_SIZE,IMAGE_SIZE))

class_names = dataset.class_names

train_ds, val_ds, test_ds = get_partition_data(dataset)

#train_ds, val_ds, test_ds = tfds.load('dataset', split=['train', 'validation[10%]','test[10%]'])

print(f'{len(train_ds)}:{len(val_ds)}:{len(test_ds)}')

# ## cache and shuffle and prefetch
train_ds = cache_prefetch(train_ds)
val_ds = cache_prefetch(val_ds)
test_ds = cache_prefetch(test_ds)

## for plotting images (default no of images = 12)
#plot_images(train_ds.take(1),class_names)
input_shape = (BATCH_SIZE,IMAGE_SIZE,IMAGE_SIZE,CHANNELS)
n_classes = len(class_names)
model = models.Sequential([
        resize_rescale_layer(),
        data_augmentation(),
        layers.Conv2D(32,kernel_size = (3,3), activation='relu', input_shape = input_shape),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(32,kernel_size = (3,3), activation = 'relu'),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(64,kernel_size = (3,3), activation = 'relu'),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(64,kernel_size = (3,3), activation = 'relu'),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(64,kernel_size = (3,3), activation = 'relu'),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(64,kernel_size = (3,3), activation = 'relu'),
        layers.MaxPooling2D((2,2)),
        layers.Flatten(),
        layers.Dense(64, activation = 'relu'),
        layers.Dense(n_classes, activation = 'softmax')
    ])

model.build(input_shape=input_shape)

model.summary()

model.compile(
    optimizer='adam',
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy'])

history = model.fit(train_ds,
                    epochs = EPOCHS,
                    batch_size = BATCH_SIZE,
                    verbose = 1,
                    validation_data = val_ds)

scores = model.evaluate(test_ds)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']


plt.figure(figsize=(8,8))
plt.subplot(1,2,1)
plt.plot(range(EPOCHS), acc, label = 'Training Accuracy')
plt.plot(range(EPOCHS), val_acc, label = 'Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1,2,2)
plt.plot(range(EPOCHS), loss, label = 'Training Loss')
plt.plot(range(EPOCHS), val_loss, label = 'Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

plt.figure(figsize=(15,15))
for images, labels in test_ds.take(1):
    for i in range(12):
        ax = plt.subplot(3,4, i+1)
        plt.imshow(images[i].numpy().astype('uint8'))
        predicted_class, confidence = predict(model, images[i], class_names)
        actual_class = class_names[labels[i]]
        plt.title(f"Acutal: {actual_class} \n Predicted: {predicted_class} \n Confidence: {confidence}")
        plt.axis("off")

model_version = "tfmodel"+str(len(next(os.walk('.'))[1]) + 1)
model.save(f"../models/{model_version}")
