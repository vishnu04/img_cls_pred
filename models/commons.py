# -*- coding: utf-8 -*-
"""
preprocessing of the data
data: tomato images
"""

import tensorflow as tf
from tensorflow.keras import models, layers, utils
import matplotlib.pyplot as plt
from params import SEED_INIT, TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT,IMAGE_SIZE,SHUFFLE_SIZE
from math import ceil
import numpy as np 

def get_partition_data(ds, train_split=TRAIN_SPLIT, val_split = VAL_SPLIT, test_split = TEST_SPLIT, shuffle = True, shuffle_size = SHUFFLE_SIZE):
    ds_size = len(ds)
    train_size = int(ds_size * train_split)
    val_size = int(ds_size * val_split)
    if shuffle:
        ds = ds.shuffle(shuffle_size, seed = SEED_INIT)
    train_ds = ds.take(train_size)
    val_ds = ds.skip(train_size).take(val_size)
    test_ds = ds.skip(train_size).skip(val_size)
    return train_ds, val_ds, test_ds

def resize_rescale_layer():
    resize_rescale = tf.keras.Sequential([
            layers.experimental.preprocessing.Resizing(IMAGE_SIZE,IMAGE_SIZE),
            layers.experimental.preprocessing.Rescaling(1.0/255)
        ])
    return resize_rescale

def data_augmentation():
    data_augmentation = tf.keras.Sequential([
            layers.experimental.preprocessing.RandomFlip('horizontal_and_vertical'),
            layers.experimental.preprocessing.RandomRotation(0.2)
        ])
    return data_augmentation

def cache_prefetch(ds):
    return ds.cache().shuffle(SHUFFLE_SIZE).prefetch(buffer_size = tf.data.AUTOTUNE)

def plot_images(tf_ds,class_names,no_of_images=12):
    plt.figure(figsize=(20,20))
    n_x_img = ceil(no_of_images/4)
    for image_batch, label_batch in tf_ds:
        for i in range(no_of_images):
            ax = plt.subplot(n_x_img,4,i+1)
            plt.imshow(image_batch[i].numpy().astype("uint8"))
            plt.axis('off')
            plt.title(class_names[label_batch[i]])
    return plt

def predict(model, img, class_names):
    img_array = tf.keras.preprocessing.image.img_to_array(img.numpy())
    img_array = tf.expand_dims(img_array,0) #create a batch
    predictions = model.predict(img_array)
    #print(predictions)
    #print(np.argmax(predictions[0]))
    predicted_class = class_names[np.argmax(predictions[0])]
    #print(predicted_class)
    confidence = round(100 * np.max(predictions[0]),2)
    #print(confidence)
    return predicted_class, confidence