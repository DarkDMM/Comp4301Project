# related scripts and functions for a CNN from tensorflow that will sort through the cropped images from 
#the object localizer to create a smaller subset of signs for the more complicated sign classifier

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def TrainFPSC(dataset):

    class_names = ["RoadSign", "No Sign"]

    print("Beginning FPSC training")
    FPSC = tf.keras.models.Sequential()
    FPSC.add(tf.keras.layers.Conv2D(32, (3,3), activation = 'relu', input_shape=(64,64,3)))
    FPSC.add(tf.keras.layers.MaxPooling2D((2,2)))
    FPSC.add(tf.keras.layers.Conv2D(64, (3,3), activation = 'relu'))
    FPSC.add(tf.keras.layers.MaxPooling2D((2,2)))
    FPSC.add(tf.keras.layers.Conv2D(64, (3,3), activation = 'relu'))

    FPSC.add(tf.keras.layers.Flatten())
    FPSC.add(tf.keras.layers.Dense(64, activation='relu'))
    FPSC.add(tf.keras.layers.Dense(10))

    #FPSC.summary()

    FPSC.compile(optimizer='adam',
                 loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                 metrics=['accuracy'])

    (train_img, train_lbl, test_img, test_lbl) = dataset.loadData((64,64,3))
    train_img = tf.convert_to_tensor(train_img)
    train_lbl = tf.convert_to_tensor(train_lbl)
    test_img = tf.convert_to_tensor(test_img)
    test_lbl = tf.convert_to_tensor(test_lbl)
    history = FPSC.fit(train_img, train_lbl, epochs = 30, validation_data = (test_img, test_lbl))
