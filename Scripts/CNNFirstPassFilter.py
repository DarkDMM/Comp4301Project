# related scripts and functions for a CNN from tensorflow that will sort through the cropped images from 
#the object localizer to create a smaller subset of signs for the more complicated sign classifier

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from Localization import LibraryLocalization
import cv2

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
    FPSC.add(tf.keras.layers.Dense(2))
    FPSC.add(tf.keras.layers.Softmax())

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
    FPSC.save("Models\\FPSC")
    return FPSC

def imgProcess(Image, TrainedModel = None):
    if(TrainedModel == None):
        #attempt to load a trained FPSC model if a model is not specified for this function
        Model = tf.keras.models.load_model("Models\\FPSC")

    SegmentedData = LibraryLocalization(Image, True, False)

    ImageWithBoxes = cv2.imread(Image).copy()
    for item in SegmentedData:
        img = np.expand_dims(item[0], axis=0)
        Lbl = np.array(Model(img, training = False)).argmax()
        if(Lbl == 1):
            x,y,w,h = item[1]
            x = x * 2
            w = w * 2
            y = y * 2
            h = h * 2
            cv2.rectangle(ImageWithBoxes, (x,y), (x+w, y+h), (0, 255, 0), 1, cv2.LINE_AA)
    while True:
        cv2.imshow("Localized Signs",ImageWithBoxes)

        k = cv2.waitKey(0) & 0xff
        #q is pressed
        if k == 113:
            break