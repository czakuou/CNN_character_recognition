from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Nadam 

import tensorflow as tf

from sklearn.model_selection import train_test_split

import numpy as np

from img_preprocessing.data_loader import DataLoader
from model_hyperparameters.one_cycle_scheduler import OneCycleScheduler

import matplotlib.pyplot as plt

import os

# load train data
labels = ["aqua", "megumin", "ami", "yukino"]
data = DataLoader("./data/train/", labels).get_data()

# plot data
def plot_data(data, num):
    plt.figure(figsize = (5,5))
    plt.imshow(data[num][0])
    plt.title(labels[data[num][1]])
    plt.show()

# split data into train and validation sets
X = [item[0] for item in data]
y = [item[1] for item in data]
x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.33, random_state=42)

# Normalize the data
x_train = np.array(x_train, dtype=np.float32) / 255
x_val = np.array(x_val, dtype=np.float32) / 255

y_train = tf.keras.utils.to_categorical(y_train, 4)
y_val = tf.keras.utils.to_categorical(y_val, 4)

# data augumentation
datagen = ImageDataGenerator(
        rotation_range = 30,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.2, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip = True)  # randomly flip images

datagen.fit(x_train)

# import Inception v3 model trained on imagenet
def create_model():
    base_model = tf.keras.applications.InceptionV3(
        input_shape=(96, 96, 3),
        include_top=False,
        weights="imagenet")
    base_model.trainable = False

    # initialize top layers
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dropout(0.3),
        Dense(4, activation='softmax')
    ])

    # compile model
    model.compile(
        optimizer=Nadam(lr=0.001,
                        beta_1=0.9,
                        beta_2=0.999,
                        epsilon=1e-07),
        loss="categorical_crossentropy",
        metrics=["accuracy"])
    
    return model

# create model instance
model = create_model()

# Display the model's architecture
model.summary()

# model checkpoints
checkpoint_path = "./training/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# create the callback that saves the model's weights
model_checkpoint_callback =tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    save_weights_only=True,
    save_best_only=True,
    verbose=1)

# train model
history = model.fit(datagen.flow(x_train, y_train, batch_size=32),
                    epochs=500,
                    validation_data=(x_val, y_val),
                    callbacks=[model_checkpoint_callback])

# save model to .h5 file
model.save('my_model.h5')
