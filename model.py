from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Nadam 

import tensorflow as tf

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

import numpy as np

from img_preprocessing.data_loader import DataLoader
from model_hyperparameters.one_cycle_scheduler import OneCycleScheduler

import matplotlib.pyplot as plt

import os

# load train and test data
labels = ["aqua", "megumin"]
train = DataLoader("./data/train/", labels).get_data()
test = DataLoader("./data/test/", labels).get_data()

# plot data
def plot_data(data, num):
    plt.figure(figsize = (5,5))
    plt.imshow(data[num][0])
    plt.title(labels[data[num][1]])
    plt.show()

# split data into train and validation sets
X = [item[0] for item in train]
y = [item[1] for item in train]
x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.33, random_state=42)

# testing data
x_test = [item[0 for item in test]
y_test = [item[1] for item in test]

# Normalize the data
x_train = np.array(x_train, dtype=np.float32) / 255
x_val = np.array(x_val, dtype=np.float32) / 255

y_train = np.array(y_train, dtype=np.float32)
y_val = np.array(y_val, dtype=np.float32)

# data augumentation
datagen = ImageDataGenerator(
        rotation_range = 30,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.2, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip = True)  # randomly flip images

datagen.fit(x_train)
# %%
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
        Dense(2, activation='softmax')
    ])

    # compile model
    model.compile(
        optimizer=Nadam(lr=0.001,
                        beta_1=0.9,
                        beta_2=0.999,
                        epsilon=1e-07),
        loss="categorial_crossentropy",
        metrics=["categorial_crossentropy_accuracy"])
    
    return model

# create model instance
model = create_model()

# Display the model's architecture
model.summary()

# model checkpoints
checkpoint_filepath = "./training/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# create the callback that saves the model's weights
model_checkpoint_callback =tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1)

# # train model
history = model.fit(x_train,
                    y_train,
                    epochs=500,
                    validation_data=(x_val, y_val),
                    batch_size=32,
                    callbacks=[model_checkpoint_callback])

# Loads the weights
model.load_weights(checkpoint_path)

# Evaluate the model
loss, acc = model.evaluate(x_test, x_train, verbose=2)
print("Trained Model, accuracy: {:5.2f}%".format(100 * acc))

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(500)

plt.figure(figsize=(15, 15))
plt.subplot(2, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(2, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()


if __name__ == "__main__":

