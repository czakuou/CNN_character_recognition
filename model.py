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

import matplotlib.pyplot as plt

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

def model():
    # import Inception v3 model trained on imagenet
    base_model = tf.keras.applications.InceptionV3(
        input_shape=(96, 96, 3),
        include_top=False,
        weights="imagnet",)
    base_model.trainable = False
    
    # initialize top layers
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dropout(0.3),
        Dense(2, activation='softmax')
    ])
    
    
    
if __name__ == "__main__":
    main()
    
