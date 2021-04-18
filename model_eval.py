import tensorflow as tf
from img_preprocessing.data_loader import DataLoader

import numpy as np


labels = ["aqua", "megumin", "ami", "yukino"]
model = tf.keras.models.load_model('./training/my_model.h5')

model.summary()

test = DataLoader("./data/test/", labels).get_data()

# test data
x_test = [item[0] for item in test]
y_test = [item[1] for item in test]

x_test = np.array(x_test, dtype=np.float32) / 255
y_test = tf.keras.utils.to_categorical(y_test, 4)

loss, acc = model.evaluate(x_test, y_test, verbose=2)
print('Restored model, accuracy: {:5.2f}%'.format(100 * acc))

