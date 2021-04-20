
import tensorflow as tf
from img_preprocessing.data_loader import DataLoader

import numpy as np

import matplotlib.pyplot as plt


labels = ["aqua", "megumin", "ami", "yukino"]
model = tf.keras.models.load_model('./training/my_model.h5')

model.summary()

test = DataLoader("./data/test/", labels).get_data()

# test data
x_test = [item[0] for item in test]
y_test = [item[1] for item in test]

x_te = tf.constant(x_test, dtype=np.float32) / 255
y_test = tf.keras.utils.to_categorical(y_test, 4)

loss, acc = model.evaluate(x_te, y_test, verbose=2)
print('Restored model, accuracy: {:5.2f}%'.format(100 * acc))

predictions = model.predict(x_te)
print(predictions)

label = tf.nn.softmax(predictions)
print(label)


print(y_test)

print(len(y_test))

print(21/22)

plt.figure(figsize = (5,5))
plt.imshow(test[-7][0])
plt.title(labels[-2])
plt.show()





