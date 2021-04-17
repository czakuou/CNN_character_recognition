import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

labels = ["aqua", "megumin"]
class DataLoader:
    @staticmethod
    def get_data(data_dir):
        # initialize list for data
        data = []
        # loop through the labels
        for label in labels:
            path = os.path.join(data_dir, label)
            # give index to the label name
            class_num = labels.index(label)
            # loop through the all images
            for img in os.listdir(path):
                try:
                    # read and label images
                    img_arr = cv2.imread(os.path.join(path, img))[...,::-1]
                    data.append([img_arr, class_num])
                except Exception as e:
                    print(e)
        # return numpy array with labeled images
        return np.array(data, dtype=object)
    

data = ImageLabel()
data = data.get_data("./cropped/")

plt.figure(figsize = (5,5))
plt.imshow(data[30][0])
plt.title(labels[data[30][1]])
plt.show()