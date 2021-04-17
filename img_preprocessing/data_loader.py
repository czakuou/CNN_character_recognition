import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import traceback

labels = ["aqua", "megumin"]
class DataLoader:
    '''
    Load and label images from folders
    Parameters:
    data_dor: path to folder with dirs with data
    '''
    def __init__(self, data_dir):
        self.data_dir = data_dir
        # initialize list for data
        self.data = []
        
    def get_data(self):
        # loop through the labels
        for label in labels:
            path = os.path.join(self.data_dir, label)
            # give index to the label name
            class_num = labels.index(label)
            # loop through the all images
            for img in os.listdir(path):
                try:
                    # read and label images
                    img_arr = cv2.imread(os.path.join(path, img))[...,::-1]
                    self.data.append([img_arr, class_num])
                except Exception:
                    print(traceback.format_exc())
        # return numpy array with labeled images
        return np.asarray(self.data, dtype=object)
    
    def plot(self, img_num):
        # plot labeled images
        # img_num: number of an image
        plt.figure(figsize = (5,5))
        plt.imshow(self.data[img_num][0])
        plt.title(labels[self.data[img_num][1]])
        plt.show()

if __name__ == "__main__":
    main()
    