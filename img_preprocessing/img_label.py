import cv2
import os
import numpy as np

class DatasetLoader:
    def __init__(self, preporcesor=None):
        # store the image preprocessor
        self.preporcesor = preporcesor
        
        # if the preprocessor if None, initialize then as an
        # empty array
        if self.preporcesor is None:
            self.preporcesor = []
    
    def load(self, imagePaths, verbose=-1):
        # initialize the list of features and labels
        data = []
        labels = []
        
        # loop over the input images
        for (i, imagePath) in enumerate(imagePaths):
            # load images and extract the class label
            # assuming our path has following format:
            # path/dataset/{class}/{image}.jpg
            image = cv2.imread(imagePath)
            label = imagePath.split(os.path.sep)[-2]
            
            # check if our preprocessors are not None
            if self.preporcesor is not None:
                # loop over preprocessors and apply each to
                # the image
                for p in self.preporcesor:
                    image = p.preprocess(image)
            
            # treat preporcess image as feature vector
            # update data list followed by the labels
            data.append(image)
            label.append(image)
            
            # show an update every "verbose" images
            if verbose > 0 and i > 0 and (i + 1) % verbose ==0:
                print("INFO processed {}/{}".format(i+1, len(imagePaths)))
                
            # return a tuple of the data and labels
            return (np.array(data), np.array(labels))