# Transfer Learning for Characters Recognition
## Introduction

In this experiment, we will try to train an Neural Network using Transfer Learning, to recognize 4 anime characters.

![alt text](https://github.com/czakuou/CNN_character_recognition/blob/main/img/temp1.jpg?raw=true)

This experiment has 5 main steps:
  1. Collect data
  2. Recognize character face from each images
  3. Image preprocessing
  4. Split images into training and test sets
  5. Create the final model
  
`data/raw` directory contains 26 images for each character. The first 20 are used for training while the last 6 are used for test
`data/cropped` contains images after applaying `lbpcascade_animeface` with `convert.py` script. It can detect character faces with *83%* accuracy.
Next the script rename and resize the images to 96x96 px.

## Steps
1. The following command is udes to populate `cropped` directory
```
$ python convert.py raw/[character_name] croppend <rename option 0=False 1=True>
```
After that, you can decide how many images will be used for `training` and `test` datasets
2. In `training/my_model.h5` you can find the saved model with trained weights
3. By editing `labels` and `directory` of your images in `model.py`, you can train it on your own data

## Building the model
I've decided to use pretrained Inception V3 layers pretrained on ImageNet. After that I've builded top layers that included
`
pooling
`
`
Dropout
`
`
Classification
`
Nadam was the gradien optimizer that gave the best results

