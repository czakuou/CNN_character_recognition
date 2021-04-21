# Transfer Learning for Characters Recognition
## Introduction

In this experiment, we will try to train an Neural Network using Transfer Learning, to recognize 4 anime characters.

![alt text](/img/temp1.jpg?raw=true)

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
- `Pooling`
- `Dropout`
- `Classification`

Nadam was the gradien optimizer that gave the best results

The total dataset wasn't big, so to avoid overfiting I've used data augumentation
```
datagen = ImageDataGenerator(
        rotation_range = 30,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.2, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip = True)  # randomly flip images
```

![alt text](/img/model_sum.png?raw=true)

|Image|Classification| OK/NG |
| --- | --- | --- |
| ![](/data/test/aqua/20.jpg) | Aqua (score = 0.4751119)<br>Megumin (score = 0.1749283)<br>Ami (score = 0.17504045)<br>Yukino (score = 0.17491941) | OK |
| ![](/data/test/aqua/23.jpg) | Aqua (score = 0.46201083)<br>Megumin (score = 0.18137312)<br>Ami (score = 0.17948475)<br>Yukino (score = 0.17713128) | OK |
| ![](/data/test/megumin/23.jpg) | Aqua (score = 0.17491232)<br>Megumin (score = 0.47514862)<br>Ami (score = 0.1750255)<br>Yukino (score = 0.17491359) | OK |
| ![](/data/test/megumin/24.jpg) | Aqua (score = 0.17491819)<br>Megumin (score = 0.4751372)<br>Ami (score = 0.17502083)<br>Yukino (score = 0.17492385) | OK |
| ![](/data/test/ami/18.jpg) | Aqua (score = 0.17777166)<br>Megumin (score = 0.17811155)<br>Ami (score = 0.4617612)<br>Yukino (score = 0.1823556) | OK |
| ![](/data/test/ami/19.jpg) | Aqua (score = 0.17548439)<br>Megumin (score = 0.17505354)<br>Ami (score = 0.4742613)<br>Yukino (score = 0.17561169) | OK |
| ![](/data/test/yukino/22.jpg) | Aqua (score = 0.17548439)<br>Megumin (score = 0.17577267)<br>Ami (score = 0.17670384)<br>Yukino (score = 0.4720391) | OK |
| ![](/data/test/yukino/25.jpg) | Aqua (score = 0.17924392)<br>Megumin (score = 0.18930177)<br>Ami (score = 0.1848114)<br>Yukino (score = 0.44664288) | OK |

Most of the restult on test set have treshold > 0.46, while the lowest threshold is 0.44.

```
1/1 - 1s - loss: 0.1443 - accuracy: 0.9545
Restored model, accuracy: 95.45%
```
The accurancy is pretty good, only one image got classivied badly
|Image|Classification| OK/NG |
| --- | --- | --- |
| ![](/data/test/megumin/22.jpg) | Aqua (score = 0.17869905)<br>Megumin (score = 0.18980493)<br>Ami (score = 0.1816511)<br>Yukino (score = 0.44984487) | NG |
