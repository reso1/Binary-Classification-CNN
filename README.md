## Introduction
This is a **binary classifier** of a course project based on **AlexNet**, which classifies images of units into qualified one and unqualified one.

The AlexNet implementation with tensorflow is basically forked from (), which includes the `alexnet.py` and `data_generator.py`. 

I've also made modifications on the training file (`train.py`) and testing file (`test.py`) to be adapted to the project images.

The main work I've done focused on the preprocessing of input images and generating units of good quality (`utils.py`), the process includes image segmentation, augmentation, template matching, hough line detections, all of which is implemented using opencv. 

## CNN Model
Original AlexNet, paper can be found in [HERE](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf).

## Dataset
+ The preprocess is done seperatly on each of the units after segmentation.

+ Apart from the preprocessing mentioned above, I've done random brightness, contrast transformation using tensorflow to enlarge the original training data by 4 times.

## Requirements
+ **Python3.x** environment with installed packages of:
    + tensorflow
    + numpy
    + opencv
    + opencv-python

+ Pre-trained AlexNet model files.

+ Trained model files.

## Usage
1. Put the original test images on `./images/` folder.

2. Change directory to `./src` by as followed:

    `$cd ./src`

3. Start classfication by as followed:

    `$python test.py`

*After the process is over with images on `./images/`, you should expected to get a classfication output written in `./result.txt`.*