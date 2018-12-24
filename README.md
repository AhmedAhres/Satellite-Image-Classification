# Road Segmentation - Ahmed Ahres, Ali El Abridi, Khalil Elleuch

## Important:
Due to the restricted size of 100MB to upload, please make sure that there is a folder called "provided" in which there are 2 folders:

- training: containing the training images (1 folder for the images and one for the grountruth as provided)
- test_set_images: containing the test images. This folder already exists (to be able to run the run.py), however the training images need to be added manually to the "provided" to run the jupyter notebook training.ipynb.

## Description: 

This repository contains the design and implementation of a convolutional neural networks to classify satellite images. More specifically, the goal is to separate 16x16 blocks of pixels between roads and the rest. 
The training set consists of 100 satellite images (400x400) with their respective ground truth. The testing set consists of 50 satellite images (608x608).

## Libraries: 
This project uses 2 fundamental libraries that need to be installed in order to run it:
- TensorFlow 1.12.0
- Keras 2.2.4
- Numpy 1.15.1
- Scipy 1.1.0
- Matplotlib 3.0.2

## Technical Details: 
- Overall architecture: The neural network is fed mini-batches of 72x72 pixels from the input images. The mini-batches are created in the generate_minibatch() function. The data augmentation is also done within the same method, on the generated batches.
- Callbacks: The model uses two callback function: EarlyStopping and ReduceLROnPlateau. EarlyStopping is used to stop the training when the loss stops decreasing. In this case, the patience is 10 steps. ReduceLROnPlateau is used to reduce the learning rate when the loss stops decreasing. In this case, the patience is 4 steps. As a result, if the model stops improving, it first reduces the learning rate. If after 4 additional steps it still does not improve, then it reduces it again. If there is still no improvement, the model stops and is returned.
- Classification: After the training, the classification is done on 16x16 pixels. First, we split the test images into 16x16 pixels. Then, we use the methods predict() and classify() in order to return a result (0 for background and 1 for road).

## Training Hardware: 

The training was done on a private server create using the Google Cloud Platform. The is intel® optimized Deep Learning Image: TensorFlow 1.12.0 m14 (with Intel® MKL-DNN/MKL and CUDA 10.0)
- Debian:  intel® optimized Deep Learning Image: TensorFlow 1.12.0 m14 (with Intel® MKL-DNN/MKL and CUDA 10.0) 
- GPU: 1 x NVIDIA Tesla P100 (16GB CoWoS HBM2 at 732 GB/s)
- CPU: 8 vCPU
- RAM: 52 Go

## Files:
* **run.py** : contains the steps to do to run our project and have a csv file submission in the end. In order to use this, type in the command line python3 run.py (or python run.py). 
* **final_model.h5** : model trained with result 89.5%.
* **final_submission.csv** : csv file with result 89.5%, generated through final_model.h5.
* **helpers.py** : contains all the utilities functions used by the neural network.
* **training.ipynb** : notebook containing an example of the training steps using the model. It also contains the architecture of the neural network, including the minibatch generating and data augmentation.
