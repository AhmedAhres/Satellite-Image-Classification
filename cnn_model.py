#Import TensorFlow and Keras
import tensorflow as tf
from tensorflow import keras

#Import all the necessary for our model
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, TensorBoard
from keras.regularizers import l2
from keras.models import Sequential, load_model
from keras.layers import Activation, Dropout, Flatten, Dense, Conv2D, MaxPooling2D, LeakyReLU

#Import helper libraries
import numpy as np
import scipy as scipy

from helpers import *

class cnn_model:
    
    # Initialize the class
    def __init__(self, shape):
        self.shape = shape
        self.model = self.initialize_cnn_model(shape)
    
    def initialize_cnn_model(self, shape):
        
        # INPUT
        # shape     - Size of the input images
        # OUTPUT
        # model    - Compiled CNN
        
        # Define hyperparamters
        KERNEL3 = (3, 3)
        KERNEL5 = (5, 5)
        
        # Define a model
        model = Sequential()
        
        # Add the layers
        # Selection of the model is described in the report
        # We use padding = 'same' to avoid issues with the matrix sizes
        model.add(Conv2D(64, KERNEL5, input_shape = shape, padding='same'))
        model.add(LeakyReLU(alpha=0.1))
        model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
        model.add(Dropout(0.25))
        
        model.add(Conv2D(128, KERNEL3, padding='same'))
        model.add(LeakyReLU(alpha=0.1))
        model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
        model.add(Dropout(0.25))
        
        model.add(Conv2D(256, KERNEL3, padding='same'))
        model.add(LeakyReLU(alpha=0.1))
        model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
        model.add(Dropout(0.25))
        
        model.add(Conv2D(256, KERNEL3, padding='same'))
        model.add(LeakyReLU(alpha=0.1))
        model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
        model.add(Dropout(0.25))
        
        model.add(Conv2D(256, KERNEL3, padding='same'))
        model.add(LeakyReLU(alpha=0.1))
        model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
        model.add(Dropout(0.25))
        
        model.add(Conv2D(256, KERNEL3, padding='same'))
        model.add(LeakyReLU(alpha=0.1))
        model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
        model.add(Dropout(0.25))
        
        # Flatten it and use regularizers to avoid overfitting
        # The parameters have been chosen empirically
        model.add(Flatten())
        model.add(Dense(128, kernel_regularizer=l2(0.000001), activity_regularizer=l2(0.000001)))
        model.add(LeakyReLU(alpha=0.1))
        model.add(Dropout(0.5))
        
        # Add output layer
        model.add(Dense(2, kernel_regularizer=l2(0.000001), activity_regularizer=l2(0.000001)))
        model.add(Activation('sigmoid'))
        
        # Compile the model using the binary crossentropy loss and the Adam optimizer for it
        # We used the accuracy as a metric, but F1 score is also a plausible choice
        model.compile(loss='binary_crossentropy',
                      optimizer=Adam(lr=0.001),
                      metrics=['accuracy'])
            
        # Print a summary of the model to see what has been generated
        model.summary()
                      
        return model
    
    def train(self):
        
        # We define the number of epochs and steps per epochs
        EPOCHS = 150
        STEPS_PER_EPOCH = 1500
        
        # Early stopping callback after 10 steps
        early_stopping = EarlyStopping(monitor='loss', min_delta=0, patience=10, verbose=1, mode='auto')
        
        # Reduce learning rate on plateau after 4 steps
        lr_callback = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=4, verbose=1, mode='auto')
        
        # Tensorboard to visualize the training
        tensorboard = keras.callbacks.TensorBoard(log_dir='./log_last_test', histogram_freq=0, batch_size=125, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None)
        
        # Place the callbacks in a list to be used when training
        callbacks = [early_stopping, lr_callback, tensorboard]
        
        # Train the model using the previously defined functions and callbacks
        self.model.fit_generator(create_minibatch(), steps_per_epoch=STEPS_PER_EPOCH, epochs=EPOCHS,\
                                 use_multiprocessing=False, workers=1, callbacks=callbacks, verbose=1)
    
    def classify(self, X):
        # Subdivide the images into blocks with a stride and patch_size of 16
        img_patches = create_patches(X, 16, 16, padding=28)
        
        # Predict
        predictions = self.model.predict(img_patches)
        predictions = (predictions[:,0] < predictions[:,1]) * 1
        
        # Regroup patches into images
        return group_patches(predictions, X.shape[0])
    
    def load(self, filename):
        # Load the model (used for submission)
        self.model = load_model(filename)
    
    def save(self, filename):
        # Save the model (used to then load to submit)
        self.model.save(filename)

