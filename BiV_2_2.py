#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 11:35:04 2024

@author: user
"""

# -*- coding: utf-8 -*-
"""
Übung 2 (auf den MNIST Bildern): 
    
    - Berechnung des Gradienten (Differenz in x- und y-Richtung)
    - Berechnung eines Gradientenhistogramms für jedes Label

Re-used Functions from previous Exercises:
    get_mnist_data()

New Functions:
    get_gradient_direction_histograms()

* should go into "my_utilities" module for later re-use

Created on Mon Jan 31 12:56:33 2022

@author: Klaus Toennies
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

def get_mnist_data(n_train=60000):
    
    # read MNIST data from OpenML
    print('fetching MNIST data (be patient, this takes a minute!)')
    mnist = fetch_openml('mnist_784')
    data  = np.array(mnist.data)
    target= (np.array(mnist.target)).astype('uint8')
    
    # separate data into training and test data and reshape to pictures
    X_train, X_test, y_train, y_test = train_test_split(data,target,
                                                        train_size=n_train)
    X_train = (np.reshape(X_train,(X_train.shape[0],28,28)))
    X_test  = (np.reshape(X_test,(X_test.shape[0],28,28)))
    
    return((X_train, y_train),(X_test,y_test))

def get_dir_histograms(images,labels,n_bins=18):
    
    # compute gradient by difference in x and y
    edges_x = images-np.roll(images, 1, axis=1)
    edges_y = images-np.roll(images, 1, axis=2)
    
    # edges will be weighted by their length in the histogram
    length     = (edges_x**2+edges_y**2)**0.5
    
    # compute gradient angles for all pixels
    grad_angles= 180.*np.arctan2(edges_y,edges_x)/np.pi
     
    # initialize array to contain direction histograms for the 10 labels
    grad_histogram =np.zeros((10,n_bins),dtype='float')  
    
    # compute gradient histograms separately for all 10 labels
    for i in range(10):
        idx = np.nonzero(labels==i) # indices of images with label i
        grad_histogram[i,:]=(np.histogram(grad_angles[idx[0]], bins=n_bins,
                                          weights=length[idx[0]]))[0]
            
    return(grad_histogram)

