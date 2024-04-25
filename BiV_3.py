#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 11:39:34 2024

@author: user
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

def get_mnist_data(n_train=60000):
    # Fetch MNIST data from OpenML and split into training and testing data
    mnist = fetch_openml('mnist_784')
    data = np.array(mnist.data)
    target = np.array(mnist.target).astype('uint8')
    X_train, X_test, y_train, y_test = train_test_split(data, target, train_size=n_train)
    X_train = np.reshape(X_train, (X_train.shape[0], 28, 28))
    X_test = np.reshape(X_test, (X_test.shape[0], 28, 28))
    return (X_train, y_train), (X_test, y_test)

def get_image_histograms(images, n_bins=18):
    edges_x = images - np.roll(images, 1, axis=1)
    edges_y = images - np.roll(images, 1, axis=2)
    # Edges will be weighted by their length in the histogram
    length = (edges_x**2 + edges_y**2)**0.5
    
    # Compute gradient angles for all pixels
    grad_angles = 180. * np.arctan2(edges_y, edges_x) / np.pi
    
    # Calculate gradient histograms for each image
    histograms = []
    
    for i in range(len(images)):
        histograms.append((np.histogram(grad_angles[i], bins=n_bins, weights=length[i]))[0])
    
    return np.array(histograms)

def dispersion(features, labels):
    def covariance_matrix(features):
        mean = np.mean(features, axis=0)
        centered_features = features - mean
        covariance_matrix = np.matmul(centered_features.T, features) / len(features)
        return covariance_matrix

    # Within-Class Dispersion
    wcd = 0
    for label in np.unique(labels):
        class_features = features[labels == label]
        class_covariance = covariance_matrix(class_features)
        eigenvalues = np.linalg.eig(class_covariance)[0]
        wcd += np.sum(np.abs(eigenvalues))

    # Between-Class Dispersion
    bcd_covariance = covariance_matrix(features)
    bcd_eigenvalues = np.linalg.eig(bcd_covariance)[0]
    bcd = np.sum(np.abs(bcd_eigenvalues))

    dispersion_ratio = bcd / wcd

    return dispersion_ratio

if __name__ == "__main__":
    # Load MNIST data
    (X_train, y_train), (_, _) = get_mnist_data()

    # Compute dispersion ratio for image data
    dispersion_image = dispersion(X_train.reshape(len(X_train), -1), y_train)

    # Compute dispersion ratio for image histograms
    image_histograms = get_image_histograms(X_train)
    dispersion_histograms = dispersion(image_histograms, y_train)

    print("Dispersion Ratio for Image Data:", dispersion_image)
    print("Dispersion Ratio for Image Histograms:", dispersion_histograms)
##########af#####aa#########asf ###