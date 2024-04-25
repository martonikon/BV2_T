#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 11:21:47 2024

@author: user
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
from sklearn.neighbors import NearestCentroid
from skimage.feature import hog
from sklearn.decomposition import PCA
from skimage.transform import resize
from skimage.feature import local_binary_pattern
from sklearn.datasets import fetch_openml
from sklearn.neighbors import NearestCentroid
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


def get_mnist_data(n_train=60000):
    # Fetch MNIST data from OpenML and split into training and testing data
    mnist = fetch_openml('mnist_784')
    data = np.array(mnist.data)
    target = np.array(mnist.target).astype('uint8')
    X_train, X_test, y_train, y_test = train_test_split(data, target, train_size=n_train)
    X_train = np.reshape(X_train, (X_train.shape[0], 28, 28))
    X_test = np.reshape(X_test, (X_test.shape[0], 28, 28))
    return (X_train, y_train), (X_test, y_test)

def dispersion(features, labels):
    def covariance_matrix(features):
        mean = np.mean(features, axis=0)
        centered_features = features - mean
        covariance_matrix = np.matmul(centered_features.T, centered_features) / len(features)
        return covariance_matrix

    # Within-Class Dispersion
    wcd = 0
    for label in np.unique(labels):
        class_features = features[labels == label]
        class_covariance = covariance_matrix(class_features)
        eigenvalues = np.linalg.eigvals(class_covariance)
        wcd += np.sum(np.abs(eigenvalues))

    # Between-Class Dispersion
    bcd_covariance = covariance_matrix(features)
    bcd_eigenvalues = np.linalg.eigvals(bcd_covariance)
    bcd = np.sum(np.abs(bcd_eigenvalues))

    dispersion_ratio = bcd / wcd

    return dispersion_ratio

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

def lbp_from_seq(images, n_directions=8, radius=2.0):
    lbp_features = []
    for i in range(len(images)):
#        ticks(i) # Anzeigen des Fortschritts
        lbp = local_binary_pattern(images[i,:,:], n_directions, radius)
        hist = np.histogram(lbp, bins=n_directions**2)[0]
        lbp_features.append(hist)
    return np.array(lbp_features)

def ticks(i, interval=1000):
    if int(i/interval) < int((i+1)/interval):
        if (int(i/(10*interval)) < int((i+1)/(10*interval))):
            print('|', end='\r', flush=True)
        else:
            print('+', end='\r', flush=True)
    return

def hog_from_seq(images, px=4, py=4, n_dir=9, m_ch=False):
    hog_features = []
    for i in range(len(images)):
        hog_feat = hog(images[i,:,:], orientations=n_dir, pixels_per_cell=(px,py),
                       cells_per_block=(2, 2), feature_vector=True, multichannel=m_ch)
        hog_features.append(hog_feat)
    return np.array(hog_features)

# Einlesen der MNIST-Daten
(X_train, y_train), (X_test, y_test) = get_mnist_data()

# Berechnung der verschiedenen Merkmale
M1 = X_train.reshape(len(X_train), -1)
M2 = get_image_histograms(X_train)
M3 = lbp_from_seq(X_train)
M4 = hog_from_seq(X_train)

# Berechnung der Dispersionswerte für jedes Merkmal
disp_M1 = dispersion(M1, y_train)
disp_M2 = dispersion(M2, y_train)
disp_M3 = dispersion(M3, y_train)
disp_M4 = dispersion(M4, y_train)

# Klassifikation mit dem Nearest-Centroid-Classifier und Auswertung der Genauigkeit
ncc = NearestCentroid()
ncc.fit(M1, y_train)
acc_M1_train = ncc.score(M1, y_train)
acc_M1_test = ncc.score(X_test.reshape(len(X_test), -1), y_test)

ncc.fit(M2, y_train)
acc_M2_train = ncc.score(M2, y_train)
acc_M2_test = ncc.score(get_image_histograms(X_test), y_test)


ncc.fit(M3, y_train)
acc_M3_train = ncc.score(M3, y_train)
acc_M3_test = ncc.score(lbp_from_seq(X_test), y_test)

ncc.fit(M4, y_train)
acc_M4_train = ncc.score(M4, y_train)
acc_M4_test = ncc.score(hog_from_seq(X_test), y_test)

print("Dispersionswerte:")
print("M1:", disp_M1)
print("M2:", disp_M2)
print("M3:", disp_M3)
print("M4:", disp_M4)

print("Accuracy Nearest-Centroid-Classifier:")
print("M1 - Train:", acc_M1_train, "Test:", acc_M1_test)
print("M2 - Train:", acc_M2_train, "Test:", acc_M2_test)
print("M3 - Train:", acc_M3_train, "Test:", acc_M3_test)
print("M4 - Train:", acc_M4_train, "Test:", acc_M4_test)