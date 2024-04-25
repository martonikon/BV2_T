#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 13:01:45 2024

@author: user
"""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from skimage.transform import resize
from skimage.feature import local_binary_pattern
from sklearn.datasets import fetch_openml

def get_mnist_data(n_train=60000):
    # Fetch MNIST data from OpenML and split into training and testing data
    mnist = fetch_openml('mnist_784')
    data = np.array(mnist.data)
    target = np.array(mnist.target).astype('uint8')
    X_train, X_test, y_train, y_test = train_test_split(data, target, train_size=n_train)
    X_train = np.reshape(X_train, (X_train.shape[0], 28, 28))
    X_test = np.reshape(X_test, (X_test.shape[0], 28, 28))
    return (X_train, y_train), (X_test, y_test)

def lbp_from_seq(images, n_directions=8, radius=2.0):
    lbp_features = []
    for i in range(len(images)):
        ticks(i)  # Anzeigen des Fortschritts
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

# Einlesen der MNIST-Daten
(X_train, y_train), (X_test, y_test) = get_mnist_data()

# Erzeugen verschiedener Merkmalsvarianten
# 1. Grauwerte durch Skalierung auf 14x14
scaled_images = np.array([resize(image, (14, 14)) for image in X_train])
scaled_features = scaled_images.reshape(len(X_train), -1)

# 2. Grauwerte durch PCA auf 196 Merkmale
pca = PCA(n_components=196)
pca.fit(X_train.reshape(len(X_train), -1))
pca_features = pca.transform(X_train.reshape(len(X_train), -1))

# 3. LBP-Merkmale durch PCA auf 196 Merkmale
lbp_features = lbp_from_seq(X_train)
pca_lbp = PCA(n_components=min(X_train.shape[0], X_train.shape[1]))
pca_lbp.fit(lbp_features)
pca_lbp_features = pca_lbp.transform(lbp_features)


# Berechnung der Dispersion Ratio für jedes Merkmal
disp_ratio_scaled = dispersion(scaled_features, y_train)
disp_ratio_pca = dispersion(pca_features, y_train)
disp_ratio_pca_lbp = dispersion(pca_lbp_features, y_train)


print("Dispersion Ratio für Grauwerte nach Skalierung: ", disp_ratio_scaled)
print("Dispersion Ratio für Grauwerte nach PCA: ", disp_ratio_pca)
print("Dispersion Ratio für LBP-Merkmale nach PCA: ", disp_ratio_pca_lbp)

##########################################################################################
