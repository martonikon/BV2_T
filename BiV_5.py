#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 11:19:49 2024

@author: user
"""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
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

def resize_images(images):
    # Funktion zum Verkleinern der Bilder auf 14x14 und Umwandeln in 1D-Vektoren
    resized_images = np.zeros((images.shape[0], 14, 14))
    for i in range(images.shape[0]):
        resized_images[i] = np.resize(images[i], (14, 14))
    return resized_images.reshape(images.shape[0], -1)

def classify_ncc(f_train, f_test, y_train, y_test):
    # Funktion zur Klassifikation mit Nearest Centroid Classifier
    ncc = NearestCentroid()
    ncc.fit(f_train, y_train)
    pred_train = ncc.predict(f_train)
    pred_test = ncc.predict(f_test)
    return pred_train, pred_test



def display_samples(images, true_label, pred_label):
    # Funktion zur Anzeige von korrekt und falsch klassifizierten Beispielen
    correct_samples = [[] for _ in range(10)]
    incorrect_samples = [[] for _ in range(10)]
    
    for i in range(len(true_label)):
        true = true_label[i]
        pred = pred_label[i]
        if true == pred:
            if len(correct_samples[true]) < 10:
                correct_samples[true].append(images[i])
        else:
            if len(incorrect_samples[pred]) < 10:
                incorrect_samples[pred].append(images[i])
    
    plt.figure(figsize=(14, 14))
    plt.suptitle('Correctly Classified Samples')
    for i in range(10):
        for j in range(len(correct_samples[i])):
            plt.subplot(10, 10, i*10 + j + 1)
            plt.imshow(correct_samples[j][i].reshape(28, 28), cmap='gray')
            plt.axis('off')
    
    plt.figure(figsize=(14, 14))
    plt.suptitle('Incorrectly Classified Samples')
    for i in range(10):
        for j in range(len(incorrect_samples[i])):
            plt.subplot(10, 10, i*10 + j + 1)
            plt.imshow(incorrect_samples[j][i].reshape(28, 28), cmap='gray')
            plt.axis('off')
    
    plt.show()


# Main program
(X_train, y_train), (X_test, y_test) = get_mnist_data()

# Feature Scaling
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train.reshape(-1, 784))
X_test_scaled = scaler.transform(X_test.reshape(-1, 784))

scaled_images = np.array([resize(image, (14, 14)) for image in X_train])
scaled_features = scaled_images.reshape(len(X_train), -1)

# 2. Grauwerte durch PCA auf 196 Merkmale
pca = PCA(n_components=196)
pca.fit(X_train.reshape(len(X_train), -1))
pca_features = pca.transform(X_train.reshape(len(X_train), -1))

# Dimensionality Reduction
pca = PCA(n_components=0.95)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# Classification
pred_train, pred_test = classify_ncc(X_train_pca, X_test_pca, y_train, y_test)

accuracy_train = accuracy_score(y_train, pred_train)
accuracy_test = accuracy_score(y_test, pred_test)

print("Accuracy on training data:", accuracy_train)
print("Accuracy on test data:", accuracy_test)


# Display samples
display_samples(X_test, y_test, pred_test)

###########################################################################################################


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