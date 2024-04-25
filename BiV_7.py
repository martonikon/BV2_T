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
from sklearn.neighbors import KNeighborsClassifier


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


def classify_ncc(f_train, f_test, y_train, y_test):
    # Funktion zur Klassifikation mit Nearest Centroid Classifier
    ncc = NearestCentroid()
    ncc.fit(f_train, y_train)
    pred_train = ncc.predict(f_train)
    pred_test = ncc.predict(f_test)
    return pred_train, pred_test

def classify_knn(f_train, y_train, f_test, y_test, n_nbs=5):
    # Funktion zur Klassifikation mit kNN Classifier
    knn = KNeighborsClassifier(n_neighbors=n_nbs)
    knn.fit(f_train, y_train)
    pred_train = knn.predict(f_train)
    pred_test = knn.predict(f_test)
    return pred_train, pred_test


# Laden der MNIST-Daten
(X_train, y_train), (X_test, y_test) = get_mnist_data(n_train=60000)

# Originalgrauwerte, reduziert durch PCA auf 196 Merkmale
pca = PCA(n_components=196)
pca.fit(X_train.reshape((X_train.shape[0], -1)))
X_train_pca = pca.transform(X_train.reshape((X_train.shape[0], -1)))
X_test_pca = pca.transform(X_test.reshape((X_test.shape[0], -1)))

# LBP-Merkmale
X_train_lbp = lbp_from_seq(X_train)
X_test_lbp = lbp_from_seq(X_test)

# HOG-Merkmale
X_train_hog = hog_from_seq(X_train)
X_test_hog = hog_from_seq(X_test)

# HOG-Merkmale, reduziert durch PCA auf 196 Merkmale
pca_hog = PCA(n_components=196)
pca_hog.fit(X_train_hog)
X_train_hog_pca = pca_hog.transform(X_train_hog)
X_test_hog_pca = pca_hog.transform(X_test_hog)

# Klassifikation mit Nearest Centroid Classifier
pred_train_ncc1, pred_test_ncc1 = classify_ncc(X_train_pca, X_test_pca, y_train, y_test)
pred_train_ncc2, pred_test_ncc2 = classify_ncc(X_train_lbp, X_test_lbp, y_train, y_test)
pred_train_ncc3, pred_test_ncc3 = classify_ncc(X_train_hog, X_test_hog, y_train, y_test)
pred_train_ncc4, pred_test_ncc4 = classify_ncc(X_train_hog_pca, X_test_hog_pca, y_train, y_test)

# Klassifikation mit k-Nearest-Neighbors-Klassifikator
n_nbs = 5
pred_train_knn1, pred_test_knn1 = classify_knn(X_train_pca, y_train, X_test_pca, y_test, n_nbs)
pred_train_knn2, pred_test_knn2 = classify_knn(X_train_lbp, y_train, X_test_lbp, y_test, n_nbs)
pred_train_knn3, pred_test_knn3 = classify_knn(X_train_hog, y_train, X_test_hog, y_test, n_nbs)
pred_train_knn4, pred_test_knn4 = classify_knn(X_train_hog_pca, y_train, X_test_hog_pca, y_test, n_nbs)

# Berechnung der Accuracies
accuracy_ncc1 = accuracy_score(y_test, pred_test_ncc1)
accuracy_ncc2 = accuracy_score(y_test, pred_test_ncc2)
accuracy_ncc3 = accuracy_score(y_test, pred_test_ncc3)
accuracy_ncc4 = accuracy_score(y_test, pred_test_ncc4)

accuracy_knn1 = accuracy_score(y_test, pred_test_knn1)
accuracy_knn2 = accuracy_score(y_test, pred_test_knn2)
accuracy_knn3 = accuracy_score(y_test, pred_test_knn3)
accuracy_knn4 = accuracy_score(y_test, pred_test_knn4)

# Ausgabe der Trainingsdaten
print("Trainingsdaten für M1:")
print(X_train_pca)
print("Trainingsdaten für M2:")
print(X_train_lbp)
print("Trainingsdaten für M3:")
print(X_train_hog)
print("Trainingsdaten für M4:")
print(X_train_hog_pca)

print("Accuracies für NCC und kNN:")
print("M1: NCC - {}, kNN - {}".format(accuracy_ncc1, accuracy_knn1))
print("M2: NCC - {}, kNN - {}".format(accuracy_ncc2, accuracy_knn2))
print("M3: NCC - {}, kNN - {}".format(accuracy_ncc3, accuracy_knn3))
print("M4: NCC - {}, kNN - {}".format(accuracy_ncc4, accuracy_knn4))
