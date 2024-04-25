import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
 
# Funktionen aus my_utilities.py
def get_mnist_data(n_train=60000):
    mnist = fetch_openml('mnist_784')
    data = np.array(mnist.data)
    target = np.array(mnist.target)
    X_train, X_test, y_train, y_test = train_test_split(data, target, train_size=n_train)
    # Reshape the feature vectors to 28x28 images and cast to uint8
    X_train = X_train.reshape(-1, 28, 28).astype(np.float32)
    X_test = X_test.reshape(-1, 28, 28).astype(np.float32)
    #trying to flatten the image
    X_train=X_train.reshape(len(X_train), -1)
    X_test= X_test.reshape(len(X_test), -1)
    return (X_train, y_train), (X_test, y_test)


# Funktionen für Gradientenrichtungshistogramme
def compute_gradients(image):
    # Beispiel für Gradientenberechnung (kann je nach Bedarf angepasst werden)
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    grad_x = np.convolve(image.flatten().astype(float), sobel_x.flatten(), mode='same')
    grad_y = np.convolve(image.flatten().astype(float), sobel_y.flatten(), mode='same')
    grad_x = grad_x.reshape(image.shape)
    grad_y = grad_y.reshape(image.shape)
    return grad_x, grad_y
 
def get_dir_histograms(images, labels, n_bins=18):
    grad_hist = np.zeros((10, n_bins))
 
    for i in range(len(labels)):
        edge_x, edge_y = compute_gradients(images[i])
        grad_len = np.sqrt(edge_x**2 + edge_y**2)
        grad_angles = np.arctan2(edge_y, edge_x) * (180 / np.pi)  
 
        hist, _ = np.histogram(grad_angles, bins=n_bins, weights=grad_len)
        label = int(labels[i])
        grad_hist[label] += hist
 
    return grad_hist
 
# Hauptprogramm
(X_train, y_train), _ = get_mnist_data()
 
grad_hist = get_dir_histograms(X_train, y_train)
 
for i in range(10):
    plt.bar(10*np.arange(18), grad_hist[i,:])
    plt.title('Orientation histogram for label ' + str(i))
    plt.xlabel('angle')
    plt.ylabel('frequency')
    plt.show()