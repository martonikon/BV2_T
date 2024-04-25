
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
 
def get_mnist_data(n_train=60000):
    mnist = fetch_openml('mnist_784')
    data = np.array(mnist.data)
    target = np.array(mnist.target)
    X_train, X_test, y_train, y_test = train_test_split(data, target, train_size=n_train)
    # Reshape the feature vectors to 28x28 images and cast to uint8
    X_train = X_train.reshape(-1, 28, 28).astype(np.uint8)
    X_test = X_test.reshape(-1, 28, 28).astype(np.uint8)
    return (X_train, y_train), (X_test, y_test)
 
def display_numbers(images, labels):
    picture = np.zeros((28*3, 28*10), dtype=np.uint8)
    for i in range(10):
        indices = np.nonzero(labels == str(i))[0]
        class_images = images[indices]
        mean_image = np.mean(class_images, axis=0)
        abs_diff = np.abs(class_images - mean_image)
        similar_index = np.argmin(np.sum(abs_diff, axis=(1, 2)))
        dissimilar_index = np.argmax(np.sum(abs_diff, axis=(1, 2)))
        picture[0:28, 28*i:28*(i+1)] = mean_image
        picture[28:56, 28*i:28*(i+1)] = class_images[similar_index]
        picture[56:84, 28*i:28*(i+1)] = class_images[dissimilar_index]
 
    plt.imshow(picture, cmap='gray')
    plt.axis('off')
    plt.show()
 
# Example usage
(X_train, y_train), _ = get_mnist_data()
display_numbers(X_train, y_train)