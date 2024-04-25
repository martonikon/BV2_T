import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from skimage.feature import hog
import matplotlib.pyplot as plt

def get_cifar10_data(n_train=5000):
    print('Fetching CIFAR-10 data (be patient, this takes a minute!)')
    cifar10 = fetch_openml('cifar_10')
    data = np.array(cifar10.data)
    target = np.array(cifar10.target).astype('uint8')

    X_train, X_test, y_train, y_test = train_test_split(data, target, train_size=n_train)
    
    X_train = (np.reshape(X_train,(X_train.shape[0],3,32,32)))
    X_train = X_train.transpose(0,2,3,1)
    X_test  = (np.reshape(X_test,(X_test.shape[0],3,32,32)))
    X_test  = X_test.transpose(0,2,3,1)
    
    return (X_train, y_train), (X_test, y_test)

def extract_features(X_train, X_test, feature_type='gray'):
    if feature_type == 'gray':
        X_train_gray = np.mean(X_train, axis=3)
        X_test_gray = np.mean(X_test, axis=3)
        return X_train_gray.reshape(len(X_train), -1), X_test_gray.reshape(len(X_test), -1)
    elif feature_type == 'hog':
        X_train_hog = np.array([hog(image, block_norm='L2-Hys') for image in X_train])
        X_test_hog = np.array([hog(image, block_norm='L2-Hys') for image in X_test])
        return X_train_hog, X_test_hog

def train_and_evaluate(X_train, X_test, y_train, y_test, model_type='svm', feature_type='gray'):
    if feature_type == 'gray':
        pca = PCA(n_components=196)
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca = pca.transform(X_test)
    elif feature_type == 'hog':
        X_train_pca, X_test_pca = X_train, X_test
    
    if model_type == 'svm':
        clf = SVC(kernel='rbf', C=3)
    elif model_type == 'knn':
        clf = KNeighborsClassifier(n_neighbors=5)

    clf.fit(X_train_pca, y_train)
    y_train_pred = clf.predict(X_train_pca)
    y_test_pred = clf.predict(X_test_pca)

    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)

    train_cm = confusion_matrix(y_train, y_train_pred)
    test_cm = confusion_matrix(y_test, y_test_pred)

    return train_accuracy, test_accuracy, train_cm, test_cm

# Laden der Daten
(X_train, y_train), (X_test, y_test) = get_cifar10_data()

# Extrahieren der Merkmale (hier: HOG)
X_train_features, X_test_features = extract_features(X_train, X_test, feature_type='hog')

# Trainieren und evaluieren von SVM und kNN
svm_train_accuracy, svm_test_accuracy, svm_train_cm, svm_test_cm = train_and_evaluate(X_train_features, X_test_features, y_train, y_test, model_type='svm')
knn_train_accuracy, knn_test_accuracy, knn_train_cm, knn_test_cm = train_and_evaluate(X_train_features, X_test_features, y_train, y_test, model_type='knn')

print("SVM Training Accuracy:", svm_train_accuracy)
print("SVM Test Accuracy:", svm_test_accuracy)

print("kNN Training Accuracy:", knn_train_accuracy)
print("kNN Test Accuracy:", knn_test_accuracy)

# Confusion Matrix für SVM (Trainingsdaten)
svm_train_cm_display = ConfusionMatrixDisplay(svm_train_cm)
svm_train_cm_display.plot()
plt.title('SVM Train Confusion Matrix')
plt.show()

# Confusion Matrix für SVM (Testdaten)
svm_test_cm_display = ConfusionMatrixDisplay(svm_test_cm)
svm_test_cm_display.plot()
plt.title('SVM Test Confusion Matrix')
plt.show()

# Confusion Matrix für kNN (Trainingsdaten)
knn_train_cm_display = ConfusionMatrixDisplay(knn_train_cm)
knn_train_cm_display.plot()
plt.title('kNN Train Confusion Matrix')
plt.show()

# Confusion Matrix für kNN (Testdaten)
knn_test_cm_display = ConfusionMatrixDisplay(knn_test_cm)
knn_test_cm_display.plot()
plt.title('kNN Test Confusion Matrix')
plt.show()