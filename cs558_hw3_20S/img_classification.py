import matplotlib.image as mpimg
import numpy as np
import re
from sklearn.neighbors import KNeighborsClassifier
from os import listdir
from os.path import isfile, join


def histogram(img):
    colors = ['Red', 'Green', 'Blue']

    hist_r = np.histogram(img[:, :, 0].ravel(), bins = range(33))[0]
    hist_g = np.histogram(img[:, :, 1].ravel(), bins = range(33))[0]
    hist_b = np.histogram(img[:, :, 2].ravel(), bins = range(33))[0]

    hist = np.array([hist_r, hist_g, hist_b])

    hist = hist.ravel()

    print('Verifying iteration of img pixels is three times: ' + str(len(hist)))

    return hist


if __name__ == '__main__':

    training_files = [f for f in listdir('./ImClass/train/') if isfile(join('./ImClass/train/', f))]

    features = []
    labels = []

    for i in range(len(training_files)):
        img = mpimg.imread('./ImClass/train/' + training_files[i])

        hist = histogram(img)
        
        string = re.search(r'([\w.-]+)_([\w.-]+)', training_files[i])
        label = string.group(1)


        features.append(hist)
        labels.append(label)

    features = np.array(features)
    labels = np.array(labels)

    testing_files = [f for f in listdir('./ImClass/test/') if isfile(join('./ImClass/test/', f))]

    knn = KNeighborsClassifier(n_neighbors=1)

    knn.fit(features, labels)

    testing_features = []
    testing_labels = []

    for i in range(len(testing_files)):
        img = mpimg.imread('./ImClass/test/' + testing_files[i])

        hist = histogram(img)

        string = re.search(r'([\w.-]+)_([\w.-]+)', testing_files[i])
        label = string.group(1)


        testing_features.append(hist)
        testing_labels.append(label)

    testing_features = np.array(testing_features)

    predicted = knn.predict(testing_features)

    print()

    for i in range(len(predicted)):
        print('Test Image ' + testing_files[i] + ' of class ' + testing_labels[i] + ' has been assigned to ' + predicted[i])
    
    acc = knn.score(testing_features, testing_labels)

    print()

    print('Classifier Accuracy: ' + str(acc))

