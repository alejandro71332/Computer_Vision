import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from os import listdir
from os.path import isfile, join
from PIL import ImageChops, Image
from sklearn.cluster import KMeans


if __name__ == '__main__':


    sky_train = Image.open('./sky/train/sky_train.jpg')
    sky_gimp = Image.open('./sky/train/sky_train_gimp.jpg')

    sky_points = []
    non_sky_points = []

    diff = ImageChops.difference(sky_train, sky_gimp)

    diff = np.array(diff, dtype = np.uint8)
    sky_train = np.array(sky_train, dtype = np.uint8)

    img_y = sky_train.shape[0]
    img_x = sky_train.shape[1]

    for i in range(img_y):
        for j in range(img_x):
            total_diff = int(diff[i][j][0]) + int(diff[i][j][1]) + int(diff[i][j][2])
            if total_diff != 0:
                r = sky_train[i][j][0]
                g = sky_train[i][j][1]
                b = sky_train[i][j][2]

                sky_points.append([r, g, b])
                
            else:
                r = sky_train[i][j][0]
                g = sky_train[i][j][1]
                b = sky_train[i][j][2]

                non_sky_points.append([r, g, b])

    sky_points = np.array(sky_points)
    non_sky_points = np.array(non_sky_points)

    kmeans_sky = KMeans(n_clusters=10)
    kmeans_sky.fit(sky_points)
    sky_centroids = kmeans_sky.cluster_centers_
    sky_centroids = sky_centroids.astype(int)

    kmeans_non_sky = KMeans(n_clusters=10)
    kmeans_non_sky.fit(non_sky_points)
    non_sky_centroids = kmeans_non_sky.cluster_centers_
    non_sky_centroids = non_sky_centroids.astype(int)


    features = []
    labels = []

    sky_len = len(sky_centroids)
    non_sky_len = len(non_sky_centroids)

    for i in range(sky_len + non_sky_len):
        if i < sky_len:
            label = 'sky'
            features.append(sky_centroids[i])
        else:
            label = 'non-sky'
            features.append(non_sky_centroids[i - sky_len])
        
        labels.append(label)

    features = np.array(features)
    labels = np.array(labels)    

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(features, labels)

    testing_files = [f for f in listdir('./sky/test/') if isfile(join('./sky/test/', f))]

    for i in range(len(testing_files)):
        test_img = mpimg.imread('./sky/test/' + testing_files[i])

        testing_features = []

        test_img_y = test_img.shape[0]
        test_img_x = test_img.shape[1]

        for y in range(test_img_y):
            for x in range(test_img_x):
                r = test_img[y][x][0]
                g = test_img[y][x][1]
                b = test_img[y][x][2]

                testing_features.append([r, g, b])

        testing_features = np.array(testing_features)

        predicted = knn.predict(testing_features)

        copy_img = test_img.copy()

        for y in range(test_img_y):
            for x in range(test_img_x):
                if predicted[test_img_x*y + x] == 'sky':
                    copy_img[y][x][0] = 255
                    copy_img[y][x][1] = 0
                    copy_img[y][x][2] = 0

        plt.imshow(copy_img)
        img_name = 'pixel_classification_' + str(i + 1) + '.jpg'
        plt.savefig(img_name)
        plt.show()
