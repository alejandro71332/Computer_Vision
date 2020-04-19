import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import random
import math

def histogram(img):
    colors = ['Orange', 'Red', 'Green', 'Blue']

    hist = plt.hist(img.ravel(), bins = 256, color = colors[0], alpha = 0.5)
    for i in range(1, 4):
        hist = plt.hist(img[:, :, i - 1].ravel(), bins = 256, color = colors[i], alpha = 0.5)
    hist = plt.xlabel('Intensity Value')
    hist = plt.ylabel('Count')
    hist = plt.legend(['Total', 'Red_Channel', 'Green_Channel', 'Blue_Channel'])
    plt.show()

def points_in_clusters(ran_points):
    clusters = [[] for i in range(len(ran_points))]

    ran_points.sort()

    bins = list(range(256))

    for p in range(len(bins)):
        min = 256
        idx = 0
        for i in range(len(ran_points)):
            if min > abs(ran_points[i] - bins[p]):
                min = abs(ran_points[i] - bins[p])
                idx = i 
        clusters[idx] += [p]
    
    return clusters

def set_mean(clusters):
    points = [0] * len(clusters)
    for i in range(len(clusters)):
        amount = len(clusters[i])
        sums = sum(clusters[i])
        if len(clusters[i]):
            points[i] = int(sums / amount)
        else:
            points[i] = 0

    return points

def make_clusters(ran_points):
    cluster_old = []

    while not(cluster_old == ran_points):
        clusters = points_in_clusters(ran_points)
        cluster_means = set_mean(clusters)
        cluster_old = ran_points
        ran_points = cluster_means
    
    return clusters, cluster_means

def set_cluster_def(clusters, cluster_means):
    dict = {}
    for i in range(len(clusters)):
        for j in clusters[i]:
            dict[j] = cluster_means[i]
    return dict

def Kmeans(img, k):
    #histogram(img)
    img_y = img.shape[0]
    img_x = img.shape[1]

    r_points = [random.randint(0, 255) for i in range(k)]
    g_points = [random.randint(0, 255) for i in range(k)]
    b_points = [random.randint(0, 255) for i in range(k)]


    #red

    clusters_r, r_mean = make_clusters(r_points)
    dict_r = set_cluster_def(clusters_r, r_mean)

    #green

    clusters_g, g_mean = make_clusters(g_points)
    dict_g = set_cluster_def(clusters_g, g_mean)

    #blue

    clusters_b, b_mean = make_clusters(b_points)
    dict_b = set_cluster_def(clusters_b, b_mean)


    #apply to img

    for y in range(img_y):
        for x in range(img_x):
            r = int(img[y][x][0] * 255)
            g = int(img[y][x][1] * 255)
            b = int(img[y][x][2] * 255)
            img[y][x][0] = dict_r[r] / 255
            img[y][x][1] = dict_g[g] / 255
            img[y][x][2] = dict_b[b] / 255
    
    return img


        


if __name__ == '__main__':

    img = mpimg.imread('white-tower.png')

    img_out = Kmeans(img, 10)

    plt.imshow(img_out)
    plt.savefig('k-means_output.png')
    plt.show()
