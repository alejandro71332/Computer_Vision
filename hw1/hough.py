import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as plp
import numpy as np
import random
import math
from numpy import unravel_index

def Convolution(img, kernel):
    height = img.shape[0]
    width = img.shape[1]
    kheight = kernel.shape[0] // 2
    kwidth = kernel.shape[1] // 2

    pad = ((kheight, kwidth), (kheight, kwidth))
    imgout = np.empty(img.shape, dtype=np.float64)
    img = np.pad(img, pad, mode='constant', constant_values=0)

    for i in np.arange(kheight, height + kheight):
        for j in np.arange(kwidth, width + kwidth):
            partition = img[i - kheight:i + kheight + 1, j - kwidth:j + kwidth + 1]
            imgout[i - kheight, j - kwidth] = (partition*kernel).sum()

    return imgout


def GuassianFilter(sigma):
    fsize = 2 * int(4 * sigma + 0.5) + 1
    gaussian = np.zeros((fsize, fsize), np.float64)
    m = fsize // 2
    n = fsize // 2
    
    for x in range(-m, m+1):
        for y in range(-n, n+1):
            x1 = 2*np.pi*(sigma**2)
            x2 = np.exp(-(x**2 + y**2)/(2* sigma**2))
            gaussian[x+m, y+n] = (1/x1)*x2

    return gaussian

def non_max_suppression(points):
    for y in range(len(points) - 2):
        for x in range(len(points[y]) - 2):
            sample = []
            
            for yy in range(3):
                for xx in range(3):
                    sample += [points[y+yy][x+xx]]

            max_idx = np.argmax(sample)

            for yy in range(3):
                for xx in range(3):
                    if yy*3 + xx != max_idx:
                        points[y+yy][x+xx] = 0

    return points

def accumulator(img, points):

    height = img.shape[0]
    width = img.shape[1]
    diag_len = int(round(math.sqrt(width * width + height * height)))
    thetas = np.deg2rad(np.arange(0.0, 180.0, 1))

    accumulator = np.zeros((len(thetas), diag_len), np.float64)

    for i in range(len(points[0])):
        y = points[0][i]
        x = points[1][i]

        for j in range(len(thetas)):
            rho = int(x * math.cos(thetas[j]) + y * math.sin(thetas[j]))
            accumulator[j][rho] += 1
    
    plt.imshow(accumulator)
    plt.savefig('accumulator.png')
    plt.show()



    return accumulator

def hough(img, accumulator):
    plt.imshow(img)

    height = img.shape[0]
    width = img.shape[1]

    max_idx = []
    copy = accumulator.copy()
    for i in range(4):
        idx = unravel_index(copy.argmax(), copy.shape)
        max_idx.append(idx)
        copy[max_idx[i]] = 0


    for theta, rho in max_idx:
        x = rho * np.cos(theta*math.pi/180)
        y = rho * np.sin(theta*math.pi/180)
        m = -x/y

        x1 = np.linspace(0, width, 2)
        b = -m * x + y
        y1 = m * x1 + b

        plt.plot(x1, y1, 'r')


    plt.axis([0, len(img[0]), 0, len(img)])
    plt.gca().invert_yaxis()
    plt.savefig('hough_transform.png')
    plt.show()



def hessian(img_orig, threshold):

    img = np.zeros((img_orig.shape[0], img_orig.shape[1]))

    for y in range(len(img_orig)):
        for x in range(len(img_orig[y])):
            img[y][x] = img_orig[y][x][0]

    gaussian = GuassianFilter(0.84)

    sobelx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    sobely = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    dx = Convolution(img, sobelx)
    dy = Convolution(img, sobely)

    dxx = Convolution(img, sobelx)
    dyy = Convolution(img, sobely)

    dxx2 = np.square(dxx)
    dyy2 = np.square(dyy)
    dxy = dx * dy

    Ixx2 = Convolution(dxx2, gaussian)
    Iyy2 = Convolution(dyy2, gaussian)
    Ixy = Convolution(dxy, gaussian)

    det_hessian = Ixx2*Iyy2 - np.square(Ixy)
   
    for x in range(len(det_hessian)):
        for y in range(len(det_hessian[x])):
            if det_hessian[x][y] <= threshold:
                det_hessian[x][y] = 0

    points = non_max_suppression(det_hessian)

    points = np.where(points != 0)

    return points, img_orig


if __name__ == '__main__':
    
    img = mpimg.imread('road.png')

    points, img = hessian(img, 0.04)

    accumulator = accumulator(img, points)
    
    hough(img, accumulator)
