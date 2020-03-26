import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as plp
import numpy as np
import random
import math

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

def ransac(img, points, inlier_threshold, dis_threshold):
    li = []
    plt.imshow(img)
    arbitrary_coordinates_system_already_previous_examined_fully = []
    for k in range(100):
        ran_range = len(points[0]) - 1
        plt.axis([0, len(img[0]), 0, len(img)])
        plt.gca().invert_yaxis()
        x = []
        y = []

        ran_idx = random.randrange(ran_range)
        (t1,t2) = (points[0][ran_idx],points[1][ran_idx])
        ran_idx = random.randrange(ran_range)
        (t3,t4) = (points[0][ran_idx],points[1][ran_idx])

        if ((t1,t2),(t3,t4)) in arbitrary_coordinates_system_already_previous_examined_fully:
            print("Collision:", t1, t2, t3, t4)
            continue

        arbitrary_coordinates_system_already_previous_examined_fully.append(((t1,t2),(t3,t4)))

        x.append(t1)
        x.append(t3)
        y.append(t2)
        y.append(t4)

        x = np.array(x)
        y = np.array(y)

    

        m, b = np.polyfit(x, y, 1)


        inlier_c = 0

        for j in range(len(points[0])):
            y1 = points[0][j]
            x1 = points[1][j]

            if m == 0:
                continue

            b1 = y1 + x1 / m
            xp = (b1 - b) / (m + 1 / m)
            yp = m * xp + b
            d = math.sqrt((y1 - yp)**2 + (x1 - xp)**2)
   
            if d < dis_threshold:
                inlier_c += 1
                y = np.append(y, yp)
                x = np.append(x, xp)

        if inlier_c >= inlier_threshold:
            m, b = np.polyfit(x, y, 1)
            li.append((inlier_c, (m,b), (x,y)))

    li = sorted(li, key=lambda x: x[0], reverse=True)
    
    for i in range(4 if len(li) >= 4 else len(li)):
        x = li[i][2][0]
        m = li[i][1][0]
        b = li[i][1][1]
        y = li[i][2][1]
        for j in range(len(x)):
            plt.gca().add_patch(plp.Rectangle((x[j]-1, y[j]+1), 3, 3, linewidth=1))
        plt.plot(x, x*m+b)

    plt.savefig('ransac.png')
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

    ransac(img, points, 7, 10)