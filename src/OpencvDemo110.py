#!/usr/bin/env python3
#coding = utf-8

import logging
import numpy as np
from matplotlib import pyplot as plt
import cv2 as cv

# KMeans数据分类
def OpencvDemo110():
    logging.basicConfig(level=logging.INFO)

    # 生成随机数
    X = np.random.randint(25, 50, (25, 2))
    Y = np.random.randint(60, 85, (25, 2))
    pts = np.vstack((X, Y))
    pts = np.float32(pts)

    # 使用KMeans进行数据分类
    term_crit = (cv.TERM_CRITERIA_COUNT | cv.TERM_CRITERIA_EPS, 10, 1.0)
    _, labels, centers = cv.kmeans(pts, 2, None, term_crit, 2, cv.KMEANS_RANDOM_CENTERS)
    for i in range(len(centers)):
        logging.info("index: {:d}, center: ({:.3f}, {:.3f})".format(i, centers[i][0], centers[i][1]))

    # 获取不同标签的点
    labels = labels.ravel()
    A = pts[labels == 0]
    B = pts[labels == 1]

    # 用不同颜色显示分类
    plt.scatter(A[:, 0], A[:, 1], c = 'r')
    plt.scatter(B[:, 0], B[:, 1], c = 'b')

    # 为每个聚类的中心绘制方块
    plt.scatter(centers[:, 0], centers[:, 1], s = 80, c = 'y', marker = 's')

    plt.title("K-means clustering")
    plt.xlabel('X'); plt.ylabel('Y')
    plt.show()
    
    return cv.Error.StsOk

if __name__ == "__main__":
    OpencvDemo110()

# end of file
