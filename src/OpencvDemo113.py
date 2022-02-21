#!/usr/bin/env python3
#coding = utf-8

import logging
import numpy as np
import cv2 as cv

# KMeans图像分割—主色彩提取
def OpencvDemo113():
    logging.basicConfig(level=logging.DEBUG)
    
    src = cv.imread("images/yuan_test.png")
    if src is None:
        logging.error("could not load an image!")
        return cv.Error.StsError
    
    height, width, _ = src.shape
    
    # 将RGB数据转换为样本数据
    sample_data = src.reshape((-1, 3))
    data = np.float32(sample_data)

    # 使用KMeans进行图像分割
    num_clusters = 4
    term_crit = (cv.TERM_CRITERIA_COUNT | cv.TERM_CRITERIA_EPS, 10, 1.0)
    _, labels, centers = cv.kmeans(data, num_clusters, None, term_crit, num_clusters, cv.KMEANS_RANDOM_CENTERS)

    # 计算各聚类的比例
    clusters = np.zeros((4), dtype=np.int32)
    for i in range(len(labels)):
        clusters[labels[i][0]] += 1
    clusters = np.float32(clusters) / (width * height)

    # 建立色卡
    card = np.zeros((50, width, 3), dtype=np.uint8)
    x_offset = 0
    for i in range(num_clusters):
        rect_x = x_offset
        rect_y = 0
        rect_width = round(clusters[i] * width)
        rect_height = 50
        rect = (rect_x, rect_y, rect_width, rect_height)

        b = int(centers[i][0])
        g = int(centers[i][1])
        r = int(centers[i][2])
        cv.rectangle(card, rect, (b, g, r), cv.FILLED, cv.LINE_8, 0)

        x_offset += rect_width

    h, w, ch = src.shape
    result = np.zeros([50 + h, w, ch], dtype=src.dtype)
    result[0 : 50, 0 : w, :] = card
    result[50 : 50 + h, 0 : w, :] = src
    cv.putText(result, "colour card", (10, 30), cv.FONT_ITALIC, .8, (0, 0, 255), 1)
    cv.putText(result, "original image", (10, 50 + 30), cv.FONT_ITALIC, .8, (0, 0, 255), 1)
    cv.imshow("K-means clustering - primary colour extraction", result)

    cv.waitKey(0)
    
    cv.destroyAllWindows()
    
    return cv.Error.StsOk

if __name__ == "__main__":
    OpencvDemo113()

# end of file
