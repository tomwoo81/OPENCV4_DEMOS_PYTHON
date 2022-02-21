#!/usr/bin/env python3
#coding = utf-8

import logging
import numpy as np
import cv2 as cv

# KMeans图像分割—背景替换
def OpencvDemo112():
    logging.basicConfig(level=logging.DEBUG)
    
    src = cv.imread("images/toux.jpg")
    if src is None:
        logging.error("could not load an image!")
        return cv.Error.StsError
    
    height, width, _ = src.shape
    
    # 将RGB数据转换为样本数据
    sample_data = src.reshape((-1, 3))
    data = np.float32(sample_data)

    # 使用KMeans进行图像分割
    num_clusters = 3
    term_crit = (cv.TERM_CRITERIA_COUNT | cv.TERM_CRITERIA_EPS, 10, 1.0)
    _, labels, centers = cv.kmeans(data, num_clusters, None, term_crit, num_clusters, cv.KMEANS_RANDOM_CENTERS)

    # 生成掩膜
    mask = np.zeros(src.shape[:2], dtype=np.uint8)
    index = labels[0][0]
    labels = np.reshape(labels, src.shape[:2])
    mask[labels == index] = 255

    # 对掩膜进行高斯模糊
    se = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    mask = cv.dilate(mask, se)
    mask = cv.GaussianBlur(mask, (5, 5), 0)

    # 融合前景与新背景
    dst = np.empty(src.shape, dtype=np.uint8)
    for row in range(height):
        for col in range(width):
            w1 = mask[row, col] / 255
            b, g, r = src[row, col]
            b = 255 * w1 + b * (1 - w1)
            g = 0 * w1 + g * (1 - w1)
            r = 255 * w1 + r * (1 - w1)
            dst[row, col] = (b, g, r)

    h, w, ch = src.shape
    result = np.zeros([h, w * 3, ch], dtype=src.dtype)
    result[0 : h, 0 : w, :] = src
    mask = cv.cvtColor(mask, cv.COLOR_GRAY2BGR) # 1 channel -> 3 channels
    result[0 : h, w : w * 2, :] = mask
    result[0 : h, w * 2 : w * 3, :] = dst
    cv.putText(result, "original image", (10, 30), cv.FONT_ITALIC, .6, (0, 255, 0), 1)
    cv.putText(result, "background mask", (w + 10, 30), cv.FONT_ITALIC, .6, (0, 255, 0), 1)
    cv.putText(result, "image after background replacement", (w * 2 + 10, 30), cv.FONT_ITALIC, .6, (0, 255, 0), 1)
    cv.imshow("K-means clustering - background replacement", result)

    cv.waitKey(0)
    
    cv.destroyAllWindows()
    
    return cv.Error.StsOk

if __name__ == "__main__":
    OpencvDemo112()

# end of file
