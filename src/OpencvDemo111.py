#!/usr/bin/env python3
#coding = utf-8

import logging
import numpy as np
import cv2 as cv

# KMeans图像分割
def OpencvDemo111():
    logging.basicConfig(level=logging.DEBUG)
    
    src = cv.imread("images/toux.jpg")
    if src is None:
        logging.error("could not load an image!")
        return cv.Error.StsError
    
    # 将RGB数据转换为样本数据
    sample_data = src.reshape((-1, 3))
    data = np.float32(sample_data)

    # 使用KMeans进行图像分割
    num_clusters = 3
    term_crit = (cv.TERM_CRITERIA_COUNT | cv.TERM_CRITERIA_EPS, 10, 1.0)
    _, labels, centers = cv.kmeans(data, num_clusters, None, term_crit, num_clusters, cv.KMEANS_RANDOM_CENTERS)

    # 显示图像分割结果
    centers = np.uint8(centers)
    dst = centers[labels.flatten()]
    dst = dst.reshape((src.shape))

    h, w, ch = src.shape
    result = np.zeros([h, w * 2, ch], dtype=src.dtype)
    result[0 : h, 0 : w, :] = src
    result[0 : h, w : w * 2, :] = dst
    cv.putText(result, "original image", (10, 30), cv.FONT_ITALIC, .5, (0, 0, 255), 1)
    cv.putText(result, "segmented image after K-means clustering", (w + 10, 30), cv.FONT_ITALIC, .5, (0, 0, 255), 1)
    cv.imshow("K-means clustering - image segmentation", result)

    cv.waitKey(0)
    
    cv.destroyAllWindows()
    
    return cv.Error.StsOk

if __name__ == "__main__":
    OpencvDemo111()

# end of file
