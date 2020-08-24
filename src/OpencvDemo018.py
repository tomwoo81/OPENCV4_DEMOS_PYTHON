#!/usr/bin/env python3
#coding = utf-8

import logging
import numpy as np
from matplotlib import pyplot as plt
import cv2 as cv

# 图像直方图均衡化
def OpencvDemo018():
    logging.basicConfig(level=logging.DEBUG)
    
    src = cv.imread("images/flower.png")
    if src is None:
        logging.error("could not load an image!")
        return cv.Error.StsError
    
    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    cv.namedWindow("input_gray", cv.WINDOW_AUTOSIZE)
    cv.imshow("input_gray", gray)
    dst = cv.equalizeHist(gray)
    cv.imshow("eq_gray", dst)
    
    custom_hist(gray)
    custom_hist(dst)
    
    cv.waitKey(0)
    
    cv.destroyAllWindows()
    
    return cv.Error.StsOk

def custom_hist(gray):
    h, w = gray.shape
    hist = np.zeros([256], dtype=np.int32)
    for row in range(h):
        for col in range(w):
            pv = gray[row, col]
            hist[pv] += 1
    
    y_pos = np.arange(0, 256, 1, dtype=np.int32)
    plt.bar(y_pos, hist, align='center', color='r', alpha=0.5)
    plt.xticks(y_pos, y_pos)
    plt.ylabel('Frequency')
    plt.title('Histogram')
    plt.show()

if __name__ == "__main__":
    OpencvDemo018()

# end of file
