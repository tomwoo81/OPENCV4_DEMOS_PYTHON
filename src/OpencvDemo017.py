#!/usr/bin/env python3
#coding = utf-8

import logging
import numpy as np
from matplotlib import pyplot as plt
import cv2 as cv

# 图像直方图
def OpencvDemo017():
    logging.basicConfig(level=logging.DEBUG)
    
    src = cv.imread("images/flower.png")
    if src is None:
        logging.error("could not load an image!")
        return cv.Error.StsError
    
    cv.namedWindow("input", cv.WINDOW_AUTOSIZE)
    cv.imshow("input", src)
    image_hist(src)
    
    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    cv.imshow("input", gray)
    custom_hist(gray)
    
    cv.waitKey(0)
    
    cv.destroyAllWindows()
    
    return cv.Error.StsOk

def image_hist(image):
    color = ('blue', 'green', 'red')
    for i, color in enumerate(color):
        hist = cv.calcHist([image], [i], None, [256], [0, 256])
        plt.plot(hist, color=color)
        plt.xlim([0, 255])
    plt.show()

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
    plt.plot(hist, color='r')
    plt.xlim([0, 255])
    plt.show()

if __name__ == "__main__":
    OpencvDemo017()

# end of file
