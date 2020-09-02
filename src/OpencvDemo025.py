#!/usr/bin/env python3
#coding = utf-8

import logging
import numpy as np
import cv2 as cv

# 图像去噪声
def OpencvDemo025():
    logging.basicConfig(level=logging.DEBUG)
    
    src = cv.imread("images/example.png")
    if src is None:
        logging.error("could not load an image!")
        return cv.Error.StsError
    cv.namedWindow("input", cv.WINDOW_AUTOSIZE)
    cv.imshow("input", src)
    
    src = add_gaussian_noise(src)
    cv.imshow("gaussian noise", src)
    
    result1 = cv.blur(src, (5, 5))
    cv.imshow("blur", result1)
    
    result2 = cv.GaussianBlur(src, (5, 5), 0)
    cv.imshow("gaussian blur", result2)
    
    result3 = cv.medianBlur(src, 5)
    cv.imshow("median blur", result3)
    
    result4 = cv.fastNlMeansDenoisingColored(src, None, 15, 15, 10, 30)
    cv.imshow("non-local means denoising", result4)
    
    cv.waitKey(0)
    
    cv.destroyAllWindows()
    
    return cv.Error.StsOk

def add_gaussian_noise(image):
    noise = np.zeros(image.shape, image.dtype)
    
    m = (15, 15, 15)
    s = (30, 30, 30)
    cv.randn(noise, m, s)
    dst = cv.add(image, noise)
    
    cv.imshow("gaussian noise", dst)
    
    return dst

if __name__ == "__main__":
    OpencvDemo025()

# end of file
