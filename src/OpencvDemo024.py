#!/usr/bin/env python3
#coding = utf-8

import logging
import numpy as np
import cv2 as cv

# 图像噪声
def OpencvDemo024():
    logging.basicConfig(level=logging.DEBUG)
    
    src = cv.imread("images/cos.jpg")
    if src is None:
        logging.error("could not load an image!")
        return cv.Error.StsError
    
    h, w, ch = src.shape
    result = np.zeros([h, w * 2, ch], dtype=src.dtype)
    
    dst = add_salt_pepper_noise(src)
    
    result[0 : h, 0 : w, :] = src
    result[0 : h, w : w * 2, :] = dst
    cv.putText(result, "original image", (10, 30), cv.FONT_HERSHEY_PLAIN, 2.0, (0, 255, 255), 1)
    cv.putText(result, "salt-pepper noise image", (w + 10, 30), cv.FONT_HERSHEY_PLAIN, 2.0, (0, 255, 255), 1)
    cv.imshow("salt-pepper noise", result)
    
    dst = add_gaussian_noise(src)
    
    result[0 : h, 0 : w, :] = src
    result[0 : h, w : w * 2, :] = dst
    cv.putText(result, "original image", (10, 30), cv.FONT_HERSHEY_PLAIN, 2.0, (0, 255, 255), 1)
    cv.putText(result, "gaussian noise image", (w + 10, 30), cv.FONT_HERSHEY_PLAIN, 2.0, (0, 255, 255), 1)
    cv.imshow("gaussian noise", result)
    
    cv.waitKey(0)
    
    cv.destroyAllWindows()
    
    return cv.Error.StsOk

def add_salt_pepper_noise(src):
    h, w = src.shape[:2]
    nums = 10000
    dst = np.copy(src)
    
    rows = np.random.randint(0, h, nums, dtype=np.int)
    cols = np.random.randint(0, w, nums, dtype=np.int)
    for i in range(nums):
        if i % 2 == 1:
            dst[rows[i], cols[i]] = (255, 255, 255)
        else:
            dst[rows[i], cols[i]] = (0, 0, 0)
    
    return dst

def add_gaussian_noise(src):
    noise = np.zeros(src.shape, src.dtype)
    
    m = (15, 15, 15)
    s = (30, 30, 30)
    cv.randn(noise, m, s)
    dst = cv.add(src, noise)
    
    return dst

if __name__ == "__main__":
    OpencvDemo024()

# end of file
