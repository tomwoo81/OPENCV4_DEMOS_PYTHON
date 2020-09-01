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
    
    result = add_salt_pepper_noise(src)
    cv.imshow("salt-pepper noise", result)
    result = gaussian_noise(src)
    cv.imshow("gaussian noise", result)
    
    cv.waitKey(0)
    
    cv.destroyAllWindows()
    
    return cv.Error.StsOk

def add_salt_pepper_noise(image):
    h, w = image.shape[:2]
    nums = 10000
    dst = np.copy(image)
    
    rows = np.random.randint(0, h, nums, dtype=np.int)
    cols = np.random.randint(0, w, nums, dtype=np.int)
    for i in range(nums):
        if i % 2 == 1:
            dst[rows[i], cols[i]] = (255, 255, 255)
        else:
            dst[rows[i], cols[i]] = (0, 0, 0)
    
    result = np.zeros([h, w * 2, 3], dtype=image.dtype)
    result[0 : h, 0 : w, :] = image
    result[0 : h, w : w * 2, :] = dst
    cv.putText(result, "original image", (10, 30), cv.FONT_HERSHEY_PLAIN, 2.0, (0, 255, 255), 1)
    cv.putText(result, "salt-pepper noise image", (w + 10, 30), cv.FONT_HERSHEY_PLAIN, 2.0, (0, 255, 255), 1)
    
    return result

def gaussian_noise(image):
    noise = np.zeros(image.shape, image.dtype)
    
    m = (15, 15, 15)
    s = (30, 30, 30)
    cv.randn(noise, m, s)
    dst = cv.add(image, noise)
    
    h, w = image.shape[:2]
    result = np.zeros([h, w * 2, 3], dtype=image.dtype)
    result[0 : h, 0 : w, :] = image
    result[0 : h, w : w * 2, :] = dst
    cv.putText(result, "original image", (10, 30), cv.FONT_HERSHEY_PLAIN, 2.0, (0, 255, 255), 1)
    cv.putText(result, "gaussian noise image", (w + 10, 30), cv.FONT_HERSHEY_PLAIN, 2.0, (0, 255, 255), 1)
    
    return result

if __name__ == "__main__":
    OpencvDemo024()

# end of file
