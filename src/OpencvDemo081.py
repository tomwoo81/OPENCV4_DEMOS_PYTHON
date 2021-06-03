#!/usr/bin/env python3
#coding = utf-8

import logging
import numpy as np
import cv2 as cv

# 角点检测—Harris角点检测
def OpencvDemo081():
    logging.basicConfig(level=logging.DEBUG)
    
    src = cv.imread("images/ele_panel.bmp")
    if src is None:
        logging.error("could not load an image!")
        return cv.Error.StsError
    
    dst = process(src)

    h, w, ch = src.shape
    result = np.zeros([h, w * 2, ch], dtype=src.dtype)
    result[0 : h, 0 : w, :] = src
    result[0 : h, w : w * 2, :] = dst
    cv.putText(result, "original image", (10, 30), cv.FONT_ITALIC, .6, (0, 0, 255), 1)
    cv.putText(result, "image with results of corner detection", (w + 10, 30), cv.FONT_ITALIC, .6, (0, 0, 255), 1)
    cv.imshow("Harris corner detection", result)

    cv.waitKey(0)
    
    cv.destroyAllWindows()
    
    return cv.Error.StsOk

def process(src):
    dst = np.copy(src)

    # detector parameters
    blockSize = 2
    apertureSize = 3
    k = 0.04

    # detecting corners
    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    rsp = cv.cornerHarris(gray, blockSize, apertureSize, k)

    # normalizing
    rsp_norm = np.empty(rsp.shape, dtype=rsp.dtype)
    cv.normalize(rsp, rsp_norm, alpha=0, beta=255, norm_type=cv.NORM_MINMAX)
    rsp_norm = cv.convertScaleAbs(rsp_norm)

    # drawing circles around corners
    for i in range(rsp_norm.shape[0]):
        for j in range(rsp_norm.shape[1]):
            if rsp_norm[i, j] > 80:
                b = np.random.randint(0, 256)
                g = np.random.randint(0, 256)
                r = np.random.randint(0, 256)
                cv.circle(dst, (j, i), 5, (int(b), int(g), int(r)), 2)
    
    return dst

if __name__ == "__main__":
    OpencvDemo081()

# end of file
