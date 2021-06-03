#!/usr/bin/env python3
#coding = utf-8

import logging
import numpy as np
import cv2 as cv

# 角点检测—Shi-Tomasi角点检测
def OpencvDemo082():
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
    cv.imshow("Shi-Tomasi corner detection", result)

    cv.waitKey(0)
    
    cv.destroyAllWindows()
    
    return cv.Error.StsOk

def process(src):
    dst = np.copy(src)

    # detector parameters
    maxCorners = 100
    qualityLevel = 0.05
    minDistance = 10

    # detecting corners
    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    corners = cv.goodFeaturesToTrack(gray, maxCorners, qualityLevel, minDistance)
    logging.info("number of corners: {:d}".format(len(corners)))

    # drawing circles around corners
    for c in corners:
        x, y = np.int32(c[0])
        b = np.random.randint(0, 256)
        g = np.random.randint(0, 256)
        r = np.random.randint(0, 256)
        cv.circle(dst, (x, y), 5, (int(b), int(g), int(r)), 2)
    
    return dst

if __name__ == "__main__":
    OpencvDemo082()

# end of file
