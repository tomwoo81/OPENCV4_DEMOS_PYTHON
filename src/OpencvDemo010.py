#!/usr/bin/env python3
#coding = utf-8

import logging
import numpy as np
import cv2 as cv

# 图像像素值统计
def OpencvDemo010():
    logging.basicConfig(level=logging.DEBUG)
    
    src = cv.imread("images/test.png", cv.IMREAD_GRAYSCALE)
    if src is None:
        logging.error("could not load an image!")
        return cv.Error.StsError
    cv.namedWindow("input", cv.WINDOW_AUTOSIZE)
    cv.imshow("input", src)
    
    minVal, maxVal, minLoc, maxLoc = cv.minMaxLoc(src)
    logging.info("min: {:.2f}, max: {:.2f}".format(minVal, maxVal))
    logging.info("min loc: {}, max loc: {}".format(minLoc, maxLoc))
    
    means, stddevs = cv.meanStdDev(src)
    logging.info("mean: {}, stddev: {}".format(means, stddevs))
    src[np.where(src <= means)] = 0
    src[np.where(src > means)] = 255
    cv.imshow("binary", src)
    
    cv.waitKey(0)
    
    cv.destroyAllWindows()
    
    return cv.Error.StsOk

if __name__ == "__main__":
    OpencvDemo010()

# end of file
