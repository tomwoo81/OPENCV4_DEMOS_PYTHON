#!/usr/bin/env python3
#coding = utf-8

import logging
import numpy as np
import cv2 as cv

# HOG特征描述子—提取描述子
def OpencvDemo102():
    logging.basicConfig(level=logging.DEBUG)

    src = cv.imread("images/gaoyy_min.png")
    if src is None:
        logging.error("could not load an image!")
        return cv.Error.StsError

    cv.imshow("input", src)

    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)

    hog = cv.HOGDescriptor()

    features = hog.compute(gray, winStride=(8, 8), padding=(0, 0))

    logging.info("number of elements in the HOG descriptor: {:d}".format(len(features)))

    for i in range(len(features)):
        logging.debug("[{:d}]: {:f}".format(i, features[i][0]))

    cv.waitKey(0)
    
    cv.destroyAllWindows()
    
    return cv.Error.StsOk

if __name__ == "__main__":
    OpencvDemo102()

# end of file
