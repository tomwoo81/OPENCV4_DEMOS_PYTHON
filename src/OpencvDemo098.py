#!/usr/bin/env python3
#coding = utf-8

import logging
import numpy as np
import cv2 as cv

# SIFT特征提取—关键点提取
def OpencvDemo098():
    logging.basicConfig(level=logging.DEBUG)
    
    src = cv.imread("images/flower.png")
    if src is None:
        logging.error("could not load an image!")
        return cv.Error.StsError
    
    sift = cv.SIFT().create()

    kps = sift.detect(src)

    dst = cv.drawKeypoints(src, kps, None, (-1), cv.DRAW_MATCHES_FLAGS_DEFAULT)

    h, w, ch = src.shape
    result = np.zeros([h, w * 2, ch], dtype=src.dtype)
    result[0 : h, 0 : w, :] = src
    result[0 : h, w : w * 2, :] = dst
    cv.putText(result, "original image", (10, 30), cv.FONT_ITALIC, .4, (0, 0, 255), 1)
    cv.putText(result, "image with results of SIFT keypoint detection", (w + 10, 30), cv.FONT_ITALIC, .4, (0, 0, 255), 1)
    cv.imshow("SIFT keypoint detection", result)

    cv.waitKey(0)
    
    cv.destroyAllWindows()
    
    return cv.Error.StsOk

if __name__ == "__main__":
    OpencvDemo098()

# end of file
