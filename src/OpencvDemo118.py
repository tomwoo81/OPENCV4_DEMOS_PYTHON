#!/usr/bin/env python3
#coding = utf-8

import logging
import numpy as np
import cv2 as cv

# GrabCut图像分割
def OpencvDemo118():
    logging.basicConfig(level=logging.DEBUG)
    
    src = cv.imread("images/master.jpg")
    if src is None:
        logging.error("could not load an image!")
        return cv.Error.StsError
    
    rect = (180, 20, 180, 220)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    mask = cv.grabCut(src, None, rect, bgdModel, fgdModel, 5, cv.GC_INIT_WITH_RECT)[0]

    mask2 = np.where((mask == cv.GC_FGD) + (mask == cv.GC_PR_FGD), 255, 0).astype('uint8')

    dst = cv.bitwise_and(src, src, None, mask2)

    h, w, ch = src.shape
    result = np.zeros([h, w * 2, ch], dtype=src.dtype)
    result[0 : h, 0 : w, :] = src
    result[0 : h, w : w * 2, :] = dst
    cv.putText(result, "original image", (10, 30), cv.FONT_ITALIC, .8, (0, 0, 255), 1)
    cv.putText(result, "GrabCut segmented image", (w + 10, 30), cv.FONT_ITALIC, .8, (0, 0, 255), 1)
    cv.imshow("GrabCut image segmentation", result)

    cv.waitKey(0)
    
    cv.destroyAllWindows()
    
    return cv.Error.StsOk

if __name__ == "__main__":
    OpencvDemo118()

# end of file
