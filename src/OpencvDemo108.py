#!/usr/bin/env python3
#coding = utf-8

import logging
import numpy as np
import cv2 as cv

# 特征提取之关键点检测—GFTTDetector
def OpencvDemo108():
    logging.basicConfig(level=logging.DEBUG)
    
    src = cv.imread("images/test1.png")
    if src is None:
        logging.error("could not load an image!")
        return cv.Error.StsError
    
    # 通过GFTTDetector检测关键点
    gftt = cv.GFTTDetector_create(1000, 0.01, 1.0, 3, False, 0.04)

    kps = gftt.detect(src)

    dst = cv.drawKeypoints(src, kps, None, (-1), cv.DRAW_MATCHES_FLAGS_DEFAULT)

    h, w, ch = src.shape
    result = np.zeros([h, w * 2, ch], dtype=src.dtype)
    result[0 : h, 0 : w, :] = src
    result[0 : h, w : w * 2, :] = dst
    cv.putText(result, "original image", (10, 30), cv.FONT_ITALIC, .6, (0, 0, 255), 1)
    cv.putText(result, "image with results of GFTT keypoint detection", (w + 10, 30), cv.FONT_ITALIC, .6, (0, 0, 255), 1)
    cv.imshow("GFTT keypoint detection", result)

    cv.waitKey(0)
    
    cv.destroyAllWindows()
    
    return cv.Error.StsOk

if __name__ == "__main__":
    OpencvDemo108()

# end of file
