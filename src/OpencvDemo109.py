#!/usr/bin/env python3
#coding = utf-8

import logging
import numpy as np
import cv2 as cv

# BLOB特征分析—SimpleBlobDetector使用
def OpencvDemo109():
    logging.basicConfig(level=logging.DEBUG)
    
    src = cv.imread("images/zhifang_ball.png")
    if src is None:
        logging.error("could not load an image!")
        return cv.Error.StsError
    
    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)

    # 初始化参数设置
    params = cv.SimpleBlobDetector_Params()

    params.minThreshold = 0
    params.maxThreshold = 256
    params.filterByArea = True
    params.minArea = 100
    params.filterByCircularity = True
    params.minCircularity = 0.1
    params.filterByConvexity = True
    params.minConvexity = 0.5
    params.filterByInertia = True
    params.minInertiaRatio = 0.5

    # 通过SimpleBlobDetector检测关键点
    blob = cv.SimpleBlobDetector_create(params)

    kps = blob.detect(gray)

    dst = cv.drawKeypoints(src, kps, None, (0, 0, 255), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    h, w, ch = src.shape
    result = np.zeros([h * 2, w, ch], dtype=src.dtype)
    result[0 : h, 0 : w, :] = src
    result[h : h * 2, 0 : w, :] = dst
    cv.putText(result, "original image", (10, 30), cv.FONT_ITALIC, .8, (0, 0, 255), 1)
    cv.putText(result, "image with results of blob keypoint detection", (10, h + 30), cv.FONT_ITALIC, .8, (0, 0, 255), 1)
    cv.imshow("blob keypoint detection", result)

    cv.waitKey(0)
    
    cv.destroyAllWindows()
    
    return cv.Error.StsOk

if __name__ == "__main__":
    OpencvDemo109()

# end of file
