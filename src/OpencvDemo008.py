#!/usr/bin/env python3
#coding = utf-8

import logging
import numpy as np
import cv2 as cv

# 通道分离与合并
def OpencvDemo008():
    logging.basicConfig(level=logging.DEBUG)
    
    src = cv.imread("images/flower.png")
    if src is None:
        logging.error("could not load an image!")
        return cv.Error.StsError
    cv.namedWindow("input", cv.WINDOW_AUTOSIZE)
    cv.imshow("input", src)
    
    # 蓝色通道为零
    mv = cv.split(src)
    mv[0][:, :] = 0
    dst1 = cv.merge(mv)
    cv.imshow("output1", dst1)
    
    # 绿色通道为零
    mv = cv.split(src)
    mv[1][:, :] = 0
    dst2 = cv.merge(mv)
    cv.imshow("output2", dst2)
    
    # 红色通道为零
    mv = cv.split(src)
    mv[2][:, :] = 0
    dst3 = cv.merge(mv)
    cv.imshow("output3", dst3)
    
    dst = np.zeros(src.shape, dtype=np.uint8)
    logging.debug("src.shape: %s, dst.shape: %s", src.shape, dst.shape)
    cv.mixChannels([src], [dst], fromTo=[2, 0, 1, 1, 0, 2])
    cv.imshow("output4", dst)
    
    cv.waitKey(0)
    
    cv.destroyAllWindows()
    
    return cv.Error.StsOk

if __name__ == "__main__":
    OpencvDemo008()

# end of file
