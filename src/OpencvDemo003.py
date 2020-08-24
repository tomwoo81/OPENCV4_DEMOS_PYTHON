#!/usr/bin/env python3
#coding = utf-8

import logging
import numpy as np
import cv2 as cv

# 图像对象的创建与赋值
def OpencvDemo003():
    logging.basicConfig(level=logging.DEBUG)
    
    src = cv.imread("images/test.png")
    if src is None:
        logging.error("could not load an image!")
        return cv.Error.StsError
    
    cv.namedWindow("input", cv.WINDOW_AUTOSIZE)
    cv.imshow("input", src)
    
    # 克隆图像
    m1 = np.copy(src)
    cv.imshow("m1", m1)
    
    # 赋值
    m2 = src
    src[100:200,200:300,:] = 255
    cv.imshow("m2", m2)
    
    m3 = np.zeros(src.shape, src.dtype)
    cv.imshow("m3", m3)
    
    m4 = np.zeros([512,512], np.uint8)
    # m4[:,:] =127 try to give gray value 127
    cv.imshow("m4", m4)
    
    m5 = np.ones(shape=[512,512,3], dtype=np.uint8)
    m5[:,:,0] = 255
    cv.imshow("m5", m5)
    
    cv.waitKey(0)
    
    cv.destroyAllWindows()
    
    return cv.Error.StsOk

if __name__ == "__main__":
    OpencvDemo003()

# end of file
