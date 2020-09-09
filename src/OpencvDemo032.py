#!/usr/bin/env python3
#coding = utf-8

import logging
import numpy as np
import cv2 as cv

# 图像梯度–更多梯度算子
def OpencvDemo032():
    logging.basicConfig(level=logging.DEBUG)
    
    src = cv.imread("images/test.png")
    if src is None:
        logging.error("could not load an image!")
        return cv.Error.StsError
    cv.namedWindow("input", cv.WINDOW_AUTOSIZE)
    cv.imshow("input", src)
    
    h, w, ch = src.shape
    result = np.zeros([h, w * 2, ch], dtype=src.dtype)
    
    robert_x = np.array([[ 1,  0], 
                         [ 0, -1]], dtype=np.float32)
    robert_y = np.array([[ 0, -1], 
                         [ 1,  0]], dtype=np.float32)
    
    robert_grad_x = cv.filter2D(src, cv.CV_16S, robert_x)
    robert_grad_y = cv.filter2D(src, cv.CV_16S, robert_y)
    robert_grad_x = cv.convertScaleAbs(robert_grad_x)
    robert_grad_y = cv.convertScaleAbs(robert_grad_y)
    
    result[0 : h, 0 : w, :] = robert_grad_x
    result[0 : h, w : w * 2, :] = robert_grad_y
    cv.putText(result, "robert operator (x)", (10, 30), cv.FONT_HERSHEY_PLAIN, 2.0, (0, 255, 255), 1)
    cv.putText(result, "robert operator (y)", (w + 10, 30), cv.FONT_HERSHEY_PLAIN, 2.0, (0, 255, 255), 1)
    cv.imshow("gradient - robert operator", result)
    
    prewitt_x = np.array([[-1,  0,  1], 
                          [-1,  0,  1], 
                          [-1,  0,  1]], dtype=np.float32)
    prewitt_y = np.array([[-1, -1, -1], 
                          [ 0,  0,  0], 
                          [ 1,  1,  1]], dtype=np.float32)
    
    prewitt_grad_x = cv.filter2D(src, cv.CV_32F, prewitt_x)
    prewitt_grad_y = cv.filter2D(src, cv.CV_32F, prewitt_y)
    prewitt_grad_x = cv.convertScaleAbs(prewitt_grad_x)
    prewitt_grad_y = cv.convertScaleAbs(prewitt_grad_y)
    
    result[0 : h, 0 : w, :] = prewitt_grad_x
    result[0 : h, w : w * 2, :] = prewitt_grad_y
    cv.putText(result, "prewitt operator (x)", (10, 30), cv.FONT_HERSHEY_PLAIN, 2.0, (0, 255, 255), 1)
    cv.putText(result, "prewitt operator (y)", (w + 10, 30), cv.FONT_HERSHEY_PLAIN, 2.0, (0, 255, 255), 1)
    cv.imshow("gradient - prewitt operator", result)
    
    cv.waitKey(0)
    
    cv.destroyAllWindows()
    
    return cv.Error.StsOk

if __name__ == "__main__":
    OpencvDemo032()

# end of file
