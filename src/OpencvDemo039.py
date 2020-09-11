#!/usr/bin/env python3
#coding = utf-8

import logging
import numpy as np
import cv2 as cv

# 图像模板匹配
def OpencvDemo039():
    logging.basicConfig(level=logging.DEBUG)
    
    src = cv.imread("images/llk.jpg")
    tpl = cv.imread("images/llk_tpl.png")
    if (src is None) or (tpl is None):
        logging.error("could not load image(s)!")
        return cv.Error.StsError
    cv.imshow("input", src)
    cv.imshow("template", tpl)
    
    result = cv.matchTemplate(src, tpl, cv.TM_CCOEFF_NORMED)
    cv.imshow("comparison results", result)
    
    t = 0.95
    
    loc = np.where(result > t)
    tpl_h, tpl_w = tpl.shape[:2]
    for pt in zip(*loc[::-1]):
        cv.rectangle(src, pt, (pt[0] + tpl_w, pt[1] + tpl_h), (255, 0, 0), 1, cv.LINE_8, 0)
    cv.imshow("template-matched results", src)
    
    cv.waitKey(0)
    
    cv.destroyAllWindows()
    
    return cv.Error.StsOk

if __name__ == "__main__":
    OpencvDemo039()

# end of file
