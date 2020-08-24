#!/usr/bin/env python3
#coding = utf-8

import logging
import cv2 as cv

# 图像直方图比较
def OpencvDemo019():
    logging.basicConfig(level=logging.DEBUG)
    
    src1 = cv.imread("images/m1.png")
    src2 = cv.imread("images/m2.png")
    src3 = cv.imread("images/flower.png")
    src4 = cv.imread("images/test.png")
    if (src1 is None) or (src2 is None) or (src3 is None) or (src4 is None):
        logging.error("could not load image(s)!")
        return cv.Error.StsError
    cv.imshow("input1", src1)
    cv.imshow("input2", src2)
    cv.imshow("input3", src3)
    cv.imshow("input4", src4)
    
    hsv1 = cv.cvtColor(src1, cv.COLOR_BGR2HSV)
    hsv2 = cv.cvtColor(src2, cv.COLOR_BGR2HSV)
    hsv3 = cv.cvtColor(src3, cv.COLOR_BGR2HSV)
    hsv4 = cv.cvtColor(src4, cv.COLOR_BGR2HSV)
    
    hist1 = cv.calcHist([hsv1], [0, 1], None, [60, 64], [0, 180, 0, 256])
    hist2 = cv.calcHist([hsv2], [0, 1], None, [60, 64], [0, 180, 0, 256])
    hist3 = cv.calcHist([hsv3], [0, 1], None, [60, 64], [0, 180, 0, 256])
    hist4 = cv.calcHist([hsv4], [0, 1], None, [60, 64], [0, 180, 0, 256])
    logging.debug("hist1.dtype: %s", hist1.dtype)
    
    cv.normalize(hist1, hist1, 0, 1.0, cv.NORM_MINMAX, dtype=cv.CV_32F)
    cv.normalize(hist2, hist2, 0, 1.0, cv.NORM_MINMAX)
    cv.normalize(hist3, hist3, 0, 1.0, cv.NORM_MINMAX)
    cv.normalize(hist4, hist4, 0, 1.0, cv.NORM_MINMAX)
    
    methods = [cv.HISTCMP_CORREL, cv.HISTCMP_CHISQR,
               cv.HISTCMP_INTERSECT, cv.HISTCMP_BHATTACHARYYA]
    str_method = ""
    for method in methods:
        src1_src2 = cv.compareHist(hist1, hist2, method)
        src3_src4 = cv.compareHist(hist3, hist4, method)
        if method == cv.HISTCMP_CORREL:
            str_method = "Correlation"
        elif method == cv.HISTCMP_CHISQR:
            str_method = "Chi-square"
        elif method == cv.HISTCMP_INTERSECT:
            str_method = "Intersection"
        elif method == cv.HISTCMP_BHATTACHARYYA:
            str_method = "Bhattacharyya"
        logging.info("Method [%s]: src1_src2: %.3f, src3_src4: %.3f", str_method, src1_src2, src3_src4)
    
    cv.waitKey(0)
    
    cv.destroyAllWindows()
    
    return cv.Error.StsOk

if __name__ == "__main__":
    OpencvDemo019()

# end of file
