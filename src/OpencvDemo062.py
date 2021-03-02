#!/usr/bin/env python3
#coding = utf-8

import logging
import numpy as np
import cv2 as cv

# 图像形态学—膨胀与腐蚀
def OpencvDemo062():
    logging.basicConfig(level=logging.DEBUG)
    
    src = cv.imread("images/wm.jpg")
    if src is None:
        logging.error("could not load an image!")
        return cv.Error.StsError
    
    # 定义结构元素 3x3大小矩形
    se = np.ones((3, 3), dtype=np.uint8)

    # 膨胀
    dilate = cv.dilate(src, se, None, (-1, -1), 1)

    # 腐蚀
    erode = cv.erode(src, se, None, (-1, -1), 1)
    
    h, w, ch = src.shape
    result = np.zeros([h, w * 3, ch], dtype=src.dtype)
    result[0 : h, 0 : w, :] = src
    result[0 : h, w : w * 2, :] = dilate
    result[0 : h, w * 2 : w * 3, :] = erode
    cv.putText(result, "original image", (10, 30), cv.FONT_ITALIC, .6, (0, 0, 255), 1)
    cv.putText(result, "image after dilation", (w + 10, 30), cv.FONT_ITALIC, .6, (0, 0, 255), 1)
    cv.putText(result, "image after erosion", (w * 2 + 10, 30), cv.FONT_ITALIC, .6, (0, 0, 255), 1)
    cv.imshow("dilation & erosion", result)
    
    cv.waitKey(0)
    
    cv.destroyAllWindows()
    
    return cv.Error.StsOk

if __name__ == "__main__":
    OpencvDemo062()

# end of file
