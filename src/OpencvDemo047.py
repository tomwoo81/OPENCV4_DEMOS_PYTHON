#!/usr/bin/env python3
#coding = utf-8

import logging
import numpy as np
import cv2 as cv

# 二值图像连通组件状态统计
def OpencvDemo047():
    logging.basicConfig(level=logging.DEBUG)
    
    src = cv.imread("images/rice.png")
    if src is None:
        logging.error("could not load an image!")
        return cv.Error.StsError
    
    dst = connected_components_with_stats_demo(src)
    
    h, w, ch = src.shape
    result = np.zeros([h * 2, w, ch], dtype=src.dtype)
    result[0 : h, 0 : w, :] = src
    result[h : h * 2, 0 : w, :] = dst
    cv.putText(result, "original image", (10, 30), cv.FONT_ITALIC, 1.0, (0, 0, 255), 2)
    cv.putText(result, "labeled image", (10, h + 30), cv.FONT_ITALIC, 1.0, (0, 0, 255), 2)
    cv.imshow("connected components with statistics", result)
    
    cv.waitKey(0)
    
    cv.destroyAllWindows()
    
    return cv.Error.StsOk

def connected_components_with_stats_demo(src):
    blurred = cv.GaussianBlur(src, (3, 3), 0)
    gray = cv.cvtColor(blurred, cv.COLOR_BGR2GRAY)
    _, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    
    # extract labels
    num_labels, _, stats, centroids = cv.connectedComponentsWithStats(binary, connectivity=8, ltype=cv.CV_32S)
    logging.info("number of foreground labels: {:d}".format(num_labels - 1))
    
    colors = [0] * num_labels
    
    # background color
    colors[0] = (0, 0, 0)
    
    # foreground colors
    for i in range(1, num_labels):
        colors[i] = (np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256))
    
    # extract statistics info
    dst = np.copy(src)
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        
        cv.rectangle(dst, (x, y, w, h), colors[i], 1, cv.LINE_8, 0)
        cv.putText(dst, "index: {:d}".format(i), (x, y), cv.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 255), 1)
        logging.info("index: {:d}, area: {:d}".format(i, area))
        
        cx, cy = np.int32(centroids[i])
        
        cv.circle(dst, (cx, cy), 2, (0, 255, 0), 2, cv.LINE_8, 0)
    
    return dst

if __name__ == "__main__":
    OpencvDemo047()

# end of file
