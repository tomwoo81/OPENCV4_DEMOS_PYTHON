#!/usr/bin/env python3
#coding = utf-8

import logging
import numpy as np
import cv2 as cv

# 二值图像分析—寻找最大内接圆
def OpencvDemo058():
    logging.basicConfig(level=logging.DEBUG)

    # 绘制六边形
    r = 100
    src = np.zeros((4 * r, 4 * r), dtype=np.uint8)
    vert = [None] * 6
    vert[0] = (3 * r // 2, int(1.34 * r))
    vert[1] = (1 * r, 2 * r)
    vert[2] = (3 * r // 2, int(2.866 * r))
    vert[3] = (5 * r // 2, int(2.866 * r))
    vert[4] = (3 * r, 2 * r)
    vert[5] = (5 * r // 2, int(1.34 * r))
    for i in range(6):
        cv.line(src, vert[i], vert[(i + 1) % 6], 255, 3)
    
    # 轮廓发现
    contours, _ = cv.findContours(src, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    # 计算每个像素到轮廓的距离
    raw_dist = np.ndarray(src.shape, dtype=np.float32)
    for row in range(src.shape[0]):
        for col in range(src.shape[1]):
            raw_dist[row, col] = cv.pointPolygonTest(contours[0], (col, row), True)
    
    # 获取最大内接圆的半径和圆心
    minVal, maxVal, _, maxValPt = cv.minMaxLoc(raw_dist)
    minVal = abs(minVal)
    maxVal = abs(maxVal)

    dst = np.zeros((src.shape[0], src.shape[1], 3), dtype=np.uint8)
    for row in range(src.shape[0]):
        for col in range(src.shape[1]):
            if raw_dist[row, col] < 0:
                dst[row, col, 0] = 255 - abs(raw_dist[row, col]) * 255 / minVal
            elif raw_dist[row, col] > 0:
                dst[row, col, 2] = 255 - raw_dist[row, col] * 255 / maxVal
            else:
                dst[row, col, 0] = 255
                dst[row, col, 1] = 255
                dst[row, col, 2] = 255
    
    # 绘制最大内接圆
    cv.circle(dst, maxValPt, int(maxVal), (255, 255, 255))

    src = cv.cvtColor(src, cv.COLOR_GRAY2BGR)

    h, w, ch = src.shape
    result = np.zeros([h, w * 2, ch], dtype=src.dtype)
    result[0 : h, 0 : w, :] = src
    result[0 : h, w : w * 2, :] = dst
    cv.putText(result, "original image", (10, 30), cv.FONT_ITALIC, .6, (0, 255, 0), 1)
    cv.putText(result, "image with maximum inscribed circle", (w + 10, 30), cv.FONT_ITALIC, .6, (0, 255, 0), 1)
    cv.imshow("maximum inscribed circle searching", result)

    cv.waitKey(0)
    
    cv.destroyAllWindows()
    
    return cv.Error.StsOk

if __name__ == "__main__":
    OpencvDemo058()

# end of file
