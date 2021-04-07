#!/usr/bin/env python3
#coding = utf-8

import logging
import numpy as np
import cv2 as cv

# 二值图像分析—缺陷检测二
def OpencvDemo073():
    logging.basicConfig(level=logging.DEBUG)
    
    src = cv.imread("images/ce_02.jpg")
    if src is None:
        logging.error("could not load an image!")
        return cv.Error.StsError
    
    # 二值化图像
    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    _, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)

    # 定义结构元素
    se = cv.getStructuringElement(cv.MORPH_RECT, (3, 3), (-1, -1))

    # 开操作
    open = cv.morphologyEx(binary, cv.MORPH_OPEN, se)

    # 轮廓发现/轮廓分析
    contours, _ = cv.findContours(open, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

    dst = np.copy(src)

    height = src.shape[0]

    rects = list()

    for c in contours:
        x, y, w, h = cv.boundingRect(c)
        area = cv.contourArea(c)

        if h > (height // 2):
            continue

        if area < 150:
            continue

        # 轮廓填充/扩大
        cv.drawContours(binary, [c], 0, (0), 2, cv.LINE_8)

        rects.append((x, y, w, h))
    
    # 轮廓排序
    rects = sort_boxes(rects)

    for i in range(len(rects)):
        cv.putText(dst, "num: {:d}".format(i+1), 
                   (rects[i][0]-60, rects[i][1]+15), cv.FONT_HERSHEY_PLAIN, 1.0, (255, 0, 0), 1)
    
    tpl = get_template(binary, rects)

    # 模板比对
    defects = detect_defects(binary, rects, tpl)

    # 输出结果
    for x, y, w, h in defects:
        cv.rectangle(dst, (x, y, w, h), (0, 0, 255), 1, cv.LINE_8, 0)
        cv.putText(dst, "bad", (x, y), cv.FONT_HERSHEY_PLAIN, 1.0, (0, 255, 0), 2)
    
    h, w, ch = src.shape
    result = np.zeros([h, w * 2, ch], dtype=src.dtype)
    result[0 : h, 0 : w, :] = src
    result[0 : h, w : w * 2, :] = dst
    cv.putText(result, "original image", (10, 30), cv.FONT_ITALIC, .8, (0, 0, 255), 1)
    cv.putText(result, "image with results of defect detection", (w + 10, 30), cv.FONT_ITALIC, .8, (0, 0, 255), 1)
    cv.imshow("defect detection - 2", result)

    cv.waitKey(0)
    
    cv.destroyAllWindows()
    
    return cv.Error.StsOk

def sort_boxes(boxes):
    size = len(boxes)

    for i in range(0, size - 1):
        for j in range(i, size):
            if boxes[j][1] < boxes[i][1]:
                boxes[j], boxes[i] = boxes[i], boxes[j]
    
    return boxes

def get_template(binary, rects):
    x, y, w, h = rects[0]
    return binary[y : y + h, x : x + w]

def detect_defects(binary, rects, tpl):
    height, width = tpl.shape
    
    defects = list()

    for x, y, w, h in rects:
        roi = binary[y : y + h, x : x + w]
        roi = cv.resize(roi, (width, height))

        mask = cv.subtract(tpl, roi)

        se = cv.getStructuringElement(cv.MORPH_RECT, (5, 5), (-1, -1))

        mask = cv.morphologyEx(mask, cv.MORPH_OPEN, se)
        _, mask = cv.threshold(mask, 0, 255, cv.THRESH_BINARY)
        
        count = 0

        for row in range(height):
            for col in range(width):
                pv = mask[row, col]
                if pv == 255:
                    count += 1
        
        if count > 0:
            defects.append((x, y, w, h))
    
    return defects

if __name__ == "__main__":
    OpencvDemo073()

# end of file
