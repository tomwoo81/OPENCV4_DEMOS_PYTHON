#!/usr/bin/env python3
#coding = utf-8

import logging
import math
import numpy as np
import cv2 as cv

labels = list()
text_features = list()

# 案例：识别0～9印刷体数字—Part 1
def OpencvDemo139():
    logging.basicConfig(level=logging.DEBUG)
    
    # 训练
    if cv.Error.StsOk != train_data():
        logging.error("Fail to train data!")
        return cv.Error.StsError
    
    cv.waitKey(0)
    
    cv.destroyAllWindows()
    
    return cv.Error.StsOk

def train_data():
    src = cv.imread("images/td2.png")
    if src is None:
        logging.error("could not load an image!")
        return cv.Error.StsError
    
    dst = np.copy(src)

    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    _, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)

    contours, _ = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    rects = list()
    for contour in contours:
        rect = cv.boundingRect(contour)
        rects.append(rect)
        cv.rectangle(dst, rect, (0, 0, 255), 1)
    
    # sort the ROIs in ascending order
    rects.sort(key=lambda r: r[0], reverse=False)

    for i in range(len(rects)):
        x, y, w, h = rects[i]
        roi = binary[y : y + h, x : x + w]
        feature = extract_feature(roi)

        text_features.append(feature)

        if (len(rects) - 1) == i:
            labels.append("0")
        else:
            labels.append("{:d}".format(i + 1))
    
    h, w, ch = src.shape
    result = np.zeros([h * 2, w, ch], dtype=src.dtype)
    result[0 : h, 0 : w, :] = src
    result[h : h * 2, 0 : w, :] = dst
    cv.putText(result, "original image", (10, 12), cv.FONT_ITALIC, .5, (0, 0, 255), 1)
    cv.putText(result, "image with digit ROIs", (10, h + 12), cv.FONT_ITALIC, .5, (0, 0, 255), 1)
    cv.imshow("digit recognition - train data", result)

    return cv.Error.StsOk

def extract_feature(txt_image):
    # total black pixels
    feature = list()

    height, width = txt_image.shape[:2]

    # feature
    bins = 10
    xstep = width / 4
    ystep = height / 5

    y = 0
    while y < height:
        x = 0
        while x < width:
            feature.append(get_weight_black_number(txt_image, width, height, x, y, xstep, ystep))
            x += xstep
        y += ystep
    
    # calculate Y Project
    xstep = width / bins
    x = 0
    while x < width:
        if (x + xstep) - width > 1:
            x += xstep
            continue
        feature.append(get_weight_black_number(txt_image, width, height, x, 0, xstep, height))
        x += xstep
    
    # calculate X Project
    ystep = height / bins
    y = 0
    while y < height:
        if (y + ystep) - height > 1:
            y += ystep
            continue
        feature.append(get_weight_black_number(txt_image, width, height, 0, y, width, ystep))
        y += ystep
    
    # normalization of feature vector

    # 4x5 cells = 20-element vector
    sum = 0
    for i in range(20):
        sum += feature[i]
    for i in range(20):
        feature[i] /= sum
    
    # Y Projection 10-element vector
    sum = 0
    for i in range(20, 30):
        sum += feature[i]
    for i in range(20, 30):
        feature[i] /= sum
    
    # X Projection 10-element vector
    sum = 0
    for i in range(30, 40):
        sum += feature[i]
    for i in range(30, 40):
        feature[i] /= sum
    
    return feature

def get_weight_black_number(image, width, height, x, y, xstep, ystep):
    weightNum = 0

    # 取整
    nx = int(math.floor(x))
    ny = int(math.floor(y))

    # 浮点数
    fx = x - nx
    fy = y - ny

    # 计算位置
    w = x + xstep
    h = y + ystep
    if w - width > 1e-3:
        w = width - 1
    if h - height > 1e-3:
        h = height - 1
    
    # 权重取整
    nw = int(math.floor(w))
    nh = int(math.floor(h))

    # 浮点数
    fw = w - nw
    fh = h - nh

    # 计算
    weight = 0
    for row in range(ny, nh):
        for col in range(nx, nw):
            c = image[row, col]
            if c == 0:
                weight += 1
    
    w1 = 0; w2 = 0; w3 = 0; w4 = 0

    # calculate w1
    if fx > 1e-3:
        col = nx + 1
        if col > width - 1:
            col = col - 1
        count = 0
        for row in range(ny, nh):
            c = image[row, col]
            if c == 0:
                count += 1
        w1 = count * fx
    
    # calculate w2
    if fy > 1e-3:
        row = ny + 1
        if row > height - 1:
            row = row - 1
        count = 0
        for col in range(nx, nw):
            c = image[row, col]
            if c == 0:
                count += 1
        w2 = count * fy
    
    # calculate w3
    if fw > 1e-3:
        col = nw + 1
        if col > width - 1:
            col = col - 1
        count = 0
        for row in range(ny, nh):
            c = image[row, col]
            if c == 0:
                count += 1
        w3 = count * fw
    
    # calculate w4
    if fh > 1e-3:
        row = nh + 1
        if row > height - 1:
            row = row - 1
        count = 0
        for col in range(nx, nw):
            c = image[row, col]
            if c == 0:
                count += 1
        w4 = count * fh
    
    weightNum = weight - w1 - w2 + w3 + w4

    if weightNum < 0:
        weightNum = 0
    
    return weightNum

if __name__ == "__main__":
    OpencvDemo139()

# end of file
