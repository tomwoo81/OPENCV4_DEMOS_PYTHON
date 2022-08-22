#!/usr/bin/env python3
#coding = utf-8

import logging
import numpy as np
import cv2 as cv
from OpencvDemo139 import labels, text_features, train_data, extract_feature

# 案例：识别0～9印刷体数字—Part 2
def OpencvDemo140():
    logging.basicConfig(level=logging.DEBUG)
    
    # 训练
    if cv.Error.StsOk != train_data():
        logging.error("Fail to train data!")
        return cv.Error.StsError
    
    # 测试
    if cv.Error.StsOk != test_data():
        logging.error("Fail to test data!")
        return cv.Error.StsError
    
    cv.waitKey(0)
    
    cv.destroyAllWindows()
    
    return cv.Error.StsOk

def test_data():
    src = cv.imread("images/digit-01.png")
    if src is None:
        logging.error("could not load an image!")
        return cv.Error.StsError
    
    dst = np.copy(src)

    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    _, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)

    contours, _ = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        rect = cv.boundingRect(contour)

        x, y, w, h = rect
        roi = binary[y : y + h, x : x + w]
        feature = extract_feature(roi)

        result = predict_digit(feature)
        cv.putText(dst, result, (x, y + h + 15), cv.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 255), 1)
    
    h, w, ch = src.shape
    result = np.zeros([h, w * 2, ch], dtype=src.dtype)
    result[0 : h, 0 : w, :] = src
    result[0 : h, w : w * 2, :] = dst
    cv.putText(result, "original image", (10, 12), cv.FONT_ITALIC, .5, (0, 0, 255), 1)
    cv.putText(result, "image with results", (w + 10, 12), cv.FONT_ITALIC, .5, (0, 0, 255), 1)
    cv.imshow("digit recognition - test data", result)

    return cv.Error.StsOk

def predict_digit(feature):
    min_dist = float('inf')
    index = -1

    for i in range(len(text_features)):
        dist = 0
        temp = text_features[i]

        for j in range(len(feature)):
            d = temp[j] - feature[j]
            dist += d * d
        
        if dist < min_dist:
            min_dist = dist
            index = i
    
    if index >= 0:
        result = labels[index]
    else:
        result = "-"
    
    return result

if __name__ == "__main__":
    OpencvDemo140()

# end of file
