#!/usr/bin/env python3
#coding = utf-8

import logging
import numpy as np
import cv2 as cv

# KNN算法介绍
def OpencvDemo114():
    logging.basicConfig(level=logging.DEBUG)
    
    # 读取数据
    data = cv.imread("images/digits.png")
    if data is None:
        logging.error("could not load an image!")
        return cv.Error.StsError
    
    gray = cv.cvtColor(data, cv.COLOR_BGR2GRAY)
    images = [np.hsplit(row, 100) for row in np.vsplit(gray, 50)]
    images = np.array(images)

    # 创建训练数据
    train_data = images[:, :50].reshape((-1, 400)).astype(np.float32)
    k = np.arange(10)
    train_labels = np.repeat(k, 5 * 50)[:, np.newaxis]

    # 训练KNN模型
    knn = cv.ml.KNearest_create()
    knn.setDefaultK(5)
    knn.setIsClassifier(True)
    logging.info("KNN training starts...")
    knn.train(train_data, cv.ml.ROW_SAMPLE, train_labels)
    logging.info("KNN training done.")
    knn.save("models/knn_digits.yml")

    return cv.Error.StsOk

if __name__ == "__main__":
    OpencvDemo114()

# end of file
