#!/usr/bin/env python3
#coding = utf-8

import logging
import numpy as np
import cv2 as cv

# KNN算法应用
def OpencvDemo115():
    logging.basicConfig(level=logging.DEBUG)
    
    # 读取数据
    data = cv.imread("images/digits.png")
    if data is None:
        logging.error("could not load an image!")
        return cv.Error.StsError
    
    gray = cv.cvtColor(data, cv.COLOR_BGR2GRAY)
    images = [np.hsplit(row, 100) for row in np.vsplit(gray, 50)]
    images = np.array(images)

    # 创建测试数据
    test_data = images[:, 50:100].reshape((-1, 400)).astype(np.float32)
    k = np.arange(10)
    test_labels = np.repeat(k, 5 * 50)[:, np.newaxis]

    # 加载KNN分类器进行预测
    knn = cv.ml.KNearest_load("models/knn_digits.yml")
    logging.info("KNN prediction starts...")
    results = knn.findNearest(test_data, 5)[1]
    logging.info("KNN prediction done.")
    correct_count = np.count_nonzero(results == test_labels)
    accuracy = correct_count / results.shape[0]
    logging.info("accuracy: {:.3f}".format(accuracy))

    # 测试2张图片
    t1 = cv.imread("images/knn_01.png", cv.IMREAD_GRAYSCALE)
    if t1 is None:
        logging.error("could not load an image!")
        return cv.Error.StsError
    
    t2 = cv.imread("images/knn_02.png", cv.IMREAD_GRAYSCALE)
    if t2 is None:
        logging.error("could not load an image!")
        return cv.Error.StsError
    
    m1 = cv.resize(t1, (20, 20))
    m2 = cv.resize(t2, (20, 20))

    test_data = np.empty((2, 400), dtype=np.uint8)
    test_labels = np.empty((2, 1), dtype=np.uint8)
    one_row_1 = m1.reshape(-1, 400)
    one_row_2 = m2.reshape(-1, 400)
    test_data[0, :] = one_row_1
    test_data[1, :] = one_row_2
    test_labels[0, 0] = 1
    test_labels[1, 0] = 2

    test_data = test_data.astype(np.float32)
    test_labels = test_labels.astype(np.int32)

    results = knn.findNearest(test_data, 5)[1]

    logging.info("KNN prediction - image 1: label: {:d}, prediction: {:d}".format(test_labels[0, 0], int(results[0, 0])))
    cv.imshow("KNN prediction - image 1", t1)

    logging.info("KNN prediction - image 2: label: {:d}, prediction: {:d}".format(test_labels[1, 0], int(results[1, 0])))
    cv.imshow("KNN prediction - image 2", t2)

    cv.waitKey(0)
    
    cv.destroyAllWindows()
    
    return cv.Error.StsOk

if __name__ == "__main__":
    OpencvDemo115()

# end of file
