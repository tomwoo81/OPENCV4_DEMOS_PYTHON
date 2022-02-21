#!/usr/bin/env python3
#coding = utf-8

import logging
import numpy as np
import cv2 as cv

# 决策树算法 介绍与使用
def OpencvDemo116():
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

    # 训练RTrees模型
    rTrees = cv.ml.RTrees_create()
    # rTrees.setMaxDepth(10)
    # rTrees.setMinSampleCount(10)
    # rTrees.setRegressionAccuracy(0)
    # rTrees.setUseSurrogates(False)
    # rTrees.setMaxCategories(15)
    # rTrees.setPriors(np.ndarray((0)))
    # rTrees.setCalculateVarImportance(True)
    # rTrees.setActiveVarCount(4)
    term_crit = (cv.TERM_CRITERIA_MAX_ITER | cv.TERM_CRITERIA_EPS, 100, 0.01)
    rTrees.setTermCriteria(term_crit)
    logging.info("RTrees training starts...")
    rTrees.train(train_data, cv.ml.ROW_SAMPLE, train_labels)
    logging.info("RTrees training done.")

    # 创建测试数据
    test_data = images[:, 50:100].reshape((-1, 400)).astype(np.float32)
    k = np.arange(10)
    test_labels = np.repeat(k, 5 * 50)[:, np.newaxis]

    # 使用RTrees分类器进行预测
    logging.info("RTrees prediction starts...")
    results = rTrees.predict(test_data)[1]
    logging.info("RTrees prediction done.")
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

    results = rTrees.predict(test_data)[1]

    logging.info("RTrees prediction - image 1: label: {:d}, prediction: {:d}".format(test_labels[0, 0], int(results[0, 0])))
    cv.imshow("RTrees prediction - image 1", t1)

    logging.info("RTrees prediction - image 2: label: {:d}, prediction: {:d}".format(test_labels[1, 0], int(results[1, 0])))
    cv.imshow("RTrees prediction - image 2", t2)

    cv.waitKey(0)
    
    cv.destroyAllWindows()
    
    return cv.Error.StsOk

if __name__ == "__main__":
    OpencvDemo116()

# end of file
