#!/usr/bin/env python3
#coding = utf-8

import os
import logging
import numpy as np
import cv2 as cv

positive_dir = "images/train_data/elec_watch/positive/"
negative_dir = "images/train_data/elec_watch/negative/"
test_filename = "images/train_data/elec_watch/test/box_04.bmp"
model_filename = "models/svm_elec_watch.yml"

# SVM线性分类器
def OpencvDemo104():
    logging.basicConfig(level=logging.DEBUG)

    train_data, labels = generate_dataset()

    svm_train(train_data, labels)

    svm_predict()

    return cv.Error.StsOk

def get_hog_descriptor(image):
    h, w = image.shape[:2]

    rate = 64 / w

    img = cv.resize(image, (64, int(h * rate)))
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    result = np.full((128, 64), 127, dtype=np.uint8)

    gray_h = gray.shape[0]
    gray_y = (128 - gray_h) // 2

    result[gray_y : gray_y + gray_h, :] = gray

    hog = cv.HOGDescriptor()

    desc = hog.compute(result, winStride=(8, 8), padding=(0, 0))

    return desc

def generate_dataset():
    positive_image_filenames = os.listdir(positive_dir)
    negative_image_filenames = os.listdir(negative_dir)

    num_positive_images = len(positive_image_filenames)
    num_negative_images = len(negative_image_filenames)
    num_images = num_positive_images + num_negative_images
    logging.info("number of images: {:d}".format(num_images))

    train_data = list()
    labels = list()

    logging.info("number of images as positive samples: {:d}".format(num_positive_images))
    for filename in positive_image_filenames:
        dir_filename = os.path.join(positive_dir, filename)
        image = cv.imread(dir_filename)

        desc = get_hog_descriptor(image)
        fv = np.empty((len(desc)), dtype=np.float32)
        for i in range(len(desc)):
            fv[i] = desc[i][0]

        train_data.append(fv)
        labels.append(1)

    logging.info("number of images as negative samples: {:d}".format(num_negative_images))
    for filename in negative_image_filenames:
        dir_filename = os.path.join(negative_dir, filename)
        image = cv.imread(dir_filename)

        desc = get_hog_descriptor(image)
        fv = np.empty((len(desc)), dtype=np.float32)
        for i in range(len(desc)):
            fv[i] = desc[i][0]

        train_data.append(fv)
        labels.append(-1)

    cv.destroyAllWindows()

    return np.array(train_data, dtype=np.float32), np.array(labels, dtype=np.int32)

def svm_train(train_data, labels):
    svm = cv.ml.SVM_create()

    # Default values to train SVM
    svm.setKernel(cv.ml.SVM_LINEAR)
    svm.setType(cv.ml.SVM_C_SVC)
    svm.setC(2.67)
    svm.setGamma(5.383)

    logging.info("SVM training starts...")
    svm.train(train_data, cv.ml.ROW_SAMPLE, labels)
    logging.info("SVM training done.")

    # Save SVM model
    svm.save(model_filename)

def svm_predict():
    svm = cv.ml.SVM_load(model_filename)

    src = cv.imread(test_filename)

    desc = get_hog_descriptor(src)
    fv = np.empty((len(desc)), dtype=np.float32)
    for i in range(len(desc)):
        fv[i] = desc[i][0]
    fv = np.reshape(fv, (-1, len(desc)))

    result = svm.predict(fv)[1]
    result = result[0][0]

    dst = np.copy(src)

    cv.putText(dst, "prediction result: {:.3f}".format(result), (10, 30), cv.FONT_ITALIC, .6, (0, 0, 255), 1)
    cv.imshow("SVM prediction", dst)

    cv.waitKey(0)
    
    cv.destroyAllWindows()

if __name__ == "__main__":
    OpencvDemo104()

# end of file
