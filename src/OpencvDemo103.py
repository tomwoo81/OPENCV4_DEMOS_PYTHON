#!/usr/bin/env python3
#coding = utf-8

import os
import logging
import numpy as np
import cv2 as cv

positive_dir = "images/train_data/elec_watch/positive/"
negative_dir = "images/train_data/elec_watch/negative/"

# HOG特征描述子—使用描述子特征生成样本数据
def OpencvDemo103():
    logging.basicConfig(level=logging.DEBUG)

    generate_dataset()

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
        cv.imshow("image as positive sample", image)
        cv.waitKey(0)

        desc = get_hog_descriptor(image)
        fv = np.empty((len(desc)), dtype=np.float32)
        for i in range(len(desc)):
            fv[i] = desc[i][0]
        logging.info("image path: {:s}, feature data length: {:d}".format(dir_filename, fv.shape[0]))

        train_data.append(fv)
        labels.append(1)

    logging.info("number of images as negative samples: {:d}".format(num_negative_images))
    for filename in negative_image_filenames:
        dir_filename = os.path.join(negative_dir, filename)
        image = cv.imread(dir_filename)
        cv.imshow("image as negative sample", image)
        cv.waitKey(0)

        desc = get_hog_descriptor(image)
        fv = np.empty((len(desc)), dtype=np.float32)
        for i in range(len(desc)):
            fv[i] = desc[i][0]
        logging.info("image path: {:s}, feature data length: {:d}".format(dir_filename, fv.shape[0]))

        train_data.append(fv)
        labels.append(-1)

    cv.destroyAllWindows()

    return np.array(train_data, dtype=np.float32), np.array(labels, dtype=np.int32)

if __name__ == "__main__":
    OpencvDemo103()

# end of file
