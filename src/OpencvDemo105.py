#!/usr/bin/env python3
#coding = utf-8

import os
import logging
import numpy as np
import cv2 as cv

test_dir = "images/train_data/elec_watch/test/"
model_filename = "models/svm_elec_watch.yml"

# HOG特征描述子—使用HOG进行对象检测
def OpencvDemo105():
    logging.basicConfig(level=logging.DEBUG)

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

def svm_predict():
    svm = cv.ml.SVM_load(model_filename)

    test_image_filenames = os.listdir(test_dir)
    test_image_filenames.sort()

    num_test_images = len(test_image_filenames)
    logging.info("number of test images: {:d}".format(num_test_images))

    for filename in test_image_filenames:
        dir_filename = os.path.join(test_dir, filename)
        src = cv.imread(dir_filename)

        h, w = src.shape[:2]

        if (h >= 1080) or (w >= 1920):
            img = cv.resize(src, (0, 0), fx=.20, fy=.20)
        else:
            img = np.copy(src)

        dst = np.copy(img)

        sum_x = 0; sum_y = 0; count = 0

        h, w = img.shape[:2]

        for row in range(128, h + 1, 4):
            for col in range(64, w + 1, 4):
                desc = get_hog_descriptor(img[row - 128 : row, col - 64 : col])
                fv = np.empty((len(desc)), dtype=np.float32)
                for i in range(len(desc)):
                    fv[i] = desc[i][0]
                fv = np.reshape(fv, (-1, len(desc)))

                result = svm.predict(fv)[1]
                result = result[0][0]

                if result > 0:
                    # cv.rectangle(dst, (col - 64, row - 128, 64, 128), (255, 0, 0), 1, cv.LINE_8, 0)
                    sum_x += col - 64
                    sum_y += row - 128
                    count += 1

        if count > 0:
            x = sum_x // count
            y = sum_y // count
            cv.rectangle(dst, (x, y, 64, 128), (0, 0, 255), 1, cv.LINE_8, 0)

        cv.putText(dst, "object detected: {:s}".format("yes" if count else "no"), (10, 30), cv.FONT_ITALIC, .6, (0, 0, 255), 1)
        cv.imshow("SVM prediction", dst)

        cv.waitKey(0)
    
    cv.destroyAllWindows()

if __name__ == "__main__":
    OpencvDemo105()

# end of file
