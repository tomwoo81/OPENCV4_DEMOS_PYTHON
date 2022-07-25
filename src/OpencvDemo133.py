#!/usr/bin/env python3
#coding = utf-8

import logging
import numpy as np
import cv2 as cv

model_bin = "models/colorization/colorization_release_v2.caffemodel"
model_txt = "models/colorization/colorization_deploy_v2.prototxt"
pts_bin = "models/colorization/pts_in_hull.npy"

# OpenCV DNN 图像彩色化模型使用
def OpencvDemo133():
    logging.basicConfig(level=logging.DEBUG)

    # load a DNN model
    net = cv.dnn.readNetFromCaffe(model_txt, model_bin)
    if net is None:
        logging.error("could not load a DNN model!")
        return cv.Error.StsError
    
    # load cluster centres
    pts_in_hull = np.load(pts_bin)

    # populate additional layers
    pts_in_hull = pts_in_hull.transpose().reshape(2, 313, 1, 1)
    net.getLayer("class8_ab").blobs = [pts_in_hull.astype(np.float32)]
    net.getLayer("conv8_313_rh").blobs = [np.full((1, 313), 2.606, np.float32)]

    # read an image
    src = cv.imread("images/test1.png")
    if src is None:
        logging.error("could not load an image!")
        return cv.Error.StsError
    
    # extract L channel and subtract mean
    temp = (src / 255).astype(np.float32)
    srcLab = cv.cvtColor(temp, cv.COLOR_BGR2Lab)
    srcL = cv.extractChannel(srcLab, 0)

    input = cv.dnn.blobFromImage(srcL, 1.0, (224, 224), 50)

    # run the model
    net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
    # net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA_FP16)
    net.setPreferableBackend(cv.dnn.DNN_BACKEND_DEFAULT)
    # net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
    net.setInput(input)
    output = net.forward()

    # retrieve the calculated a, b channels from the network output
    a = output[0, 0]
    b = output[0, 1]
    a = cv.resize(a, src.shape[1::-1])
    b = cv.resize(b, src.shape[1::-1])

    # merge L, a, b channels, and convert the image back to BGR space
    dstLab = cv.merge([srcL, a, b])
    dst = cv.cvtColor(dstLab, cv.COLOR_Lab2BGR)
    dst = (dst * 255).astype(np.uint8)

    time = net.getPerfProfile()[0] * 1000 / cv.getTickFrequency()
    text = "time: {:.0f} ms".format(time)
    logging.info(text)
    cv.putText(dst, text, (20, 60), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)

    h, w, ch = src.shape
    result = np.zeros([h, w * 3, ch], dtype=src.dtype)
    result[0 : h, 0 : w, :] = src
    cv.normalize(srcL, srcL, 0, 255, cv.NORM_MINMAX)
    srcL = cv.convertScaleAbs(srcL)
    srcL = cv.cvtColor(srcL, cv.COLOR_GRAY2BGR) # 1 channel -> 3 channels
    result[0 : h, w : w * 2, :] = srcL
    result[0 : h, w * 2 : w * 3, :] = dst
    cv.putText(result, "original image", (10, 30), cv.FONT_ITALIC, .6, (0, 0, 255), 1)
    cv.putText(result, "image with L component", (w + 10, 30), cv.FONT_ITALIC, .6, (0, 0, 255), 1)
    cv.putText(result, "image after colourisation", (w * 2 + 10, 30), cv.FONT_ITALIC, .6, (0, 0, 255), 1)
    cv.imshow("image colourisation", result)

    cv.waitKey(0)
    
    cv.destroyAllWindows()
    
    return cv.Error.StsOk

if __name__ == "__main__":
    OpencvDemo133()

# end of file
