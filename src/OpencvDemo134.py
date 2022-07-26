#!/usr/bin/env python3
#coding = utf-8

import logging
import numpy as np
import cv2 as cv

model_bin = "models/enet/model-best.net"

# OpenCV DNN ENet实现图像分割
def OpencvDemo134():
    logging.basicConfig(level=logging.DEBUG)

    # load a DNN model
    net = cv.dnn.readNetFromTorch(model_bin)
    if net is None:
        logging.error("could not load a DNN model!")
        return cv.Error.StsError
    
    # read an image
    src = cv.imread("images/cityscapes_test.jpg")
    if src is None:
        logging.error("could not load an image!")
        return cv.Error.StsError
    
    input = cv.dnn.blobFromImage(src, 0.00392, (512, 256), (0, 0, 0), True, False)

    # run the model
    net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
    # net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA_FP16)
    # net.setPreferableBackend(cv.dnn.DNN_BACKEND_INFERENCE_ENGINE)
    net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
    # net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
    net.setInput(input)
    output = net.forward()

    # get segmentation results
    output = np.squeeze(output)
    output = output.transpose((1, 2, 0))
    argMaxArray = np.argmax(output, 2)
    normalized = cv.normalize(argMaxArray, None, 0, 255, cv.NORM_MINMAX)
    normalized = cv.convertScaleAbs(normalized)
    colormapped = cv.applyColorMap(normalized, cv.COLORMAP_JET)
    colormapped = cv.resize(colormapped, src.shape[1::-1])
    dst = cv.addWeighted(src, 0.7, colormapped, 0.3, 0)

    time = net.getPerfProfile()[0] * 1000 / cv.getTickFrequency()
    text = "time: {:.0f} ms".format(time)
    logging.info(text)
    cv.putText(dst, text, (20, 100), cv.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

    h, w, ch = src.shape
    result = np.zeros([h * 2, w, ch], dtype=src.dtype)
    result[0 : h, 0 : w, :] = src
    result[h : h * 2, 0 : w, :] = dst
    cv.putText(result, "original image", (10, 40), cv.FONT_ITALIC, 1.2, (0, 0, 255), 2)
    cv.putText(result, "image with results", (10, h + 40), cv.FONT_ITALIC, 1.2, (0, 0, 255), 2)
    windowTitle = "image segmentation - ENet model"
    cv.namedWindow(windowTitle, cv.WINDOW_NORMAL)
    cv.resizeWindow(windowTitle, (w // 3, h * 2 // 3))
    cv.imshow(windowTitle, result)

    cv.waitKey(0)
    
    cv.destroyAllWindows()
    
    return cv.Error.StsOk

if __name__ == "__main__":
    OpencvDemo134()

# end of file
