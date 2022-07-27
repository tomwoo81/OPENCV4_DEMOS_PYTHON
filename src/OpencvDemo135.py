#!/usr/bin/env python3
#coding = utf-8

import logging
import numpy as np
import cv2 as cv

base_dir = "models/fast_style/"
models_bin = [
"composition_vii.t7", 
"starry_night.t7", 
"la_muse.t7", 
"the_wave.t7", 
"mosaic.t7", 
"the_scream.t7", 
"feathers.t7", 
"candy.t7", 
"udnie.t7"
]

# OpenCV DNN 实时快速的图像风格迁移
def OpencvDemo135():
    logging.basicConfig(level=logging.DEBUG)

    # read an image
    src = cv.imread("images/lena.png")
    if src is None:
        logging.error("could not load an image!")
        return cv.Error.StsError
    
    dsts = list()
    
    for i in range(9):
        ret, dst = RunStyleTransferModel(base_dir + models_bin[i], src)
        if cv.Error.StsOk != ret:
            logging.error("Fail to run style transfer model!")
            return cv.Error.StsError
        
        dsts.append(dst)
    
    h, w, ch = src.shape
    result = np.zeros([h * 2, w * 5, ch], dtype=src.dtype)
    result[0 : h, 0 : w, :] = src
    result[0 : h, w : w * 2, :] = dsts[0]
    result[0 : h, w * 2 : w * 3, :] = dsts[1]
    result[0 : h, w * 3 : w * 4, :] = dsts[2]
    result[0 : h, w * 4 : w * 5, :] = dsts[3]
    result[h : h * 2, 0 : w, :] = dsts[4]
    result[h : h * 2, w : w * 2, :] = dsts[5]
    result[h : h * 2, w * 2 : w * 3, :] = dsts[6]
    result[h : h * 2, w * 3 : w * 4, :] = dsts[7]
    result[h : h * 2, w * 4 : w * 5, :] = dsts[8]
    cv.putText(result, "original", (10, 40), cv.FONT_ITALIC, 1.2, (0, 0, 255), 2)
    cv.putText(result, "composition vii", (w + 10, 40), cv.FONT_ITALIC, 1.2, (0, 0, 255), 2)
    cv.putText(result, "starry night", (w * 2 + 10, 40), cv.FONT_ITALIC, 1.2, (0, 0, 255), 2)
    cv.putText(result, "la muse", (w * 3 + 10, 40), cv.FONT_ITALIC, 1.2, (0, 0, 255), 2)
    cv.putText(result, "the wave", (w * 4 + 10, 40), cv.FONT_ITALIC, 1.2, (0, 0, 255), 2)
    cv.putText(result, "mosaic", (10, h + 40), cv.FONT_ITALIC, 1.2, (0, 0, 255), 2)
    cv.putText(result, "the scream", (w + 10, h + 40), cv.FONT_ITALIC, 1.2, (0, 0, 255), 2)
    cv.putText(result, "feathers", (w * 2 + 10, h + 40), cv.FONT_ITALIC, 1.2, (0, 0, 255), 2)
    cv.putText(result, "candy", (w * 3 + 10, h + 40), cv.FONT_ITALIC, 1.2, (0, 0, 255), 2)
    cv.putText(result, "udnie", (w * 4 + 10, h + 40), cv.FONT_ITALIC, 1.2, (0, 0, 255), 2)
    windowTitle = "fast style transfer - DCGAN model"
    cv.namedWindow(windowTitle, cv.WINDOW_NORMAL)
    cv.resizeWindow(windowTitle, (w * 5 // 2, h))
    cv.imshow(windowTitle, result)

    cv.waitKey(0)
    
    cv.destroyAllWindows()
    
    return cv.Error.StsOk

def RunStyleTransferModel(modelBin, src):
    # load a DNN model
    net = cv.dnn.readNetFromTorch(modelBin)
    if net is None:
        logging.error("could not load a DNN model!")
        return cv.Error.StsError
    
    input = cv.dnn.blobFromImage(src, 1.0, (256, 256), (103.939, 116.779, 123.68), False, False)

    # run the model
    net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
    # net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA_FP16)
    net.setPreferableBackend(cv.dnn.DNN_BACKEND_DEFAULT)
    # net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
    net.setInput(input)
    output = net.forward()

    # get style transfer results
    dst = np.squeeze(output)
    dst[0] += 103.939
    dst[1] += 116.779
    dst[2] += 123.68
    dst = dst.transpose((1, 2, 0))
    dst = cv.convertScaleAbs(dst)

    dst = cv.medianBlur(dst, 5)
    dst = cv.resize(dst, src.shape[1::-1])

    time = net.getPerfProfile()[0] * 1000 / cv.getTickFrequency()
    text = "time: {:.0f} ms".format(time)
    logging.info("model: {}, ".format(modelBin) + text)
    cv.putText(dst, text, (20, 100), cv.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 0), 2)

    return (cv.Error.StsOk, dst)

if __name__ == "__main__":
    OpencvDemo135()

# end of file
