#!/usr/bin/env python3
#coding = utf-8

import logging
import cv2 as cv

bin_model = "models/googlenet/bvlc_googlenet.caffemodel"
protxt = "models/googlenet/bvlc_googlenet.prototxt"

# OpenCV DNN 获取导入模型各层信息
def OpencvDemo121():
    logging.basicConfig(level=logging.DEBUG)

    # load a DNN model
    net = cv.dnn.readNet(bin_model, protxt)
    if net is None:
        logging.error("could not load a DNN model!")
        return cv.Error.StsError
    logging.info("Successfully loaded a DNN model.")

    # get info of layers
    layer_names = net.getLayerNames()
    for name in layer_names:
        id = net.getLayerId(name)
        layer = net.getLayer(id)
        logging.info("layer id: {:d}, layer type: {}, layer name: {}".format(id, layer.type, layer.name))
    
    return cv.Error.StsOk

if __name__ == "__main__":
    OpencvDemo121()

# end of file
