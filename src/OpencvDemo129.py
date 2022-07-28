#!/usr/bin/env python3
#coding = utf-8

import logging
import numpy as np
import cv2 as cv

model_coco_bin = "models/openpose/coco/pose_iter_440000.caffemodel"
model_coco_txt = "models/openpose/coco/openpose_pose_coco.prototxt"

model_mpi_bin = "models/openpose/mpi/pose_iter_160000.caffemodel"
model_mpi_txt = "models/openpose/mpi/openpose_pose_mpi_faster_4_stages.prototxt"

model_hand_bin = "models/openpose/hand/pose_iter_102000.caffemodel"
model_hand_txt = "models/openpose/hand/pose_deploy.prototxt"

POSE_PAIRS = [
	[   # COCO body model
		[ 1,2 ],[ 1,5 ],[ 2,3 ],
		[ 3,4 ],[ 5,6 ],[ 6,7 ],
		[ 1,8 ],[ 8,9 ],[ 9,10 ],
		[ 1,11 ],[ 11,12 ],[ 12,13 ],
		[ 1,0 ],[ 0,14 ],
		[ 14,16 ],[ 0,15 ],[ 15,17 ]
    ],
	[   # MPI body model
		[ 0,1 ],[ 1,2 ],[ 2,3 ],
		[ 3,4 ],[ 1,5 ],[ 5,6 ],
		[ 6,7 ],[ 1,14 ],[ 14,8 ],[ 8,9 ],
		[ 9,10 ],[ 14,11 ],[ 11,12 ],[ 12,13 ]
    ],
	[   # hand model
		[ 0,1 ],[ 1,2 ],[ 2,3 ],[ 3,4 ],         # thumb
		[ 0,5 ],[ 5,6 ],[ 6,7 ],[ 7,8 ],         # index
		[ 0,9 ],[ 9,10 ],[ 10,11 ],[ 11,12 ],    # middle
		[ 0,13 ],[ 13,14 ],[ 14,15 ],[ 15,16 ],  # ring
		[ 0,17 ],[ 17,18 ],[ 18,19 ],[ 19,20 ]   # pinkie
	] ]

# OpenCV DNN 调用OpenPose模型实现姿态评估
def OpencvDemo129():
    logging.basicConfig(level=logging.DEBUG)

    # run COCO body OpenPose model
    ret, dst_coco = RunOpenPoseModel(model_coco_txt, model_coco_bin, "images/green.jpg")
    if cv.Error.StsOk != ret:
        logging.error("Fail to run COCO body OpenPose model!")
        return cv.Error.StsError
    
    # run MPI body OpenPose model
    ret, dst_mpi = RunOpenPoseModel(model_mpi_txt, model_mpi_bin, "images/green.jpg")
    if cv.Error.StsOk != ret:
        logging.error("Fail to run MPI body OpenPose model!")
        return cv.Error.StsError
    
    # run hand OpenPose model
    ret, dst_hand = RunOpenPoseModel(model_hand_txt, model_hand_bin, "images/hand.jpg")
    if cv.Error.StsOk != ret:
        logging.error("Fail to run hand OpenPose model!")
        return cv.Error.StsError
    
    if dst_coco.shape[1] != dst_mpi.shape[1] or dst_coco.shape[0] != dst_mpi.shape[0]:
        logging.error("The size of 2 images with results is NOT equal!")
        return cv.Error.StsError
    
    h, w, ch = dst_coco.shape
    result = np.zeros([h, w * 2, ch], dtype=dst_coco.dtype)
    result[0 : h, 0 : w, :] = dst_coco
    result[0 : h, w : w * 2, :] = dst_mpi
    cv.putText(result, "image with results (COCO body model)", (10, 30), cv.FONT_ITALIC, 1.2, (0, 0, 255), 2)
    cv.putText(result, "image with results (MPI body model)", (w + 10, 30), cv.FONT_ITALIC, 1.2, (0, 0, 255), 2)
    windowTitle = "body pose estimation - OpenPose models"
    cv.namedWindow(windowTitle, cv.WINDOW_NORMAL)
    cv.resizeWindow(windowTitle, (int(w / 2), int(h / 4)))
    cv.imshow(windowTitle, result)

    cv.putText(dst_hand, "image with results (hand model)", (10, 30), cv.FONT_ITALIC, 0.6, (0, 0, 255), 1)
    cv.imshow("hand pose estimation - OpenPose model", dst_hand)

    cv.waitKey(0)
    
    cv.destroyAllWindows()
    
    return cv.Error.StsOk

def RunOpenPoseModel(modelTxt, modelBin, imageFilename):
    # load a DNN model
    net = cv.dnn.readNetFromCaffe(modelTxt, modelBin)
    if net is None:
        logging.error("could not load a DNN model!")
        return (cv.Error.StsError, None)
    
    # read an image
    src = cv.imread(imageFilename)
    if src is None:
        logging.error("could not load an image!")
        return (cv.Error.StsError, None)
    
    height, width = src.shape[:2]
    
    dst = np.copy(src)

    input = cv.dnn.blobFromImage(src, 1.0 / 255, (368, 368), (0, 0, 0), swapRB=False, crop=False)

    # run the model
    net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
    # net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA_FP16)
    net.setPreferableBackend(cv.dnn.DNN_BACKEND_DEFAULT)
    # net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
    net.setInput(input)
    output = net.forward()

    nparts = output.shape[1]
    h_output = output.shape[2]
    w_output = output.shape[3]

    # find out which model is used
    if nparts == 19:
        # COCO body model
        midx = 0
        npairs = 17
        nparts = 18 # skip background
    elif nparts == 16:
        # MPI body model
        midx = 1
        npairs = 14
    elif nparts == 22:
        # hand model
        midx = 2
        npairs = 20
    else:
        logging.error("There should be 19 parts for the COCO model, 16 for MPI, or 22 for the hand one, but this model has {:d} parts!".format(nparts))
        return cv.Error.StsError
    
    # find the position of the body/hand parts
    points = []
    thresh = 0.1
    for i in range(nparts):
        # slice heatmap of corresponding body's/hand's part
        heatMap = output[0, i, :, :]

        _, conf, _, point = cv.minMaxLoc(heatMap)
        points.append(point if conf > thresh else None)
    
    # draw lines to connect body/hand parts
    sx = width / w_output
    sy = height / h_output
    for i in range(npairs):
        # lookup 2 connected body/hand parts
        a = points[POSE_PAIRS[midx][i][0]]
        b = points[POSE_PAIRS[midx][i][1]]

        # we did not find enough confidence before
        if a is None or b is None:
            continue

        # scale to image size
        a = (int(a[0] * sx), int(a[1] * sy))
        b = (int(b[0] * sx), int(b[1] * sy))

        cv.circle(dst, a, 3, (0, 0, 200), cv.FILLED)
        cv.circle(dst, b, 3, (0, 0, 200), cv.FILLED)
        cv.line(dst, a, b, (0, 200, 0), 2)
    
    time = net.getPerfProfile()[0] * 1000 / cv.getTickFrequency()
    text = "time: {:.0f} ms".format(time)
    cv.putText(dst, text, (20, 60), cv.FONT_HERSHEY_SIMPLEX, 0.7, (200, 0, 0), 2)

    return (cv.Error.StsOk, dst)

if __name__ == "__main__":
    OpencvDemo129()

# end of file
