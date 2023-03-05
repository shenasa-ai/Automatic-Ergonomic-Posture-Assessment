import cv2
from PIL import Image
import sys
import numpy as np
import enum
from src.pose_detector import PoseDetector


class OpenPoseDetector(PoseDetector):
    Nose = 0
    Neck = 1
    RShoulder = 2
    RElbow = 3
    RWrist = 4
    LShoulder = 5
    LElbow = 6
    LWrist = 7
    RHip = 8
    RKnee = 9
    RAnkle = 10
    LHip = 11
    LKnee = 12
    LAnkle = 13
    REye = 14
    LEye = 15
    REar = 16
    LEar = 17
    Background = 18

    def __init__(self):
        self.image = None
        self.mode = "COCO"

        if self.mode == "COCO":
            self.protoFile = "../models/coco/pose_deploy_linevec.prototxt"
            self.weightsFile = "../models/coco/pose_iter_440000.caffemodel"
            self.points_number = 18
            self.POSE_PAIRS = [[1, 0], [1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [1, 8], [8, 9], [9, 10], [1, 11],
                               [11, 12], [12, 13], [0, 14], [0, 15], [14, 16], [15, 17]]
        elif self.mode == "MPI":
            self.protoFile = "../models/mpi/pose_deploy_linevec_faster_4_stages.prototxt"
            self.weightsFile = "../models/mpi/pose_iter_160000.caffemodel"
            self.points_number = 15
            self.POSE_PAIRS = [[0, 1], [1, 2], [2, 3], [3, 4], [1, 5], [5, 6], [6, 7], [1, 14], [14, 8], [8, 9],
                               [9, 10], [14, 11], [11, 12], [12, 13]]
        self.net = cv2.dnn.readNetFromCaffe(self.protoFile, self.weightsFile)

    def load_model(self):
        in_width = 368
        in_height = 368
        blob = cv2.dnn.blobFromImage(self.image, 1.0 / 255, (in_width, in_height), (0, 0, 0), swapRB=False, crop=False)
        self.net.setInput(blob)
        return self.net.forward()

    def preprocess_image(self, image):
        self.image, dim = self.resize_image(image)
        # TODO: We can perform other preprocessing on the given image
        return self.image

    def resize_image(self, image):
        r = 400.0 / image.shape[0]
        dim = (int(image.shape[1] * r), 400)
        return cv2.resize(image, dim, interpolation=cv2.INTER_AREA), dim

    def get_joint_points(self):
        model_output = self.load_model()
        # X and Y Scale
        height, width, _ = self.image.shape
        scale_x = width / model_output.shape[3]
        scale_y = height / model_output.shape[2]

        # Empty list to store the detected keypoints
        points = []

        # Threshold
        threshold = 0.1

        for i in range(self.points_number):
            # Obtain probability map
            prob_map = model_output[0, i, :, :]

            # Find global maxima of the probMap.
            _, prob, _, point = cv2.minMaxLoc(prob_map)

            # Scale the point to fit on the original image
            x = scale_x * point[0]
            y = scale_y * point[1]

            if prob > threshold:
                # Add the point to the list if the probability is greater than the threshold
                points.append((int(x), int(y)))
            else:
                points.append(None)

        return points
