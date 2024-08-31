import cv2
from ultralytics import YOLO

from pose_detector import PoseDetector


def resize_image(image):
    r = 400.0 / image.shape[0]
    dim = (int(image.shape[1] * r), 400)
    return cv2.resize(image, dim, interpolation=cv2.INTER_AREA), dim


class YoloPoseDetector(PoseDetector):
    Nose = 0
    LEye = 1
    REye = 2
    LEar = 3
    REar = 4
    LShoulder = 5
    RShoulder = 6
    LElbow = 7
    RElbow = 8
    LWrist = 9
    RWrist = 10
    LHip = 11
    RHip = 12
    LKnee = 13
    RKnee = 14
    LAnkle = 15
    RAnkle = 16

    def __init__(self):
        self.image = None
        self.model = 'yolov8n-pose.pt'
        self.points_number = 17
        self.POSE_PAIRS = [[1, 0], [1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [1, 8], [8, 9], [9, 10], [1, 11],
                           [11, 12], [12, 13], [0, 14], [0, 15], [14, 16], [15, 17]]  # based on COCO

    def preprocess_image(self, image):
        self.image, dim = resize_image(image)
        return self.image

    def get_joint_points(self) -> []:
        model = YOLO(self.model)
        result = model.predict(self.image)
        keypoints = result[0].keypoints.xy.cpu().numpy()[0]
        points = []
        for x, y in keypoints:
            points.append((int(x), int(y)))
        return points