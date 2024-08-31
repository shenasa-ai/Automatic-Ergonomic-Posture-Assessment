from pose_detector import PoseDetector
import openpifpaf
import cv2


def resize_image(image):
    r = 400.0 / image.shape[0]
    dim = (int(image.shape[1] * r), 400)
    return cv2.resize(image, dim, interpolation=cv2.INTER_AREA), dim


class OpenpifpafPoseDetector(PoseDetector):
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
    Background = 17

    def __init__(self):
        self.image = None
        self.check_point = 'shufflenetv2k30'  # also it can be Resnet50 or etc
        self.points_number = 17
        self.POSE_PAIRS = [[1, 0], [1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [1, 8], [8, 9], [9, 10], [1, 11],
                           [11, 12], [12, 13], [0, 14], [0, 15], [14, 16], [15, 17]]  # based on COCO

    def preprocess_image(self, image):
        self.image, dim = resize_image(image)
        return self.image

    def get_joint_points(self) -> []:
        # Predict joints
        predictor = openpifpaf.Predictor(checkpoint='shufflenetv2k30')
        predictions, gt_anns, image_meta = predictor.numpy_image(self.image)
        # save joints
        points = []
        for person in range(len(predictions)):
            for i in range(self.points_number):
                x = predictions[person].data[i][0]
                y = predictions[person].data[i][1]
                points.append((int(x), int(y)))
        return points
