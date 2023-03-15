import mediapipe as mp

from pose_detector import PoseDetector
import cv2


class MediapipePoseDetector(PoseDetector):
    Nose = 0
    left_eye_inner = 1
    LEye = 2
    left_eye_outer = 3
    right_eye_inner = 4
    REye = 5
    right_eye_outer = 6
    LEar = 7
    REar = 8
    mouth_left = 9
    mouth_right = 10
    LShoulder = 11
    RShoulder = 12
    LElbow = 13
    RElbow = 14
    LWrist = 15 
    RWrist = 16
    left_pinky = 17
    right_pinky = 18
    left_index = 19
    right_index = 20
    left_thumb = 21
    right_thumb = 22
    LHip = 23
    RHip = 24
    LKnee = 25
    RKnee = 26
    LAnkle = 27
    RAnkle = 28
    left_heel = 29
    right_heel = 30
    left_foot_index = 31
    right_foot_index = 32

    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        # self.mp_drawing_styles = mp.solutions.drawing_styles

        self.keypoints = ['nose', 'left_eye_inner', 'left_eye', 'left_eye_outer', 'right_eye_inner', 'right_eye', 'right_eye_outer',
            'left_ear', 'right_ear', 'mouth_left', 'mouth_right', 'left_shoulder', 'right_shoulder', 'left_elbow',
            'right_elbow', 'left_wrist', 'right_wrist', 'left_pinky', 'right_pinky', 'left_index', 'right_index',
            'left_thumb', 'right_thumb', 'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle',
            'left_heel', 'right_heel', 'left_foot_index', 'right_foot_index']

    def preprocess_image(self, image):
        self.image, dim = self.resize_image(image)
        # TODO: We can perform other preprocessing on the given image
        return self.image

    def resize_image(self, image):
        r = 400.0 / image.shape[0]
        dim = (int(image.shape[1] * r), 400)
        return cv2.resize(image, dim, interpolation=cv2.INTER_AREA), dim

         
    def get_joint_points(self):
        with self.mp_pose.Pose(static_image_mode=True, 
                model_complexity=1,
                min_detection_confidence=0.3) as pose:
            frame = self.image
            results = pose.process(frame)
            image_hight, image_width, _ = frame.shape
            xx = []
            yy = []
            zz = []
            joints = []
            joints_3d = []
            if not results.pose_landmarks:
                print('Not detected')
                return [None] * 33
               
            else:
                for i in range(len(results.pose_landmarks.landmark)):
                    x = (int(results.pose_landmarks.landmark[i].x * image_width))
                    y = (int(results.pose_landmarks.landmark[i].y * image_hight))   
                    z = (int(results.pose_landmarks.landmark[i].z * image_hight)) 
                    prob = results.pose_landmarks.landmark[i].visibility
                    if prob > 0.1: # th
                        joints.append((x, y))
                        joints_3d.append((x, y, z))
                    else:
                        joints.append(None)
                        joints_3d.append(None)
                    #xx.append(x)
                    #yy.append(y)
                    #zz.append(z)
                    #print(joints)
                return joints


