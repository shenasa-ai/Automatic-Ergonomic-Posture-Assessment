import math
import numpy as np
from src.pose_detector import PoseDetector
from matplotlib import pyplot as plt
import cv2


class RosaRuleProvider:
    backrest_chair_score_matrix = []
    keyboard_mouse_score_matrix = []
    monitor_phone_score_matrix = []
    peripheral_chair_score_matrix = []

    def __init__(self, frame):
        self.image = frame
        self.circle_radius = 3
        self.line_thickness = 2
        self.incorrect_pairs = []

    def is_correct_position(self, points, counter):
        print(f'checking ROSA scores for {counter}.png ...\n')

        grand_rosa_score = 1
        chair_score = 1
        armrest_score = 1
        backrest_score = 1
        monitor_score = 1
        phone_score = 1
        mouse_score = 1

        # Chair Height & Pan Depth (3/7)
        r_hip_knee_ankle_angle = self.get_r_hip_knee_ankle_angle(points)
        l_hip_knee_ankle_angle = self.get_l_hip_knee_ankle_angle(points)
        r_hip_knee_ankle_points = [[PoseDetector.RHip, PoseDetector.RKnee],
                                   [PoseDetector.RKnee, PoseDetector.RAnkle]]

        l_hip_knee_ankle_points = [[PoseDetector.LHip, PoseDetector.LKnee],
                                   [PoseDetector.LKnee, PoseDetector.LAnkle]]

        if r_hip_knee_ankle_angle:
            if r_hip_knee_ankle_angle < 85 or r_hip_knee_ankle_angle > 95:
                chair_score = 2
                self.draw_lines_between_pairs(points, r_hip_knee_ankle_points, False)
            else:
                self.draw_lines_between_pairs(points, r_hip_knee_ankle_points)

        if l_hip_knee_ankle_angle:
            if l_hip_knee_ankle_angle > 95 or l_hip_knee_ankle_angle < 85:
                chair_score = 2
                self.draw_lines_between_pairs(points, l_hip_knee_ankle_points, False)
            else:
                self.draw_lines_between_pairs(points, l_hip_knee_ankle_points)

        # Armrest (3/4)
        shoulders_neck_angle = self.get_shoulders_neck_angle(points)
        shoulders_neck_points = [[PoseDetector.Neck, PoseDetector.RShoulder],
                                 [PoseDetector.Neck, PoseDetector.LShoulder]]
        if shoulders_neck_angle:
            if shoulders_neck_angle < 160:
                armrest_score = 2
                self.draw_lines_between_pairs(points, shoulders_neck_points, False)
            else:
                self.draw_lines_between_pairs(points, shoulders_neck_points)

        r_neck_shoulder_elbow = self.get_r_neck_shoulder_elbow_angle(points)
        r_neck_shoulder_elbow_points = [[PoseDetector.Neck, PoseDetector.RShoulder],
                                        [PoseDetector.RShoulder, PoseDetector.RElbow]]
        l_neck_shoulder_elbow = self.get_l_neck_shoulder_elbow_angle(points)
        l_neck_shoulder_elbow_points = [[PoseDetector.Neck, PoseDetector.LShoulder],
                                        [PoseDetector.LShoulder, PoseDetector.LElbow]]

        if r_neck_shoulder_elbow:
            if r_neck_shoulder_elbow > 120:
                armrest_score += 1
                self.draw_lines_between_pairs(points, r_neck_shoulder_elbow_points, False)
            else:
                self.draw_lines_between_pairs(points, r_neck_shoulder_elbow_points)

        if l_neck_shoulder_elbow:
            if l_neck_shoulder_elbow > 120:
                armrest_score += 1
                self.draw_lines_between_pairs(points, l_neck_shoulder_elbow_points, False)
            else:
                self.draw_lines_between_pairs(points, l_neck_shoulder_elbow_points)

        # Backrest (3/5)
        r_shoulder_hip_knee = self.get_r_shoulder_hip_knee_angle(points)
        r_shoulder_hip_knee_points = [[PoseDetector.RShoulder, PoseDetector.RHip],
                                      [PoseDetector.RHip, PoseDetector.RKnee]]
        l_shoulder_hip_knee = self.get_l_shoulder_hip_knee_angle(points)
        l_shoulder_hip_knee_points = [[PoseDetector.LShoulder, PoseDetector.LHip],
                                      [PoseDetector.LHip, PoseDetector.LKnee]]
        if r_shoulder_hip_knee:
            if r_shoulder_hip_knee > 110 or r_shoulder_hip_knee < 95:
                backrest_score = 2
                self.draw_lines_between_pairs(points, r_shoulder_hip_knee_points, False)
            else:
                self.draw_lines_between_pairs(points, r_shoulder_hip_knee_points)

        if l_shoulder_hip_knee:
            if l_shoulder_hip_knee > 110 or l_shoulder_hip_knee < 95:
                backrest_score = 2
                self.draw_lines_between_pairs(points, l_shoulder_hip_knee_points, False)
            else:
                self.draw_lines_between_pairs(points, l_shoulder_hip_knee_points)

        if shoulders_neck_angle:
            if shoulders_neck_angle < 160:
                backrest_score += 1

        # Monitor (0/6)
        # TODO: if we add image segmentation then we can use this rules

        # Telephone (1/3)

        shoulders_distance = None
        wrists_distance = None

        if points[PoseDetector.RShoulder] and points[PoseDetector.LShoulder]:
            shoulders_distance = self.calculate_distance_between_two_points(points[PoseDetector.RShoulder],
                                                                            points[PoseDetector.LShoulder])

        if points[PoseDetector.RWrist] and points[PoseDetector.LWrist]:
            wrists_distance = self.calculate_distance_between_two_points(points[PoseDetector.RWrist],
                                                                         points[PoseDetector.LWrist])
        if shoulders_distance and wrists_distance and r_neck_shoulder_elbow and l_neck_shoulder_elbow:
            if (math.fabs(wrists_distance - shoulders_distance) > shoulders_distance / 3) and \
                    math.fabs(r_neck_shoulder_elbow - l_neck_shoulder_elbow > 30):
                phone_score = 2
                # TODO: we do not know which side is longer

        # Mouse and Keyboard
        r_shoulder_wrist_angle = None
        l_shoulder_wrist_angle = None
        r_wrist_l_wrist_angle = None
        r_eye_l_eye_angle = None
        if points[PoseDetector.RWrist] and points[PoseDetector.RShoulder]:
            r_shoulder_wrist_angle = self.get_angle_between_vector_and_vertical_axis(
                np.array(points[PoseDetector.RWrist]) - np.array(points[PoseDetector.RShoulder]))
            print(f'angle between right shoulder_wrist and vertical axis: {r_shoulder_wrist_angle}')

        if points[PoseDetector.LWrist] and points[PoseDetector.LShoulder]:
            l_shoulder_wrist_angle = self.get_angle_between_vector_and_vertical_axis(
                np.array(points[PoseDetector.LWrist]) - np.array(points[PoseDetector.LShoulder]))
            print(f'angle between left shoulder_wrist and vertical axis: {l_shoulder_wrist_angle}')

        if points[PoseDetector.RWrist] and points[PoseDetector.LWrist]:
            r_wrist_l_wrist_angle = self.get_angle_between_vector_and_horizontal_axis(
                np.array(points[PoseDetector.RWrist]) - np.array(points[PoseDetector.LWrist]))
            print(f'angle between two wrists and horizontal axis: {r_wrist_l_wrist_angle}')

        if points[PoseDetector.REye] and points[PoseDetector.LEye]:
            r_eye_l_eye_angle = self.get_angle_between_vector_and_horizontal_axis(
                np.array(points[PoseDetector.REye]) - np.array(points[PoseDetector.LEye]))
            print(f'angle between two eyes and horizontal axis: {r_eye_l_eye_angle}')

        if (r_shoulder_wrist_angle and l_shoulder_wrist_angle) \
                and (r_shoulder_wrist_angle > 20 or l_shoulder_wrist_angle > 20):
            mouse_score = 2
        if r_wrist_l_wrist_angle and (r_wrist_l_wrist_angle < 160 or r_wrist_l_wrist_angle > 220):
            mouse_score += 2

        # neck rule side
        r_hip_shoulder_ear_angle = self.get_r_hip_shoulder_ear_angle(points)
        l_hip_shoulder_ear_angle = self.get_l_hip_shoulder_ear_angle(points)

        r_ear_eye_shoulder_angle = self.get_r_ear_eye_shoulder_angle(points)
        l_ear_eye_shoulder_angle = self.get_l_ear_eye_shoulder_angle(points)

        # neck rule front
        if points[PoseDetector.LEye] and points[PoseDetector.LEye] and points[PoseDetector.LShoulder] \
                and points[PoseDetector.RShoulder]:
            v1 = np.array(points[PoseDetector.LEye]) - np.array(points[PoseDetector.REye])
            v2 = np.array(points[PoseDetector.LShoulder]) - np.array(points[PoseDetector.RShoulder])
            angle_between_shoulders_and_eyes = self.get_angle_between_lines(v1, v2)

        # neck rule twist side
        neck_twisted_status_from_side = self.is_neck_twisted_from_side(points)

        # neck rule twist front

        print(f'chair score is: {chair_score}\n'
              f'armrest score is: {armrest_score}\n'
              f'backrest score is: {backrest_score}\n'
              f'monitor score is: {monitor_score}\n'
              f'phone score is: {phone_score}\n'
              f'mouse score is: {mouse_score}')

        return grand_rosa_score
        print('*******************************************************************************************************')

    # TODO: we can calculate avg of neighboring points as a solution for occlusion
    def get_r_hip_knee_ankle_angle(self, points):
        # rHipKneeAnkle_pairs = [[8, 9], [9, 10]]
        angle = self.get_angle_between_points(points[PoseDetector.RHip], points[PoseDetector.RKnee],
                                              points[PoseDetector.RAnkle])
        print(f'right hip_knee_ankle angle: {angle}')
        return angle

    def get_l_hip_knee_ankle_angle(self, points):
        # lHipKneeAnkle_pairs = [[11, 12], [12, 13]]
        angle = self.get_angle_between_points(points[PoseDetector.LHip], points[PoseDetector.LKnee],
                                              points[PoseDetector.LAnkle])
        print(f'left hip_knee_ankle angle: {angle}')
        return angle

    def get_shoulders_neck_angle(self, points):
        # shoulders_neck_pairs = [[2, 1], [1, 5]]
        angle = self.get_angle_between_points(points[PoseDetector.RShoulder], points[PoseDetector.Neck],
                                              points[PoseDetector.LShoulder])
        if angle:
            print(f'shoulders_neck angle: {angle}')
        return angle

    def get_r_neck_shoulder_elbow_angle(self, points):
        # rNeckShoulderElbow_pairs = [[1, 2], [2, 3]]
        angle = self.get_angle_between_points(points[PoseDetector.Neck], points[PoseDetector.RShoulder],
                                              points[PoseDetector.RElbow])
        print(f'right neck_shoulder_elbow angle: {angle}')
        return angle

    def get_l_neck_shoulder_elbow_angle(self, points):
        # lNeckShoulderElbow_pairs = [[1, 5], [5, 6]]
        angle = self.get_angle_between_points(points[PoseDetector.Neck], points[PoseDetector.LShoulder],
                                              points[PoseDetector.LElbow])
        print(f'left neck_shoulder_elbow angle: {angle}')
        return angle

    def get_r_shoulder_hip_knee_angle(self, points):
        # rShoulderHipKnee_pairs = [[2, 8], [8, 9]]
        angle = self.get_angle_between_points(points[PoseDetector.RShoulder], points[PoseDetector.RHip],
                                              points[PoseDetector.RKnee])
        print(f'right shoulder_hip_knee angle: {angle}')
        return angle

    def get_l_shoulder_hip_knee_angle(self, points):
        # lShoulderHipKnee_pairs = [[5, 11], [11, 12]]
        angle = self.get_angle_between_points(points[PoseDetector.LShoulder], points[PoseDetector.LHip],
                                              points[PoseDetector.LKnee])
        print(f'left shoulder_hip_knee angle: {angle}')
        return angle

    def get_r_shoulder_elbow_wrist(self, points):
        # r_shoulder_elbow_wrist_pairs = [[2, 3], [3, 4]]
        angle = self.get_angle_between_points(points[PoseDetector.RShoulder],
                                              points[PoseDetector.RElbow], points[PoseDetector.RWrist])
        print(f'right shoulder_elbow_wrist angle: {angle}')
        return angle

    def get_l_shoulder_elbow_wrist(self, points):
        # l_shoulder_elbow_wrist_pairs = [[5, 6], [6, 7]]
        angle = self.get_angle_between_points(points[PoseDetector.LShoulder],
                                              points[PoseDetector.LElbow], points[PoseDetector.LWrist])
        print(f'left shoulder_elbow_wrist angle: {angle}')
        return angle

    def get_back_angle(self, points):
        pass

    def get_r_hip_shoulder_elbow_angle(self, points):
        # rHipShoulderElbow_pairs = [[3, 2], [2, 8]]
        angle = self.get_angle_between_points(points[PoseDetector.RElbow],
                                              points[PoseDetector.RShoulder], points[PoseDetector.RHip])
        print(f'right hip_shoulder_elbow angle: {angle}')
        return angle

    def get_l_hip_shoulder_elbow_angle(self, points):
        # lHipShoulderElbow_pairs = [[6, 5], [5, 11]]
        angle = self.get_angle_between_points(points[PoseDetector.LElbow],
                                              points[PoseDetector.LShoulder], points[PoseDetector.LHip])
        print(f'left hip_shoulder_elbow angle: {angle}')
        return angle

    def get_r_shoulder_elbow_wrist_angle(self, points):
        # rShoulderElbowWrist_pairs = [[2, 3], [3, 4]]
        angle = self.get_angle_between_points(points[PoseDetector.RShoulder],
                                              points[PoseDetector.RElbow], points[PoseDetector.RWrist])
        print(f'right shoulder_elbow_wrist angle: {angle}')
        return angle

    def get_l_shoulder_elbow_wrist_angle(self, points):
        # lShoulderElbowWrist_pairs = [[5, 6], [6, 7]]
        angle = self.get_angle_between_points(points[PoseDetector.LShoulder],
                                              points[PoseDetector.LElbow], points[PoseDetector.LWrist])
        print(f'left shoulder_elbow_wrist angle: {angle}')
        return angle

    # neck rules
    def is_neck_twisted_from_side(self, points):
        if points[PoseDetector.RShoulder] and points[PoseDetector.LShoulder]:
            if points[PoseDetector.REar] and points[PoseDetector.LEar] and points[PoseDetector.Nose]:
                return 0
            else:
                return 1

        elif points[PoseDetector.RShoulder] and points[PoseDetector.LShoulder] is None:
            if points[PoseDetector.REar] and points[PoseDetector.LEar] is None:
                return 0
            else:
                return 1

        elif points[PoseDetector.RShoulder] is None and points[PoseDetector.LShoulder]:
            if points[PoseDetector.REar] is None and points[PoseDetector.LEar]:
                return 0
            else:
                return 1

    def neck_rule_twist_side_front(self, points):
        if points[PoseDetector.REar] and points[PoseDetector.LEar] and points[PoseDetector.Nose]:
            return 0

        elif points[PoseDetector.REye] and points[PoseDetector.LEye] and points[PoseDetector.Nose]:
            r_ear_nose_distance = self.calculate_distance_between_two_points(points[PoseDetector.REye],
                                                                             points[PoseDetector.Nose])
            l_ear_nose_distance = self.calculate_distance_between_two_points(points[PoseDetector.LEye],
                                                                             points[PoseDetector.Nose])
            ratio = r_ear_nose_distance / l_ear_nose_distance
            if ratio < 1:
                if ratio < 0.8:
                    print(f'{ratio} < 0.8')
                    print(f'twisted')
                    return 1
                else:
                    print(f'{ratio} > 0.8')
                    print(f'not twisted')

            else:
                if ratio > 1.25:
                    print(f' {ratio} > 1.25')
                    print(f' twisted')
                else:
                    print(f'{ratio} < 1.25')
                    print(f'not twisted')
            #     return 0
            # else:
            #     plt.text(100, 200, f'r = {ratio} < 0.8', size=15)

        elif (points[PoseDetector.REye] or points[PoseDetector.LEye]) and (points[PoseDetector.Nose]):
            print('only one eye is seen')
            return 1

    def get_r_hip_shoulder_ear_angle(self, points):
        # r_hip_shoulder_ear_pairs = [[8, 2], [2, 16]]
        angle = self.get_angle_between_points(points[PoseDetector.RHip],
                                              points[PoseDetector.RShoulder], points[PoseDetector.REar])
        print(f'right hip_shoulder_ear angle: {angle}')
        return angle

    def get_l_hip_shoulder_ear_angle(self, points):
        # l_hip_shoulder_ear_pairs = [[11, 5], [5, 17]]
        angle = self.get_angle_between_points(points[PoseDetector.LHip],
                                              points[PoseDetector.LShoulder], points[PoseDetector.LEar])
        print(f'left hip_shoulder_ear angle: {angle}')
        return angle

    def get_r_ear_eye_shoulder_angle(self, points):
        # l_ear_eye_shoulder_pairs = [[16, 14], [14, 2]]
        angle = self.get_angle_between_points(points[PoseDetector.REye],
                                              points[PoseDetector.REar], points[PoseDetector.RShoulder])
        print(f'right ear_eye_shoulder angle: {angle}')
        return angle

    def get_l_ear_eye_shoulder_angle(self, points):
        # l_ear_eye_shoulder_pairs = [[17, 15], [15, 5]]
        angle = self.get_angle_between_points(points[PoseDetector.LEye],
                                              points[PoseDetector.LEar], points[PoseDetector.LShoulder])
        print(f'left ear_eye_shoulder angle: {angle}')
        return angle

    def get_angle_between_lines(self, v1, v2):
        len_v1 = math.sqrt(v1[0] ** 2 + v1[1] ** 2)
        len_v2 = math.sqrt(v2[0] ** 2 + v2[1] ** 2)
        if len_v1 == 0 or len_v2 == 0:
            return None
        angle = math.acos(round(np.dot(v1, v2) / (len_v1 * len_v2), 5)) * 180 / math.pi
        if math.isnan(angle):
            return None

        return angle

    def calculate_distance_between_two_points(self, p1, p2):
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    def get_angle_between_vector_and_horizontal_axis(self, v1):
        v2 = np.array([1, 0])
        len_v1 = math.sqrt(v1[0] ** 2 + v1[1] ** 2)
        len_v2 = math.sqrt(v2[0] ** 2 + v2[1] ** 2)
        if len_v1 == 0:
            return None
        return math.acos(round(np.dot(v1, v2) / (len_v1 * len_v2), 5)) * 180 / math.pi

    def get_angle_between_vector_and_vertical_axis(self, v1):
        v2 = np.array([0, 1])
        len_v1 = math.sqrt(v1[0] ** 2 + v1[1] ** 2)
        len_v2 = math.sqrt(v2[0] ** 2 + v2[1] ** 2)
        if len_v1 == 0:
            return None
        return math.acos(round(np.dot(v1, v2) / (len_v1 * len_v2), 5)) * 180 / math.pi

    def get_vectors_between_points(self, p1, p2, p3, p4):
        v1 = np.array(p2) - np.array(p1)
        v2 = np.array(p4) - np.array(p3)
        return v1, v2

    def get_angle_between_points(self, p1, p2, p3):
        if p1 and p2 and p3:
            v1, v2 = self.get_vectors_between_points(p2, p1, p2, p3)
            angle = self.get_angle_between_lines(v1, v2)
            return angle
        else:
            return None

    def draw_lines_between_pairs(self, points, pairs, is_correct_edge=True):
        for pair in pairs:
            part_a = pair[0]
            part_b = pair[1]

            if points[part_a] and points[part_b]:
                if is_correct_edge and pair not in self.incorrect_pairs:
                    cv2.line(self.image, points[part_a], points[part_b], (0, 255, 255), self.line_thickness)
                else:
                    cv2.line(self.image, points[part_a], points[part_b], (0, 0, 255), self.line_thickness)
                    self.incorrect_pairs.append(pair)

    def save_image(self, counter, is_position_correct):
        img = np.array(self.image)[:, :, ::-1]
        plt.imshow(img)
        plt.axis('off')
        if is_position_correct:
            plt.text(10, 30, 'Correct', color='green', fontsize=16)
        else:
            plt.text(10, 40, 'Incorrect', color='red', fontsize=16)

        plt.savefig(f'../output/{counter}.png')

    def display_joint_points(self, points):
        for point in points:
            if point:
                cv2.circle(self.image, point, self.circle_radius, (0, 255, 0), thickness=-1,
                           lineType=cv2.FILLED)
