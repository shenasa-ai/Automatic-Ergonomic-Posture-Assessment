import math
import os
import numpy as np
from pose_detector import PoseDetector
from matplotlib import pyplot as plt
from face_blurring import FaceBlurring
import cv2


class RosaRuleProvider:
    prediction = {
        'image_number': [],
        'chair': [],
        'armrest': [],
        'back': [],
        'monitor': [],
        'phone': []
    }

    def __init__(self, pose_detector):
        self.image = None
        self.points = None
        self.pose_detector = pose_detector
        self.circle_radius = 3
        self.line_thickness = 2
        self.description = ""
        self.repetitive_pairs = []
        self.camera_view_point = "front"

    def get_posture_status(self, image, points, file_name, view_point, draw_joint_points=True):
        self.prediction['image_number'].append(file_name.split(sep='.')[0].split(sep='_')[1])
        self.description = ""
        self.image = image
        self.points = points
        self.camera_view_point = view_point
        posture_status = False
        chair_score = 1
        armrest_score = 1
        backrest_score = 1
        monitor_score = 1
        phone_score = 1
        mouse_score = 1

        if draw_joint_points:
            self.display_joint_points()

        print(f'ROSA score is checking for {file_name} ...\n')

        # Chair Height & Pan Depth (3/7)
        chair_score = self.get_chair_score()
        self.prediction['chair'].append(chair_score)

        # Armrest (3/4)
        armrest_score = self.get_armrest_score()
        self.prediction['armrest'].append(armrest_score)

        # Backrest (3/5)
        backrest_score = self.get_backrest_score()
        self.prediction['back'].append(backrest_score)

        # Monitor (0/6)
        monitor_score = self.get_monitor_score()
        self.prediction['monitor'].append(monitor_score)

        # Telephone (1/3)
        phone_score = self.get_phone_score()
        self.prediction['phone'].append(phone_score)

        if chair_score > 1 or armrest_score > 1 or backrest_score > 1 or monitor_score > 1 or phone_score > 1:
            posture_status = False
        else:
            posture_status = True

        print(f'chair score is: {chair_score}\n'
              f'armrest score is: {armrest_score}\n'
              f'backrest score is: {backrest_score}\n'
              f'monitor score is: {monitor_score}\n'
              f'phone score is: {phone_score}\n'
              f'mouse score is: {mouse_score}\n')

        return posture_status

    def get_chair_score(self):
        chair_score = 1
        if self.camera_view_point == "side":
            r_hip_knee_ankle_angle = self.get_r_hip_knee_ankle_angle()
            l_hip_knee_ankle_angle = self.get_l_hip_knee_ankle_angle()
            r_hip_knee_ankle_points = [[self.pose_detector.RHip, self.pose_detector.RKnee],
                                       [self.pose_detector.RKnee, self.pose_detector.RAnkle]]
            l_hip_knee_ankle_points = [[self.pose_detector.LHip, self.pose_detector.LKnee],
                                       [self.pose_detector.LKnee, self.pose_detector.LAnkle]]

            if r_hip_knee_ankle_angle:
                healthy = False
                if r_hip_knee_ankle_angle < 80:
                    chair_score = 2
                    self.draw_lines_between_pairs(r_hip_knee_ankle_points, False)
                    self.description = self.description + f'Chair is TOO LOW - right hip_knee_ankle angle: {r_hip_knee_ankle_angle}\n'
                elif r_hip_knee_ankle_angle > 100:
                    chair_score = 2
                    self.draw_lines_between_pairs(r_hip_knee_ankle_points, False)
                    self.description = self.description + f'Chair is TOO HIGH - right hip_knee_ankle angle: {r_hip_knee_ankle_angle}\n'
                else:
                    healthy = True
                    self.draw_lines_between_pairs(r_hip_knee_ankle_points)
                    self.description = self.description + \
                                       f'Right knee status is in CORRECT POSTURE - right hip_knee_ankle angle: {r_hip_knee_ankle_angle}\n'
                self.draw_angles(r_hip_knee_ankle_points, r_hip_knee_ankle_angle, healthy)

            if l_hip_knee_ankle_angle:
                healthy = False
                if l_hip_knee_ankle_angle < 80:
                    chair_score = 2
                    self.draw_lines_between_pairs(l_hip_knee_ankle_points, False)
                    self.description = self.description + f'Chair is TOO LOW - left hip_knee_ankle angle: {l_hip_knee_ankle_angle}\n'
                elif l_hip_knee_ankle_angle > 100:
                    chair_score = 2
                    self.draw_lines_between_pairs(l_hip_knee_ankle_points, False)
                    self.description = self.description + f'Chair is TOO HIGH - left hip_knee_ankle angle: {l_hip_knee_ankle_angle}\n'
                else:
                    healthy = True
                    self.draw_lines_between_pairs(l_hip_knee_ankle_points)
                    self.description = self.description + \
                                       f'Left knee status is in CORRECT POSTURE - left hip_knee_ankle angle: {l_hip_knee_ankle_angle}\n'
                self.draw_angles(l_hip_knee_ankle_points, l_hip_knee_ankle_angle, healthy)
        return chair_score

    def get_armrest_score(self):
        armrest_score = 1
        if self.camera_view_point == "front":
            shoulders_neck_angle = self.get_shoulders_neck_angle()
            shoulders_neck_points = [[self.pose_detector.Neck, self.pose_detector.RShoulder],
                                     [self.pose_detector.Neck, self.pose_detector.LShoulder]]
            if shoulders_neck_angle:
                healthy = False
                if shoulders_neck_angle < 160:
                    armrest_score = 2
                    self.draw_lines_between_pairs(shoulders_neck_points, False)
                    self.description = self.description + f'Shoulders are SHRUGGED - shoulders_neck angle: {shoulders_neck_angle} \n'
                else:
                    healthy = True
                    self.draw_lines_between_pairs(shoulders_neck_points)
                    self.description = self.description + f'Shoulders are in NORMAL POSTURE - shoulders_neck angle: {shoulders_neck_angle} \n'
                self.draw_angles(shoulders_neck_points, shoulders_neck_angle, healthy)

            r_shoulder_elbow_angle = None
            r_shoulder_elbow_points = [[self.pose_detector.RShoulder, self.pose_detector.RElbow]]

            l_shoulder_elbow_angle = None
            l_shoulder_elbow_points = [[self.pose_detector.LShoulder, self.pose_detector.LElbow]]

            if self.points[self.pose_detector.RShoulder] and self.points[self.pose_detector.RElbow]:
                r_shoulder_elbow_angle = self.get_angle_between_vector_and_vertical_axis(
                    np.array(self.points[self.pose_detector.RElbow]) - np.array(
                        self.points[self.pose_detector.RShoulder]))

            if self.points[self.pose_detector.LShoulder] and self.points[self.pose_detector.LElbow]:
                l_shoulder_elbow_angle = self.get_angle_between_vector_and_vertical_axis(
                    np.array(self.points[self.pose_detector.LElbow]) - np.array(
                        self.points[self.pose_detector.LShoulder]))

            if r_shoulder_elbow_angle:
                # healthy = False
                if r_shoulder_elbow_angle > 20:
                    armrest_score += 1
                    self.draw_lines_between_pairs(r_shoulder_elbow_points, False)
                    self.description = self.description + f'Right elbow is NOT INLINE with right shoulder - ' \
                                                          f'Angle between right shoulder_elbow and vertical axis: ' \
                                                          f'{r_shoulder_elbow_angle}\n'
                else:
                    # healthy = True
                    self.draw_lines_between_pairs(r_shoulder_elbow_points)
                    self.description = self.description + f'Right elbow is INLINE with right shoulder - ' \
                                                          f'Angle between right shoulder_elbow and vertical axis: ' \
                                                          f'{r_shoulder_elbow_angle}\n'
                # self.draw_angles(r_shoulder_elbow_points, r_shoulder_elbow_angle, healthy)

            if l_shoulder_elbow_angle:
                # healthy = False
                if l_shoulder_elbow_angle > 20:
                    armrest_score += 1
                    self.draw_lines_between_pairs(l_shoulder_elbow_points, False)
                    self.description = self.description + f'Left elbow is NOT INLINE with left shoulder - ' \
                                                          f'Angle between left shoulder_elbow and vertical axis:' \
                                                          f' {l_shoulder_elbow_angle}\n'
                else:
                    # healthy = True
                    self.draw_lines_between_pairs(l_shoulder_elbow_points)
                    self.description = self.description + f'Left elbow is INLINE with left shoulder - ' \
                                                          f'Angle between left shoulder_elbow and vertical axis: ' \
                                                          f'{l_shoulder_elbow_angle}\n'
                # self.draw_angles(l_shoulder_elbow_points, l_shoulder_elbow_angle, healthy)

            r_neck_shoulder_elbow = self.get_r_neck_shoulder_elbow_angle()
            r_neck_shoulder_elbow_points = [[self.pose_detector.Neck, self.pose_detector.RShoulder],
                                            [self.pose_detector.RShoulder, self.pose_detector.RElbow]]
            l_neck_shoulder_elbow = self.get_l_neck_shoulder_elbow_angle()
            l_neck_shoulder_elbow_points = [[self.pose_detector.Neck, self.pose_detector.LShoulder],
                                            [self.pose_detector.LShoulder, self.pose_detector.LElbow]]

            if r_neck_shoulder_elbow:
                healthy = False
                if r_neck_shoulder_elbow > 120:
                    armrest_score += 1
                    self.draw_lines_between_pairs(r_neck_shoulder_elbow_points, False)
                    self.description = self.description + f'TOO WIDE right elbow - ' \
                                                          f'right neck_shoulder_elbow angle: {r_neck_shoulder_elbow}\n'
                else:
                    healthy = True
                    self.draw_lines_between_pairs(r_neck_shoulder_elbow_points)
                    self.description = self.description + f'Right elbow is NOT TOO WIDE - ' \
                                                          f'right neck_shoulder_elbow angle: {r_neck_shoulder_elbow}\n'
                self.draw_angles(r_neck_shoulder_elbow_points, r_neck_shoulder_elbow, healthy)

            if l_neck_shoulder_elbow:
                healthy = False
                if l_neck_shoulder_elbow > 120:
                    armrest_score += 1
                    self.draw_lines_between_pairs(l_neck_shoulder_elbow_points, False)
                    self.description = self.description + f'TOO WIDE left elbow - ' \
                                                          f'left neck_shoulder_elbow angle: {l_neck_shoulder_elbow}\n'
                else:
                    healthy = True
                    self.draw_lines_between_pairs(l_neck_shoulder_elbow_points)
                    self.description = self.description + f'Left elbow is NOT TOO WIDE - ' \
                                                          f'left neck_shoulder_elbow angle: {l_neck_shoulder_elbow}\n'
                self.draw_angles(l_neck_shoulder_elbow_points, l_neck_shoulder_elbow, healthy)

        return armrest_score

    def get_backrest_score(self):
        backrest_score = 1
        if self.camera_view_point == "side":
            # *r_shoulder_hip_knee was commented for threshold optimizing*
            # r_shoulder_hip_knee = self.get_r_shoulder_hip_knee_angle()
            r_shoulder_hip_knee = None
            r_shoulder_hip_knee_points = [[self.pose_detector.RShoulder, self.pose_detector.RHip],
                                          [self.pose_detector.RHip, self.pose_detector.RKnee]]
            l_shoulder_hip_knee = self.get_l_shoulder_hip_knee_angle()
            l_shoulder_hip_knee_points = [[self.pose_detector.LShoulder, self.pose_detector.LHip],
                                          [self.pose_detector.LHip, self.pose_detector.LKnee]]
            if r_shoulder_hip_knee:
                healthy = False
                if r_shoulder_hip_knee < 95:
                    backrest_score = 2
                    self.draw_lines_between_pairs(r_shoulder_hip_knee_points, False)
                    self.description = self.description + f'Back rest is BENT FORWARD from right side - ' \
                                                          f'right shoulder_hip_knee angle: {r_shoulder_hip_knee}\n'
                elif r_shoulder_hip_knee > 110:
                    backrest_score = 2
                    self.draw_lines_between_pairs(r_shoulder_hip_knee_points, False)
                    self.description = self.description + f'Back rest is BENT BACKWARD from right side - ' \
                                                          f'right shoulder_hip_knee angle: {r_shoulder_hip_knee}\n'
                else:
                    healthy = True
                    self.draw_lines_between_pairs(r_shoulder_hip_knee_points)
                    self.description = self.description + f'Back rest is NORMAL from right side - ' \
                                                          f'right shoulder_hip_knee angle: {r_shoulder_hip_knee}\n'
                self.draw_angles(r_shoulder_hip_knee_points, r_shoulder_hip_knee, healthy)

            if l_shoulder_hip_knee:
                healthy = False
                if l_shoulder_hip_knee < 88:
                    backrest_score = 2
                    self.draw_lines_between_pairs(l_shoulder_hip_knee_points, False)
                    self.description = self.description + f'Back rest is BENT FORWARD from left side - ' \
                                                          f'left shoulder_hip_knee angle: {l_shoulder_hip_knee}\n'
                elif l_shoulder_hip_knee > 106:
                    backrest_score = 2
                    self.draw_lines_between_pairs(l_shoulder_hip_knee_points, False)
                    self.description = self.description + f'Back rest is BENT BACKWARD from left side - ' \
                                                          f'left shoulder_hip_knee angle: {l_shoulder_hip_knee}\n'
                else:
                    if self.check_back_validation():
                        healthy = True
                        self.description = self.description + f'Back rest is NORMAL from left side - ' \
                                                              f'left shoulder_hip_knee angle: {l_shoulder_hip_knee}\n'
                    else:
                        healthy = False
                        backrest_score = 2
                        self.description = self.description + f'Back rest is BENT FORWARD from left side OR Back rest is BENT BACKWARD from left side - ' \
                                                              f'left shoulder_hip_knee angle: {l_shoulder_hip_knee}\n'
                    self.draw_lines_between_pairs(l_shoulder_hip_knee_points)

                self.draw_angles(l_shoulder_hip_knee_points, l_shoulder_hip_knee, healthy)

        if self.camera_view_point == "front":
            shoulders_neck_angle = self.get_shoulders_neck_angle()
            if shoulders_neck_angle:
                if shoulders_neck_angle < 160:
                    backrest_score += 1
        return backrest_score

    def check_back_validation(self):
        virtual_joint = (self.points[self.pose_detector.LHip][0],
                         self.points[self.pose_detector.LShoulder][1])  # x comes from hip and y comes from shoulder
        angle = self.get_angle_between_points(self.points[self.pose_detector.LShoulder],
                                              self.points[self.pose_detector.LHip], virtual_joint)
        if angle:
            if angle > 8:
                return False
        return True

    def get_monitor_score(self):
        monitor_score = 1
        if self.camera_view_point == "side":
            r_hip_shoulder_ear_angle = self.get_r_hip_shoulder_ear_angle()
            r_hip_shoulder_ear_points = [[self.pose_detector.RHip, self.pose_detector.RShoulder],
                                         [self.pose_detector.RShoulder, self.pose_detector.REar]]
            l_hip_shoulder_ear_angle = self.get_l_hip_shoulder_ear_angle()
            l_hip_shoulder_ear_points = [[self.pose_detector.LHip, self.pose_detector.LShoulder],
                                         [self.pose_detector.LShoulder, self.pose_detector.LEar]]

            if r_hip_shoulder_ear_angle:
                healthy = False
                if r_hip_shoulder_ear_angle < 140:
                    monitor_score += 1
                    self.draw_lines_between_pairs(r_hip_shoulder_ear_points, False)
                    self.description = self.description + f'Neck is BENT FORWARD from right side view - ' \
                                                          f'right hip_shoulder_ear angle: {r_hip_shoulder_ear_angle}\n'
                elif r_hip_shoulder_ear_angle > 200:
                    monitor_score += 3
                    self.draw_lines_between_pairs(r_hip_shoulder_ear_points, False)
                    self.description = self.description + f'Neck is BENT BACKWARD from right side view - ' \
                                                          f'right hip_shoulder_ear angle: {r_hip_shoulder_ear_angle}\n'
                else:
                    healthy = True
                    self.draw_lines_between_pairs(r_hip_shoulder_ear_points)
                    self.description = self.description + f'Neck is NORMAL from right side view - ' \
                                                          f'right hip_shoulder_ear angle: {r_hip_shoulder_ear_angle}\n'
                self.draw_angles(r_hip_shoulder_ear_points, r_hip_shoulder_ear_angle, healthy)

            if l_hip_shoulder_ear_angle:
                healthy = False
                if l_hip_shoulder_ear_angle < 140:
                    monitor_score += 1
                    self.draw_lines_between_pairs(l_hip_shoulder_ear_points, False)
                    self.description = self.description + f'Neck is BENT FORWARD from left side view - ' \
                                                          f'left hip_shoulder_ear angle: {l_hip_shoulder_ear_angle} \n'
                elif l_hip_shoulder_ear_angle > 200:
                    monitor_score += 3
                    self.draw_lines_between_pairs(l_hip_shoulder_ear_points, False)
                    self.description = self.description + f'Neck is BENT BACKWARD from left side view - ' \
                                                          f'left hip_shoulder_ear angle: {l_hip_shoulder_ear_angle} \n'
                else:
                    healthy = True
                    self.draw_lines_between_pairs(l_hip_shoulder_ear_points)
                    self.description = self.description + f'Neck is NORMAL from left side view - ' \
                                                          f'left hip_shoulder_ear angle: {l_hip_shoulder_ear_angle} \n'
                self.draw_angles(l_hip_shoulder_ear_points, l_hip_shoulder_ear_angle, healthy)

            # healthy = False
            r_ear_eye_shoulder_angle = self.get_r_ear_eye_shoulder_angle()
            r_ear_eye_shoulder_points = [[self.pose_detector.REye, self.pose_detector.REar],
                                         [self.pose_detector.REar, self.pose_detector.RShoulder]]
            if r_ear_eye_shoulder_angle:
                if r_ear_eye_shoulder_angle < 75:
                    self.draw_lines_between_pairs(r_ear_eye_shoulder_points, False)
                    self.description = self.description + f'Neck is BENT DOWNWARD from side view - ' \
                                                          f'right ear_eye_shoulder angle: {r_ear_eye_shoulder_angle} \n'
                elif r_ear_eye_shoulder_angle > 125:
                    self.draw_lines_between_pairs(r_ear_eye_shoulder_points, False)
                    self.description = self.description + f'Neck is BENT UPWARD from side view - ' \
                                                          f'right ear_eye_shoulder angle: {r_ear_eye_shoulder_angle} \n'
                self.draw_angles(r_ear_eye_shoulder_points, r_ear_eye_shoulder_angle)

            # healthy = False
            l_ear_eye_shoulder_angle = self.get_l_ear_eye_shoulder_angle()
            l_ear_eye_shoulder_points = [[self.pose_detector.LEye, self.pose_detector.LEar],
                                         [self.pose_detector.LEar, self.pose_detector.LShoulder]]
            if l_ear_eye_shoulder_angle:
                if l_ear_eye_shoulder_angle < 75:
                    self.draw_lines_between_pairs(l_ear_eye_shoulder_points, False)
                    self.description = self.description + f'Neck is BENT DOWNWARD from side view - ' \
                                                          f'left ear_eye_shoulder angle: {l_ear_eye_shoulder_angle} \n'
                elif l_ear_eye_shoulder_angle > 125:
                    self.draw_lines_between_pairs(l_ear_eye_shoulder_points, False)
                    self.description = self.description + f'Neck is BENT UPWARD from side view - ' \
                                                          f'left ear_eye_shoulder angle: {l_ear_eye_shoulder_angle} \n'
                self.draw_angles(l_ear_eye_shoulder_points, l_ear_eye_shoulder_angle)

        if self.camera_view_point == "front":
            # neck rule front
            if self.points[self.pose_detector.LEye] and self.points[self.pose_detector.REye] and \
                    self.points[self.pose_detector.LShoulder] and self.points[self.pose_detector.RShoulder]:
                v1 = np.array(self.points[self.pose_detector.LEye]) - np.array(self.points[self.pose_detector.REye])
                v2 = np.array(self.points[self.pose_detector.LShoulder]) - np.array(
                    self.points[self.pose_detector.RShoulder])
                angle_between_shoulders_and_eyes = self.get_angle_between_lines(v1, v2)
                two_eyes_vertical_axis_angle = self.get_angle_between_vector_and_vertical_axis(
                    np.array(self.points[self.pose_detector.LEye]) - np.array(self.points[self.pose_detector.REye]))
                two_shoulders_vertical_axis_angle = self.get_angle_between_vector_and_vertical_axis(
                    np.array(self.points[self.pose_detector.LShoulder]) - np.array(
                        self.points[self.pose_detector.RShoulder]))
                two_eyes_points = [[self.pose_detector.LEye, self.pose_detector.REye]]
                two_shoulders_points = [[self.pose_detector.LShoulder, self.pose_detector.RShoulder]]
                if math.fabs(two_eyes_vertical_axis_angle - two_shoulders_vertical_axis_angle) > 30:
                    if 30 < angle_between_shoulders_and_eyes < 90:
                        self.description = self.description + f'Neck is BENT LEFTWARD from front view - ' \
                                                              f'angle_between_shoulders_and_eyes: {angle_between_shoulders_and_eyes} \n'
                    elif 120 < angle_between_shoulders_and_eyes < 150:
                        self.description = self.description + f'Neck is BENT RIGHTWARD from front view - ' \
                                                              f'angle_between_shoulders_and_eyes: {angle_between_shoulders_and_eyes} \n'
                    self.draw_lines_between_pairs(two_eyes_points, False)
                    self.draw_lines_between_pairs(two_shoulders_points, False)
                else:
                    self.description = self.description + f'Neck is not BENT RIGHTWARD or leftward - ' \
                                                          f'angle_between_shoulders_and_eyes: {angle_between_shoulders_and_eyes} \n'
                    self.draw_lines_between_pairs(two_eyes_points)
                    self.draw_lines_between_pairs(two_shoulders_points)
        return monitor_score

    def get_phone_score(self):
        phone_score = 1

        shoulders_distance = None
        wrists_distance = None
        wrists_horizontal_axis_angle = None
        two_wrists_points = [[self.pose_detector.LWrist, self.pose_detector.RWrist]]

        if self.camera_view_point == "front":
            if self.points[self.pose_detector.RShoulder] and self.points[self.pose_detector.LShoulder]:
                shoulders_distance = self.calculate_distance_between_two_points(
                    self.points[self.pose_detector.RShoulder],
                    self.points[self.pose_detector.LShoulder])

            if self.points[self.pose_detector.RWrist] and self.points[self.pose_detector.LWrist]:
                wrists_distance = self.calculate_distance_between_two_points(self.points[self.pose_detector.RWrist],
                                                                             self.points[self.pose_detector.LWrist])
                wrists_horizontal_axis_angle = self.get_angle_between_vector_and_horizontal_axis(
                    np.array(self.points[self.pose_detector.RWrist]) - np.array(self.points[self.pose_detector.LWrist])
                )

            if (shoulders_distance and wrists_distance) or wrists_horizontal_axis_angle:
                if math.fabs(wrists_distance - shoulders_distance) > shoulders_distance / 3:
                    phone_score = 2
                    self.description = self.description + 'TOO WIDE wrists\n'
                    self.draw_lines_between_pairs(two_wrists_points, False)
                if wrists_horizontal_axis_angle:
                    if wrists_horizontal_axis_angle > 10:
                        phone_score += 1
                        self.description = self.description + 'Two wrists  are NOT ON THE SAME SURFACE\n'
                        self.draw_lines_between_pairs(two_wrists_points, False)

            if self.points[self.pose_detector.Neck] and self.points[self.pose_detector.Nose]:
                neck_nose_points = [[self.pose_detector.Neck, self.pose_detector.Nose]]
                neck_nose_angle = self.get_angle_between_vector_and_vertical_axis(
                    np.array(self.points[self.pose_detector.Neck]) - np.array(self.points[self.pose_detector.Nose]))
                if neck_nose_angle:
                    if neck_nose_angle > 30:
                        self.description = self.description + f'Neck is BENT - ' \
                                                              f'angle between neck_nose and vertical axis: {neck_nose_angle}\n'
                        self.draw_lines_between_pairs(neck_nose_points, False)
                    else:
                        self.description = self.description + f'Neck is NORMAL - ' \
                                                              f'angle between neck_nose and vertical axis: {neck_nose_angle}\n'
                        self.draw_lines_between_pairs(neck_nose_points)

        return phone_score

    def get_r_hip_knee_ankle_angle(self):
        # rHipKneeAnkle_pairs = [[8, 9], [9, 10]]
        angle = self.get_angle_between_points(self.points[self.pose_detector.RHip],
                                              self.points[self.pose_detector.RKnee],
                                              self.points[self.pose_detector.RAnkle])
        return angle

    def get_l_hip_knee_ankle_angle(self):
        # lHipKneeAnkle_pairs = [[11, 12], [12, 13]]
        angle = self.get_angle_between_points(self.points[self.pose_detector.LHip],
                                              self.points[self.pose_detector.LKnee],
                                              self.points[self.pose_detector.LAnkle])
        return angle

    def get_shoulders_neck_angle(self):
        # shoulders_neck_pairs = [[2, 1], [1, 5]]
        angle = self.get_angle_between_points(self.points[self.pose_detector.RShoulder],
                                              self.points[self.pose_detector.Neck],
                                              self.points[self.pose_detector.LShoulder])
        return angle

    def get_r_neck_shoulder_elbow_angle(self):
        # rNeckShoulderElbow_pairs = [[1, 2], [2, 3]]
        angle = self.get_angle_between_points(self.points[self.pose_detector.Neck],
                                              self.points[self.pose_detector.RShoulder],
                                              self.points[self.pose_detector.RElbow])
        return angle

    def get_l_neck_shoulder_elbow_angle(self):
        # lNeckShoulderElbow_pairs = [[1, 5], [5, 6]]
        angle = self.get_angle_between_points(self.points[self.pose_detector.Neck],
                                              self.points[self.pose_detector.LShoulder],
                                              self.points[self.pose_detector.LElbow])
        return angle

    def get_r_shoulder_hip_knee_angle(self):
        # rShoulderHipKnee_pairs = [[2, 8], [8, 9]]
        angle = self.get_angle_between_points(self.points[self.pose_detector.RShoulder],
                                              self.points[self.pose_detector.RHip],
                                              self.points[self.pose_detector.RKnee])
        return angle

    def get_l_shoulder_hip_knee_angle(self):
        # lShoulderHipKnee_pairs = [[5, 11], [11, 12]]
        angle = self.get_angle_between_points(self.points[self.pose_detector.LShoulder],
                                              self.points[self.pose_detector.LHip],
                                              self.points[self.pose_detector.LKnee])
        return angle

    def get_r_shoulder_elbow_wrist(self):
        # r_shoulder_elbow_wrist_pairs = [[2, 3], [3, 4]]
        angle = self.get_angle_between_points(self.points[self.pose_detector.RShoulder],
                                              self.points[self.pose_detector.RElbow],
                                              self.points[self.pose_detector.RWrist])
        return angle

    def get_l_shoulder_elbow_wrist(self):
        # l_shoulder_elbow_wrist_pairs = [[5, 6], [6, 7]]
        angle = self.get_angle_between_points(self.points[self.pose_detector.LShoulder],
                                              self.points[self.pose_detector.LElbow],
                                              self.points[self.pose_detector.LWrist])
        return angle

    def get_r_hip_shoulder_elbow_angle(self):
        # rHipShoulderElbow_pairs = [[3, 2], [2, 8]]
        angle = self.get_angle_between_points(self.points[self.pose_detector.RElbow],
                                              self.points[self.pose_detector.RShoulder],
                                              self.points[self.pose_detector.RHip])
        return angle

    def get_l_hip_shoulder_elbow_angle(self):
        # lHipShoulderElbow_pairs = [[6, 5], [5, 11]]
        angle = self.get_angle_between_points(self.points[self.pose_detector.LElbow],
                                              self.points[self.pose_detector.LShoulder],
                                              self.points[self.pose_detector.LHip])
        return angle

    def get_r_shoulder_elbow_wrist_angle(self):
        # rShoulderElbowWrist_pairs = [[2, 3], [3, 4]]
        angle = self.get_angle_between_points(self.points[self.pose_detector.RShoulder],
                                              self.points[self.pose_detector.RElbow],
                                              self.points[self.pose_detector.RWrist])
        return angle

    def get_l_shoulder_elbow_wrist_angle(self):
        # lShoulderElbowWrist_pairs = [[5, 6], [6, 7]]
        angle = self.get_angle_between_points(self.points[self.pose_detector.LShoulder],
                                              self.points[self.pose_detector.LElbow],
                                              self.points[self.pose_detector.LWrist])
        return angle

    def get_r_hip_shoulder_ear_angle(self):
        # r_hip_shoulder_ear_pairs = [[8, 2], [2, 16]]
        angle = self.get_angle_between_points(self.points[self.pose_detector.RHip],
                                              self.points[self.pose_detector.RShoulder],
                                              self.points[self.pose_detector.REar])
        return angle

    def get_l_hip_shoulder_ear_angle(self):
        # l_hip_shoulder_ear_pairs = [[11, 5], [5, 17]]
        angle = self.get_angle_between_points(self.points[self.pose_detector.LHip],
                                              self.points[self.pose_detector.LShoulder],
                                              self.points[self.pose_detector.LEar])
        return angle

    def get_r_ear_eye_shoulder_angle(self):
        # l_ear_eye_shoulder_pairs = [[16, 14], [14, 2]]
        angle = self.get_angle_between_points(self.points[self.pose_detector.REye],
                                              self.points[self.pose_detector.REar],
                                              self.points[self.pose_detector.RShoulder])
        return angle

    def get_l_ear_eye_shoulder_angle(self):
        # l_ear_eye_shoulder_pairs = [[17, 15], [15, 5]]
        angle = self.get_angle_between_points(self.points[self.pose_detector.LEye],
                                              self.points[self.pose_detector.LEar],
                                              self.points[self.pose_detector.LShoulder])
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
        return round(math.acos(round(np.dot(v1, v2) / (len_v1 * len_v2), 5)) * 180 / math.pi, 2)

    def get_vectors_between_points(self, p1, p2, p3, p4):
        v1 = np.array(p2) - np.array(p1)
        v2 = np.array(p4) - np.array(p3)
        return v1, v2

    def get_angle_between_points(self, p1, p2, p3):
        if p1 and p2 and p3:
            v1, v2 = self.get_vectors_between_points(p2, p1, p2, p3)
            angle = self.get_angle_between_lines(v1, v2)
            if angle:
                return round(angle, 2)
        return None

    def display_joint_points(self):
        for point in self.points:
            if point:
                cv2.circle(self.image, point, self.circle_radius, (0, 255, 0), thickness=-1, lineType=cv2.FILLED)

    def draw_lines_between_pairs(self, pairs, is_correct_edge=True):
        for pair in pairs:
            part_a = pair[0]
            part_b = pair[1]

            if self.points[part_a] and self.points[part_b]:
                if pair in self.repetitive_pairs:
                    updated_point_a1 = self.points[part_a][0] + 5
                    updated_point_a2 = self.points[part_a][1] + 5
                    updated_point_b1 = self.points[part_b][0] + 5
                    updated_point_b2 = self.points[part_b][1] + 5
                    if is_correct_edge:
                        cv2.line(self.image, (updated_point_a1, updated_point_a2), (updated_point_b1, updated_point_b2),
                                 (0, 255, 255), self.line_thickness)
                    else:
                        cv2.line(self.image, (updated_point_a1, updated_point_a2), (updated_point_b1, updated_point_b2),
                                 (0, 0, 255), self.line_thickness)
                        self.repetitive_pairs.append(pair)
                else:
                    if is_correct_edge:
                        cv2.line(self.image, self.points[part_a], self.points[part_b], (0, 255, 255),
                                 self.line_thickness)
                    else:
                        cv2.line(self.image, self.points[part_a], self.points[part_b], (0, 0, 255), self.line_thickness)
                        self.repetitive_pairs.append(pair)
                    self.repetitive_pairs.append(pair)

    def save_image(self, is_correct_posture, output_directory, file_name):
        img_copy = self.image.copy()
        img = np.array(img_copy)[:, :, ::-1]
        face_blur_provider = FaceBlurring()
        blured_image = face_blur_provider.blur_face(self.points, self.pose_detector.Nose, self.pose_detector.LEye,
                                                    self.pose_detector.REye, self.pose_detector.LEar,
                                                    self.pose_detector.REar, img)
        plt.imshow(blured_image)
        plt.axis('off')
        if is_correct_posture:
            plt.text(10, 30, 'Healthy', color='yellow', fontsize=14)
            plt.savefig(f'{output_directory}/ModelDetection/{file_name}')
        else:
            plt.text(10, 30, 'Unhealthy', color='red', fontsize=14)
            plt.savefig(f'{output_directory}/ModelDetection/{file_name}')
        plt.close()

        text_file = open(f'{output_directory}/log.txt', "a")
        text_file.write(f'Description of {file_name}:\n{self.description}'
                        f'f"*************************************************************************\n')
        text_file.close()

    def draw_angles(self, pairs, angle, is_healthy=False):
        if pairs[0] and pairs[1]:
            in_common_element = [i for i in pairs[0] if i in pairs[1]][0]
            if is_healthy:
                # plt.text(self.points[in_common_element][0], self.points[in_common_element][1], f'{angle}',
                #          color='yellow', fontsize=8)
                cv2.putText(self.image, f"{angle}",
                            (self.points[in_common_element][0], self.points[in_common_element][1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 0), 2)  # color : light blue
            else:
                # plt.text(self.points[in_common_element][0], self.points[in_common_element][1], f'{angle}',
                #          color='red', fontsize=8)
                cv2.putText(self.image, f"{angle}",
                            (self.points[in_common_element][0], self.points[in_common_element][1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)  # color : yellow
