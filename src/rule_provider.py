import math
import os
import csv

import pandas as pd
import numpy as np
from pose_detector import PoseDetector
from matplotlib import pyplot as plt
from face_blurring import FaceBlurring
import cv2
import matplotlib.pyplot as plt


class RuleProvider:
    result = {
        'image_number': [],
        'back': []
    }
    def __init__(self, pose_detector):
        self.image = None
        self.points = None
        self.pose_detector = pose_detector
        self.circle_radius = 3
        self.line_thickness = 2
        self.description = ""
        #self.repetitive_pairs = []
        self.camera_view_point = "front"
        self.th = 0 
        self.resize_factor= 1 

    def get_labels(self, file_name):
        labels = dict()
        with open(file_name, 'r') as file:
            reader = csv.reader(file)
            for i , row in enumerate(reader):
                if i > 0:
                    labels[row[0]] = list(map(lambda x: int(x), row[1:]))
        return labels

    def get_posture_status(self, image, points, file_name, view_point, output_path, front_label_path, side_label_path, draw_joint_points=True):
        self.result['image_number'].append(file_name.split(sep='.')[0].split(sep='_')[1])
        self.description = ""
        self.image = image
        self.points = points
        self.camera_view_point = view_point
        self.output_path = output_path
        figsize = (int(image.shape[0] * self.resize_factor), int(image.shape[1] * self.resize_factor))
        if view_point == "front":
            self.figure_front, self.axs_front = plt.subplots(1, 3) #, figsize=figsize)
            self.figure_front.tight_layout()
            arm_score = 1
            phone_score = 1
            trunk_score = 1
            self.labels = self.get_labels(front_label_path)
        else:
            self.figure_side, self.axs_side = plt.subplots(1, 3) #, figsize=figsize)
            self.figure_side.tight_layout()
            chair_score = 1
            backrest_score = 1
            monitor_score = 1
            self.labels = self.get_labels(side_label_path)
        #mouse_score = 1
        posture_status = False
        self.file_name = os.path.splitext(file_name)[0]
        if draw_joint_points:
            self.display_joint_points(file_name)

        print(f'ROSA score is checking for {file_name} ...\n')

        if self.camera_view_point == "side":
            # Chair Height & Pan Depth (3/7)
            chair_score = self.get_chair_score()

            # Backrest (2/5)
            backrest_score = self.get_backrest_score()
            self.result['back'].append(backrest_score)

            # Monitor (0/6)
            monitor_score = self.get_monitor_score()
            self.figure_side.savefig(f'{self.output_path}/{self.file_name}.JPG')
            #self.figure_side.close()
            plt.close()


        else:
            #import pudb; pu.db

            # Armrest (3/4)
            arm_score = self.get_arm_score()
            trunk_score = self.get_trunk_score()

            # Telephone (1/3)
            phone_score = self.get_phone_score()
            self.figure_front.savefig(f'{self.output_path}/{self.file_name}.JPG')
            plt.close()


        if self.camera_view_point == 'side':
            print(f'chair score is: {chair_score}\n'
                  f'backrest score is: {backrest_score}\n'
                  f'monitor score is: {monitor_score}\n')
        else:
            print(f'arm score is: {arm_score}\n'
                    f'phone score is: {phone_score}\n')

        return posture_status

    def get_chair_score(self):
        self.get_blure_image()
        axis = self.axs_side[0]
        self.axs_side[0].set_title('Chair Score')
        self.axs_side[0].axis('off')
        #axis.imshow(self.blured_image)
        axis.imshow(self.resize(self.blured_image))

        chair_score = 1
        diff_x_rknee_rhip = None
        diff_x_lknee_lhip = None

        r_hip_knee_ankle_angle = self.get_r_hip_knee_ankle_angle()
        l_hip_knee_ankle_angle = self.get_l_hip_knee_ankle_angle()

        if r_hip_knee_ankle_angle or l_hip_knee_ankle_angle:
            if r_hip_knee_ankle_angle:
                r_hip_knee_ankle_points = [self.pose_detector.RHip, self.pose_detector.RKnee, self.pose_detector.RAnkle]
                diff_x_rknee_rhip = self.points[self.pose_detector.RKnee][0] - self.points[self.pose_detector.RHip][0]
                #import pudb; pu.db
                if r_hip_knee_ankle_angle < (86 - self.th):
                    r_chair_score = 2
                    c = 'red'
                    self.description = self.description + f'Right hip_knee_ankle angle: {r_hip_knee_ankle_angle}\n'
                elif r_hip_knee_ankle_angle > (104 + self.th):
                    r_chair_score = 2
                    c = 'red'
                    self.description = self.description + f'Right hip_knee_ankle angle: {r_hip_knee_ankle_angle}\n'
                else:
                    c = 'green'
                    r_chair_score = 1
                    self.description = self.description + \
                                       f'Right knee status is in CORRECT POSTURE - right hip_knee_ankle angle: {r_hip_knee_ankle_angle}\n'
                self.draw_lines_between_pairs(axis, r_hip_knee_ankle_points, c)
                self.put_text_add_description(axis, self.output_path, self.file_name, 'Right leg angle', r_hip_knee_ankle_angle, c)

            if l_hip_knee_ankle_angle:
                l_hip_knee_ankle_points = [self.pose_detector.LHip, self.pose_detector.LKnee, self.pose_detector.LAnkle]
                diff_x_lknee_lhip = self.points[self.pose_detector.LKnee][0] - self.points[self.pose_detector.LHip][0]
                if l_hip_knee_ankle_angle < (86 - self.th):
                    l_chair_score = 2
                    c = 'red'
                    self.description = self.description + f'Left hip_knee_ankle angle: {l_hip_knee_ankle_angle}\n'
                elif l_hip_knee_ankle_angle > (104 + self.th):
                    l_chair_score = 2
                    c = 'red'
                    self.description = self.description + f'Left hip_knee_ankle angle: {l_hip_knee_ankle_angle}\n'
                else:
                    c = 'green'
                    l_chair_score = 1
                    self.description = self.description + \
                                       f'Left knee status is in CORRECT POSTURE - left hip_knee_ankle angle: {l_hip_knee_ankle_angle}\n'
                self.draw_lines_between_pairs(axis, l_hip_knee_ankle_points, c)
                self.put_text_add_description(axis, self.output_path, self.file_name, 'Left leg angle', l_hip_knee_ankle_angle, c, 10 , 60)
                #import pudb; pu.db
            if r_hip_knee_ankle_angle and l_hip_knee_ankle_angle:
                # Condition 1: Front Leg
                if diff_x_lknee_lhip > 0:
                    chair_score = r_chair_score
                else:
                    chair_score = l_chair_score
                ## Condition 2: Worst Case
                #chair_score = max(r_chair_score, l_chair_score)
            elif r_hip_knee_ankle_angle:
                chair_score = r_chair_score
            else:
                chair_score = l_chair_score


            self.put_score_label(axis, 'chair', chair_score, self.labels[self.file_name][0])

        else:
            print("Not Enough Info")
            self.put_text_add_description(axis, self.output_path, self.file_name, "Not Enough Info", None, 'red')
            chair_score = None
        #import pudb; pu.db
        #axis.imshow(self.resize(self.blured_image))
        return chair_score

    def get_arm_score(self):
        arm_score = 1
        self.get_blure_image()
        axis = self.axs_front[0]
        self.axs_front[0].set_title('Armrest Score')
        self.axs_front[0].axis('off')
        axis.imshow(self.resize(self.blured_image))
        #if True: #self.camera_view_point == "front": ###################################################
        if hasattr(self.pose_detector, 'Neck'):
            shoulders_neck_angle = self.get_shoulders_neck_angle()
            shoulders_neck_points = [self.pose_detector.Neck, self.pose_detector.RShoulder, self.pose_detector.LShoulder]
            if shoulders_neck_angle:
                if shoulders_neck_angle < 160:
                    arm_score = 2
                    c = 'red'
                    #self.draw_lines_between_pairs(shoulders_neck_points, False)
                    self.description = self.description + f'Shoulders are SHRUGGED - shoulders_neck angle: {shoulders_neck_angle} \n'
                else:
                    c = 'green'
                    #self.draw_lines_between_pairs(shoulders_neck_points)
                    self.description = self.description + f'Shoulders are in NORMAL POSTURE - shoulders_neck angle: {shoulders_neck_angle} \n'
                self.draw_lines_between_pairs(axis, shoulders_neck_points, c)
                self.put_text_add_description(axis, self.output_path, self.file_name, 'Neck angle', shoulders_neck_angle, c) #, 10 , 60)
            else:
                print("Not Enough Info")
                self.put_text_add_description(axis, self.output_path, self.file_name, "NEI ", None, 'red')
        else:
            print("Not Enough Info")
            self.put_text_add_description(axis, self.output_path, self.file_name, "NEI shrogged ", None, 'red')
            #import pudb; pu.db
        #########################################################################################
        r_elbow_shoulder_hip_angle = self.get_r_elbow_shoulder_hip_angle()
        l_elbow_shoulder_hip_angle = self.get_l_elbow_shoulder_hip_angle()
        
        r_elbow_shoulder_hip_angle_points = [self.pose_detector.RElbow, self.pose_detector.RShoulder, 
                self.pose_detector.RHip]

        l_elbow_shoulder_hip_angle_points = [self.pose_detector.LElbow, self.pose_detector.LShoulder,
                self.pose_detector.LHip]
        if r_elbow_shoulder_hip_angle or l_elbow_shoulder_hip_angle:
            if r_elbow_shoulder_hip_angle is not None:
                if r_elbow_shoulder_hip_angle > 40:
                    arm_score += 1
                    c = 'red'
                    self.description = self.description + f'Right elbow is NOT INLINE with right shoulder - ' \
                                                          f'Angle between right shoulder_elbow and vertical axis: ' \
                                                          f'{r_elbow_shoulder_hip_angle}\n'
                else:
                    c = 'green'
                    self.description = self.description + f'Right elbow is INLINE with right shoulder - ' \
                                                          f'Angle between right shoulder_elbow and vertical axis: ' \
                                                          f'{r_elbow_shoulder_hip_angle}\n'
                self.draw_lines_between_pairs(axis, r_elbow_shoulder_hip_angle_points, c)
                self.put_text_add_description(axis, self.output_path, self.file_name, 'Right elbow angle', r_elbow_shoulder_hip_angle, c, 10, 60)

            if l_elbow_shoulder_hip_angle is not None:
                if l_elbow_shoulder_hip_angle > 40:
                    arm_score += 1
                    c = 'red'
                    self.description = self.description + f'Left elbow is NOT INLINE with left shoulder - ' \
                                                          f'Angle between left shoulder_elbow and vertical axis:' \
                                                          f' {l_elbow_shoulder_hip_angle}\n'
                else:
                    c = 'green'
                    self.description = self.description + f'Left elbow is INLINE with left shoulder - ' \
                                                          f'Angle between left shoulder_elbow and vertical axis: ' \
                                                          f'{l_elbow_shoulder_hip_angle}\n'
                self.draw_lines_between_pairs(axis, l_elbow_shoulder_hip_angle_points, c)
                self.put_text_add_description(axis, self.output_path, self.file_name, 'Left elbow angle', l_elbow_shoulder_hip_angle, c, 10, 90)
            self.put_score_label(axis, 'arm', arm_score, self.labels[self.file_name][0] + self.labels[self.file_name][1] -1)
        else:
            print("Not Enough Info")
            self.put_text_add_description(axis, self.output_path, self.file_name, "Not Enough Info", None, 'red', 10, 120)
            arm_score = None
        #import pudb; pu.db
        #plt.show()
        return arm_score



    def get_backrest_score(self):
        self.get_blure_image()
        axis = self.axs_side[1]
        self.axs_side[1].set_title('Backrest Score')
        self.axs_side[1].axis('off')
        #axis.imshow(self.blured_image)
        axis.imshow(self.resize(self.blured_image))
        backrest_score = 1

        if (((self.points[self.pose_detector.LHip] is not None) and 
                (self.points[self.pose_detector.LShoulder] is not None)) or
                ((self.points[self.pose_detector.RHip] is not None) and
                    (self.points[self.pose_detector.RShoulder] is not None))):

            if self.points[self.pose_detector.RHip] is None:  # if one of the hips is not available
                self.points[self.pose_detector.RHip] = self.points[self.pose_detector.LHip]
            elif self.points[self.pose_detector.LHip] is None:
                self.points[self.pose_detector.LHip] = self.points[self.pose_detector.RHip]

            if self.points[self.pose_detector.RShoulder] is None:  # if one of the shoulders is not available
                self.points[self.pose_detector.RShoulder] = self.points[self.pose_detector.LShoulder]
            elif self.points[self.pose_detector.LShoulder] is None:
                self.points[self.pose_detector.LShoulder] = self.points[self.pose_detector.RShoulder]
            
            # middle of shoulders (x coordinate)
            MidHip = tuple(((np.array(self.points[self.pose_detector.RHip]) + 
                np.array(self.points[self.pose_detector.LHip]))/2).astype(int))
            MidShoulder = tuple(((np.array(self.points[self.pose_detector.RShoulder]) + 
                np.array(self.points[self.pose_detector.LShoulder]))/2).astype(int))

            #import pudb; pu.db
            mid_shoulder_hip_knee = self.get_mid_shoulder_hip_knee_angle(MidHip, MidShoulder)
            mid_shoulder_hip_knee_points = [[list(MidShoulder)[0], list(MidShoulder)[1]], [list(MidHip)[0], list(MidHip)[1]],
                    [list(MidHip)[0] - 50, list(MidHip)[1]]]
            if mid_shoulder_hip_knee:
                if mid_shoulder_hip_knee < (95 - self.th):
                    backrest_score = 2
                    c = 'red'
                    self.description = self.description + f'Back rest is BENT FORWARD from - ' \
                                                          f'mid shoulder_hip_knee angle: {mid_shoulder_hip_knee}\n'
                elif mid_shoulder_hip_knee > (110 + self.th):
                    backrest_score = 2
                    self.description = self.description + f'Back rest is BENT BACKWARD - ' \
                                                          f'mid shoulder_hip_knee angle: {mid_shoulder_hip_knee}\n'
                    c = 'red'
                else:
                    c = 'green'
                    self.description = self.description + f'Back rest is NORMAL - ' \
                                                          f'mid shoulder_hip_knee angle: {mid_shoulder_hip_knee}\n'
            
            self.draw_lines_between_pairs(axis, mid_shoulder_hip_knee_points, c, True)
            self.put_text_add_description(axis, self.output_path, self.file_name, 'back angle', mid_shoulder_hip_knee, c, 10, 30)
            self.put_score_label(axis, 'back rest', backrest_score, self.labels[self.file_name][0])

        else:
            print("Not Enough Info")
            self.put_text_add_description(axis, self.output_path, self.file_name, "Not Enough Info", None, 'red')
            backrest_score = None
        #import pudb; pu.db
        #axis.imshow(self.blured_image)
        return backrest_score
        

    def get_monitor_score(self):
        #import pudb; pu.db
        monitor_score = 1
        self.get_blure_image()
        axis = self.axs_side[2]
        self.axs_side[2].set_title('monitor_score')
        self.axs_side[2].axis('off')
        #axis.imshow(self.blured_image)
        axis.imshow(self.resize(self.blured_image))
        # if True: #self.camera_view_point == "side":
        r_hip_shoulder_ear_angle = self.get_r_hip_shoulder_ear_angle()
        r_hip_shoulder_ear_points = [self.pose_detector.RHip, self.pose_detector.RShoulder, self.pose_detector.REar]
        l_hip_shoulder_ear_angle = self.get_l_hip_shoulder_ear_angle()
        l_hip_shoulder_ear_points = [self.pose_detector.LHip, self.pose_detector.LShoulder, self.pose_detector.LEar]

        if r_hip_shoulder_ear_angle or l_hip_shoulder_ear_angle:
            if r_hip_shoulder_ear_angle:
                if r_hip_shoulder_ear_angle < 140:
                    monitor_score += 1
                    c = 'red'
                    #self.draw_lines_between_pairs(r_hip_shoulder_ear_points, False)
                    self.description = self.description + f'Neck is BENT FORWARD from right side view - ' \
                                                          f'right hip_shoulder_ear angle: {r_hip_shoulder_ear_angle}\n'
                elif r_hip_shoulder_ear_angle > 200:
                    monitor_score += 3
                    c = 'red'
                    #self.draw_lines_between_pairs(r_hip_shoulder_ear_points, False)
                    self.description = self.description + f'Neck is BENT BACKWARD from right side view - ' \
                                                          f'right hip_shoulder_ear angle: {r_hip_shoulder_ear_angle}\n'
                else:
                    c = 'green'
                    #self.draw_lines_between_pairs(r_hip_shoulder_ear_points)
                    self.description = self.description + f'Neck is NORMAL from right side view - ' \
                                                          f'right hip_shoulder_ear angle: {r_hip_shoulder_ear_angle}\n'
                self.draw_lines_between_pairs(axis, r_hip_shoulder_ear_points, c)
                self.put_text_add_description(axis, self.output_path, self.file_name, 'neck angle', r_hip_shoulder_ear_angle, c)

            if l_hip_shoulder_ear_angle:
                if l_hip_shoulder_ear_angle < 140:
                    monitor_score += 1
                    c = 'red'
                    #self.draw_lines_between_pairs(l_hip_shoulder_ear_points, False)
                    self.description = self.description + f'Neck is BENT FORWARD from left side view - ' \
                                                          f'left hip_shoulder_ear angle: {l_hip_shoulder_ear_angle} \n'
                elif l_hip_shoulder_ear_angle > 200:
                    monitor_score += 3
                    c = 'red'
                    #self.draw_lines_between_pairs(l_hip_shoulder_ear_points, False)
                    self.description = self.description + f'Neck is BENT BACKWARD from left side view - ' \
                                                          f'left hip_shoulder_ear angle: {l_hip_shoulder_ear_angle} \n'
                else:
                    c = 'green'
                    #self.draw_lines_between_pairs(l_hip_shoulder_ear_points)
                    self.description = self.description + f'Neck is NORMAL from left side view - ' \
                                                          f'left hip_shoulder_ear angle: {l_hip_shoulder_ear_angle} \n'
                self.draw_lines_between_pairs(axis, l_hip_shoulder_ear_points, c)
                self.put_text_add_description(axis, self.output_path, self.file_name, 'neck angle', l_hip_shoulder_ear_angle, c)
            #import pudb; pu.db
            self.put_score_label(axis, 'monitor', monitor_score, self.labels[self.file_name][2])
        else:
            print("Not Enough Info")
            self.put_text_add_description(axis, self.output_path, self.file_name, "Not Enough Info", None, 'red')
            monitor_score = None
        
        #import pudb; pu.db
        #axis.imshow(self.resize(self.blured_image))
        return monitor_score 


    def get_phone_score(self):
        #import pudb; pu.db
        phone_score = 1
        self.get_blure_image()
        axis = self.axs_front[1]
        self.axs_front[1].set_title('Phone Score')
        self.axs_front[1].axis('off')
        axis.imshow(self.blured_image)

        #import pudb; pu.db
        angle_between_shoulders_and_eyes = self.get_angle_between_shoulders_and_eyes()


        two_eyes_points = [self.pose_detector.LEye, self.pose_detector.REye]
        two_shoulders_points = [self.pose_detector.LShoulder, self.pose_detector.RShoulder]
        #import pudb; pu.db
        #    if math.fabs(two_eyes_vertical_axis_angle - two_shoulders_vertical_axis_angle) > 30:
        if angle_between_shoulders_and_eyes is not None:
            if angle_between_shoulders_and_eyes > 16:
                c = 'red'
                phone_score += 1
                self.description = self.description + f'Neck is BENT LEFTWARD from front view - ' \
                                                      f'angle_between_shoulders_and_eyes: {angle_between_shoulders_and_eyes} \n'
           # elif 120 < angle_between_shoulders_and_eyes < 150:
           #     c = 'red'
           #     phone_score += 1
           #     self.description = self.description + f'Neck is BENT RIGHTWARD from front view - ' \
           #                                           f'angle_between_shoulders_and_eyes: {angle_between_shoulders_and_eyes} \n'
            else:
                c = 'green'
                self.description = self.description + f'Neck is not BENT RIGHTWARD or leftward - ' \
                                                      f'angle_between_shoulders_and_eyes: {angle_between_shoulders_and_eyes} \n'
            
            self.draw_lines_between_pairs(axis, two_eyes_points, c)
            self.draw_lines_between_pairs(axis, two_shoulders_points, c)
            self.put_text_add_description(axis, self.output_path, self.file_name, 'angle between 2 \n eyes and 2 shoulders', angle_between_shoulders_and_eyes, c)
            self.put_score_label(axis, 'phone', phone_score,
                    self.labels[self.file_name][2])
        else:
            print("Not Enough Info")
            self.put_text_add_description(axis, self.output_path, self.file_name, "Not Enough Info", None, 'red')
            monitor_score = None

            #return monitor_score
            
        # import pudb; pu.db

        return phone_score
#############################################################################################

    def get_trunk_score(self):
        self.get_blure_image()
        axis = self.axs_front[2]
        self.axs_front[2].set_title('Trunk Score')
        self.axs_front[2].axis('off')
        #axis.imshow(self.blured_image)
        axis.imshow(self.resize(self.blured_image))
        backrest_score = 1
        # import pudb; pu.db
        if self.points[self.pose_detector.RHip] and self.points[self.pose_detector.RShoulder]:
            r_shoulder_hip_vertical_angle = self.get_r_shoulder_hip_vertical_angle()
            r_shoulder_hip_vertical_points = [self.points[self.pose_detector.RShoulder],
                self.points[self.pose_detector.RHip],
                (self.points[self.pose_detector.RHip][0], self.points[self.pose_detector.RHip][1] - 100)]
        else: 
            r_shoulder_hip_vertical_angle = None

        if self.points[self.pose_detector.LHip] and self.points[self.pose_detector.LShoulder]:
            l_shoulder_hip_vertical_angle = self.get_l_shoulder_hip_vertical_angle()
            l_shoulder_hip_vertical_points = [self.points[self.pose_detector.LShoulder],
                self.points[self.pose_detector.LHip],
                (self.points[self.pose_detector.LHip][0], self.points[self.pose_detector.LHip][1] - 100)]
        else: l_shoulder_hip_vertical_angle = None
        
        if r_shoulder_hip_vertical_angle or l_shoulder_hip_vertical_angle:
            if r_shoulder_hip_vertical_angle and l_shoulder_hip_vertical_angle:
                if max(r_shoulder_hip_vertical_angle, l_shoulder_hip_vertical_angle) > 15:
                    backrest_score = 2 
                    c = 'red'
                else: 
                    c = 'green'
                self.draw_lines_between_pairs(axis, r_shoulder_hip_vertical_points, c, True)
                self.draw_lines_between_pairs(axis, l_shoulder_hip_vertical_points, c, True)
                self.put_text_add_description(axis, self.output_path, self.file_name, 'back angle_r', r_shoulder_hip_vertical_angle , c, 9, 30)
                self.put_text_add_description(axis, self.output_path, self.file_name, 'back angle_l', l_shoulder_hip_vertical_angle , c, 9, 60)

            elif r_shoulder_hip_vertical_angle:
                if r_shoulder_hip_vertical_angle > 15:
                    backrest_score = 2
                    c = 'red'
                else:
                    c = 'green'
                self.draw_lines_between_pairs(axis, r_shoulder_hip_vertical_points, c, True)
                self.put_text_add_description(axis, self.output_path, self.file_name, 'back angle_r', r_shoulder_hip_vertical_angle , c, 9, 30)
            elif l_shoulder_hip_vertical_angle:
                if l_shoulder_hip_vertical_angle > 15:
                    backrest_score = 2       
                    c = 'red'
                else:
                    c = 'green'
                self.draw_lines_between_pairs(axis, l_shoulder_hip_vertical_points, c, True)
                self.put_text_add_description(axis, self.output_path, self.file_name, 'back angle_l', l_shoulder_hip_vertical_angle , c, 9, 30)

            self.put_score_label(axis, 'back rest', backrest_score, self.labels[self.file_name][3])
        
        else:
            print("Not Enough Info")
            self.put_text_add_description(axis, self.output_path, self.file_name, "Not Enough Info", None, 'red')
            backrest_score = None
            #import pudb; pu.db
            #axis.imshow(self.blured_image)
        return backrest_score


    def get_angle_between_shoulders_and_eyes(self):
        if self.points[self.pose_detector.LEye] and self.points[self.pose_detector.REye] and \
                self.points[self.pose_detector.LShoulder] and self.points[self.pose_detector.RShoulder]:
            v1 = np.array(self.points[self.pose_detector.LEye]) - np.array(self.points[self.pose_detector.REye])
            v2 = np.array(self.points[self.pose_detector.LShoulder]) - np.array(self.points[self.pose_detector.RShoulder])
            angle = self.get_angle_between_lines(v1, v2)
            if angle is None:
                return None
            else:
                return round(angle, 1)
        else:
            return None


    def get_r_hip_knee_ankle_angle(self):
        # rHipKneeAnkle_pairs = [[8, 9], [9, 10]]
        angle = self.get_angle_between_points(self.points[self.pose_detector.RHip], self.points[self.pose_detector.RKnee],
                                              self.points[self.pose_detector.RAnkle])
        return angle

    def get_l_hip_knee_ankle_angle(self):
        # lHipKneeAnkle_pairs = [[11, 12], [12, 13]]
        angle = self.get_angle_between_points(self.points[self.pose_detector.LHip], self.points[self.pose_detector.LKnee],
                                              self.points[self.pose_detector.LAnkle])
        return angle

    def get_r_elbow_shoulder_hip_angle(self):
        angle = self.get_angle_between_points(self.points[self.pose_detector.RElbow], self.points[self.pose_detector.RShoulder],
                                                              self.points[self.pose_detector.RHip])
        return angle

    def get_l_elbow_shoulder_hip_angle(self):
        angle = self.get_angle_between_points(self.points[self.pose_detector.LElbow], self.points[self.pose_detector.LShoulder],
                                                self.points[self.pose_detector.LHip])
        return angle

    def get_shoulders_neck_angle(self):
        # shoulders_neck_pairs = [[2, 1], [1, 5]]
        angle = self.get_angle_between_points(self.points[self.pose_detector.RShoulder], self.points[self.pose_detector.Neck],
                                              self.points[self.pose_detector.LShoulder])
        return angle

    def get_r_neck_shoulder_elbow_angle(self):
        # rNeckShoulderElbow_pairs = [[1, 2], [2, 3]]
        angle = self.get_angle_between_points(self.points[self.pose_detector.Neck], self.points[self.pose_detector.RShoulder],
                                              self.points[self.pose_detector.RElbow])
        return angle

    def get_l_neck_shoulder_elbow_angle(self):
        # lNeckShoulderElbow_pairs = [[1, 5], [5, 6]]
        angle = self.get_angle_between_points(self.points[self.pose_detector.Neck], self.points[self.pose_detector.LShoulder],
                                              self.points[self.pose_detector.LElbow])
        return angle

    def get_r_shoulder_hip_vertical_angle(self):
        angle = self.get_angle_between_vector_and_vertical_axis(
                np.array(self.points[self.pose_detector.RHip]) - np.array(self.points[self.pose_detector.RShoulder]))
        return angle
    def get_l_shoulder_hip_vertical_angle(self):
        angle = self.get_angle_between_vector_and_vertical_axis(
                np.array(self.points[self.pose_detector.LHip]) - np.array(self.points[self.pose_detector.LShoulder]))
        return angle

    def get_mid_shoulder_hip_knee_angle(self, MidHip, MidShoulder):
        angle = self.get_angle_between_vector_and_horizontal_axis(
                np.array(MidHip) - np.array(MidShoulder))
        return angle

    def get_mid_shoulder_hip_vertical_angle(self, MidHip, MidShoulder):
        angle = self.get_angle_between_vector_and_vertical_axis(
                np.array(MidHip) - np.array(MidShoulder))
        return angle


    def get_r_shoulder_elbow_wrist(self):
        # r_shoulder_elbow_wrist_pairs = [[2, 3], [3, 4]]
        angle = self.get_angle_between_points(self.points[self.pose_detector.RShoulder], self.points[self.pose_detector.RElbow],
                                              self.points[self.pose_detector.RWrist])
        return angle

    def get_l_shoulder_elbow_wrist(self):
        # l_shoulder_elbow_wrist_pairs = [[5, 6], [6, 7]]
        angle = self.get_angle_between_points(self.points[self.pose_detector.LShoulder], self.points[self.pose_detector.LElbow],
                                              self.points[self.pose_detector.LWrist])
        return angle

    def get_r_hip_shoulder_elbow_angle(self):
        # rHipShoulderElbow_pairs = [[3, 2], [2, 8]]
        angle = self.get_angle_between_points(self.points[self.pose_detector.RElbow], self.points[self.pose_detector.RShoulder],
                                              self.points[self.pose_detector.RHip])
        return angle

    def get_l_hip_shoulder_elbow_angle(self):
        # lHipShoulderElbow_pairs = [[6, 5], [5, 11]]
        angle = self.get_angle_between_points(self.points[self.pose_detector.LElbow], self.points[self.pose_detector.LShoulder], self.points[self.pose_detector.LHip])
        return angle

    def get_r_shoulder_elbow_wrist_angle(self):
        # rShoulderElbowWrist_pairs = [[2, 3], [3, 4]]
        angle = self.get_angle_between_points(self.points[self.pose_detector.RShoulder], self.points[self.pose_detector.RElbow],
                                              self.points[self.pose_detector.RWrist])
        return angle

    def get_l_shoulder_elbow_wrist_angle(self):
        # lShoulderElbowWrist_pairs = [[5, 6], [6, 7]]
        angle = self.get_angle_between_points(self.points[self.pose_detector.LShoulder], self.points[self.pose_detector.LElbow],
                                              self.points[self.pose_detector.LWrist])
        return angle

    def get_r_hip_shoulder_ear_angle(self):
        # r_hip_shoulder_ear_pairs = [[8, 2], [2, 16]]
        angle = self.get_angle_between_points(self.points[self.pose_detector.RHip], self.points[self.pose_detector.RShoulder],
                                              self.points[self.pose_detector.REar])
        return angle

    def get_l_hip_shoulder_ear_angle(self):
        # l_hip_shoulder_ear_pairs = [[11, 5], [5, 17]]
        angle = self.get_angle_between_points(self.points[self.pose_detector.LHip], self.points[self.pose_detector.LShoulder],
                                              self.points[self.pose_detector.LEar])
        return angle

    def get_r_ear_eye_shoulder_angle(self):
        # l_ear_eye_shoulder_pairs = [[16, 14], [14, 2]]
        angle = self.get_angle_between_points(self.points[self.pose_detector.REye], self.points[self.pose_detector.REar],
                                              self.points[self.pose_detector.RShoulder])
        return angle

    def get_l_ear_eye_shoulder_angle(self):
        # l_ear_eye_shoulder_pairs = [[17, 15], [15, 5]]
        angle = self.get_angle_between_points(self.points[self.pose_detector.LEye], self.points[self.pose_detector.LEar],
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
        # return math.acos(round(np.dot(v1, v2) / (len_v1 * len_v2), 5)) * 180 / math.pi))
        return round(math.acos(round(np.dot(v1, v2) / (len_v1 * len_v2), 5)) * 180 / math.pi, 1)

    def get_angle_between_vector_and_vertical_axis(self, v1):
        if v1.any():
            v2 = np.array([0, 1])
            len_v1 = math.sqrt(v1[0] ** 2 + v1[1] ** 2)
            len_v2 = math.sqrt(v2[0] ** 2 + v2[1] ** 2)
            if len_v1 == 0:
                return None
            return round(math.acos(round(np.dot(v1, v2) / (len_v1 * len_v2), 5)) * 180 / math.pi, 1)
        return None

    def get_vectors_between_points(self, p1, p2, p3, p4):
        v1 = np.array(p2) - np.array(p1)
        v2 = np.array(p4) - np.array(p3)
        return v1, v2

    def get_angle_between_points(self, p1, p2, p3):
        if p1 and p2 and p3:
            v1, v2 = self.get_vectors_between_points(p2, p1, p2, p3)
            angle = self.get_angle_between_lines(v1, v2)
            #import pudb; pu.db
            if angle is None:
                return None
            else:
                return round(angle, 1)
        else:
            return None

    def display_joint_points(self, file_name):
        for point in self.points:
            if point:
                cv2.circle(self.image, point, self.circle_radius, (0, 255, 0), thickness=-1, lineType=cv2.FILLED)
        cv2.imwrite(f'../joints/{file_name}', self.image)

    def draw_lines_between_pairs(self, axis, points,  color, is_point_pixel=False):
        x = []
        y = []
        for point in points:
            if not is_point_pixel:
                x.append(self.points[point][0])
                y.append(self.points[point][1])
            else:
                x.append(point[0])
                y.append(point[1])

        #import pudb; pu.db        
        axis.plot(list(np.array(x) * self.resize_factor) , list(np.array(y) * self.resize_factor) , c=color)
            
    
    def get_blure_image(self):
        #import pudb; pu.db
        img_copy = self.image.copy()
        img = np.array(img_copy)[:, :, ::-1]
        face_blur_provider = FaceBlurring()
        self.blured_image = face_blur_provider.blur_face(self.points, self.pose_detector.Nose, self.pose_detector.LEye, 
                self.pose_detector.REye, self.pose_detector.LEar, self.pose_detector.REar, img)


    def put_text_add_description(self, axis, output_directory, file_name, rule_name, angle, color, x=10, y=30):
        angle_text = round(angle, 1) if angle is not None else None
        axis.text(int(x * self.resize_factor), int(y * self.resize_factor), f'{rule_name}: {angle_text}', color=color)

        text_file = open(f'{output_directory}/log.txt', "a")
        text_file.write(f'Description of {file_name}:\n{self.description}'
                        f'f"*************************************************************************\n')
        text_file.close()

    def put_score_label(self, axis, rule, score, label, x=1):
        axis.text(10 , int(self.image.shape[0] - 10 * x * self.resize_factor) , f'{rule} score: {score}' + '\n' + f'label: {label}')
        if score == label:
            main_label = 'True'
        else:
            main_label = 'False'
        axis.text(10 , int(self.image.shape[0] + 50 * x * self.resize_factor) , f'{main_label}')
        #axis.text(int(self.image.shape[1] - 50) , int(self.image.shape[0] - 10 * x  * self.resize_factor) , f'label: {label}')
   
    def resize(self, image):
       return cv2.resize(image, None, fx = self.resize_factor, fy = self.resize_factor)


