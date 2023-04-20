import math
import os
import numpy as np
from pose_detector import PoseDetector
from matplotlib import pyplot as plt
from face_blurring import FaceBlurring
import cv2
import matplotlib.pyplot as plt


class RosaRuleProvider:
    def __init__(self, pose_detector):
        self.image = None
        self.points = None
        self.pose_detector = pose_detector
        self.circle_radius = 3
        self.line_thickness = 2
        self.description = ""
        #self.repetitive_pairs = []
        self.camera_view_point = "front"

    def get_posture_status(self, image, points, file_name, view_point, output_path, draw_joint_points=True):
        self.description = ""
        self.image = image
        self.points = points
        self.camera_view_point = view_point
        self.output_path = output_path
        if view_point == "front":
            self.figure_front, self.axs_front = plt.subplots(1, 2)
            self.figure_front.tight_layout()
            armrest_score = 1
            phone_score = 1
        else:
            self.figure_side, self.axs_side = plt.subplots(1, 3)
            self.figure_side.tight_layout()
            chair_score = 1
            backrest_score = 1
            monitor_score = 1
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

            # Monitor (0/6)
            monitor_score = self.get_monitor_score()
            self.figure_side.savefig(f'{self.output_path}/{self.file_name}.JPG')
            #self.figure_side.close()
            plt.close()


        else:
            #import pudb; pu.db

            # Armrest (3/4)
            armrest_score = self.get_armrest_score()


            # Telephone (1/3)
            phone_score = self.get_phone_score()
            self.figure_front.savefig(f'{self.output_path}/{self.file_name}.JPG')
            plt.close()

        #import pudb; pu.db
        #if  (chair_score > 1 if chair_score else True)  or (armrest_score > 1 if armrest_score else True) or (backrest_score > 1 if backrest_score else True) or (monitor_score > 1 if monitor_score else True) or (phone_score > 1 if phone_score else True):
#       # armrest_score > 1 or backrest_score > 1 or monitor_score > 1 or phone_score > 1:
        #    posture_status = False
        #else:
        #    posture_status = True

        if self.camera_view_point == 'side':
            print(f'chair score is: {chair_score}\n'
                  f'backrest score is: {backrest_score}\n'
                  f'monitor score is: {monitor_score}\n')
        else:
            print(f'armrest score is: {armrest_score}\n'
                    f'phone score is: {phone_score}\n')

        return posture_status

    def get_chair_score(self):
        self.get_blure_image()
        axis = self.axs_side[0]
        self.axs_side[0].set_title('Chair Score')
        self.axs_side[0].axis('off')
        axis.imshow(self.blured_image)

        chair_score = 1

        r_hip_knee_ankle_angle = self.get_r_hip_knee_ankle_angle()
        l_hip_knee_ankle_angle = self.get_l_hip_knee_ankle_angle()

        if r_hip_knee_ankle_angle or l_hip_knee_ankle_angle:
            if r_hip_knee_ankle_angle:
                r_hip_knee_ankle_points = [self.pose_detector.RHip, self.pose_detector.RKnee, self.pose_detector.RAnkle]
                if r_hip_knee_ankle_angle < 80:
                    chair_score = 2
                    c = 'red'
                    self.description = self.description + f'Chair is TOO LOW - right hip_knee_ankle angle: {r_hip_knee_ankle_angle}\n'
                elif r_hip_knee_ankle_angle > 100:
                    chair_score = 2
                    c = 'red'
                    self.description = self.description + f'Chair is TOO HIGH - right hip_knee_ankle angle: {r_hip_knee_ankle_angle}\n'
                else:
                    c = 'green'
                    self.description = self.description + \
                                       f'Right knee status is in CORRECT POSTURE - right hip_knee_ankle angle: {r_hip_knee_ankle_angle}\n'
                self.draw_lines_between_pairs(axis, r_hip_knee_ankle_points, c)
                self.put_text_add_description(axis, self.output_path, self.file_name, 'Right leg angle', r_hip_knee_ankle_angle, c)

            if l_hip_knee_ankle_angle:
                l_hip_knee_ankle_points = [self.pose_detector.LHip, self.pose_detector.LKnee, self.pose_detector.LAnkle]
                if l_hip_knee_ankle_angle < 80:
                    chair_score = 2
                    c = 'red'
                    self.description = self.description + f'Chair is TOO LOW - left hip_knee_ankle angle: {l_hip_knee_ankle_angle}\n'
                elif l_hip_knee_ankle_angle > 100:
                    chair_score = 2
                    c = 'red'
                    self.description = self.description + f'Chair is TOO HIGH - left hip_knee_ankle angle: {l_hip_knee_ankle_angle}\n'
                else:
                    c = 'green'
                    self.description = self.description + \
                                       f'Left knee status is in CORRECT POSTURE - left hip_knee_ankle angle: {l_hip_knee_ankle_angle}\n'
                self.draw_lines_between_pairs(axis, l_hip_knee_ankle_points, c)
                self.put_text_add_description(axis, self.output_path, self.file_name, 'Left leg angle', l_hip_knee_ankle_angle, c, 10 , 60)
        else:
            print("Not Enough Info")
            self.put_text_add_description(axis, self.output_path, self.file_name, "Not Enough Info", None, 'red')
            chair_score = None
        #import pudb; pu.db
        return chair_score

    def get_armrest_score(self):
        armrest_score = 1
        self.get_blure_image()
        axis = self.axs_front[0]
        self.axs_front[0].set_title('Armrest Score')
        self.axs_front[0].axis('off')
        axis.imshow(self.blured_image)
        #if True: #self.camera_view_point == "front": ###################################################
        if hasattr(self.pose_detector, 'Neck'):
            shoulders_neck_angle = self.get_shoulders_neck_angle()
            shoulders_neck_points = [self.pose_detector.Neck, self.pose_detector.RShoulder, self.pose_detector.LShoulder]
            if shoulders_neck_angle:
                if shoulders_neck_angle < 160:
                    armrest_score = 2
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
                self.put_text_add_description(axis, self.output_path, self.file_name, "Not Enough Info aboud sholders being shrogged ", None, 'red', 10, 90)
        else:
            print("Not Enough Info")
            self.put_text_add_description(axis, self.output_path, self.file_name, "Not Enough Info aboud sholders being shrogged ", None, 'red', 10, 60)
            #import pudb; pu.db
        #########################################################################################
        status = 'vertical'
        if status == 'vertical':

            r_shoulder_elbow_angle = self.get_r_shoulder_elbow_angle()
            l_shoulder_elbow_angle = self.get_l_shoulder_elbow_angle()
            
            r_shoulder_elbow_points = [self.points[self.pose_detector.RElbow], self.points[self.pose_detector.RShoulder], 
                    [self.points[self.pose_detector.RShoulder][0], self.points[self.pose_detector.RShoulder][1] + 50]]

            l_shoulder_elbow_points = [self.points[self.pose_detector.LElbow], self.points[self.pose_detector.LShoulder],
                                        [self.points[self.pose_detector.LShoulder][0], self.points[self.pose_detector.LShoulder][1] + 50]]
            if r_shoulder_elbow_angle or l_shoulder_elbow_angle:
                if r_shoulder_elbow_angle is not None:
                    if r_shoulder_elbow_angle > 20:
                        armrest_score += 1
                        c = 'red'
                        self.description = self.description + f'Right elbow is NOT INLINE with right shoulder - ' \
                                                              f'Angle between right shoulder_elbow and vertical axis: ' \
                                                              f'{r_shoulder_elbow_angle}\n'
                    else:
                        c = 'green'
                        self.description = self.description + f'Right elbow is INLINE with right shoulder - ' \
                                                              f'Angle between right shoulder_elbow and vertical axis: ' \
                                                              f'{r_shoulder_elbow_angle}\n'
                    self.draw_lines_between_pairs(axis, r_shoulder_elbow_points, c, True)
                    self.put_text_add_description(axis, self.output_path, self.file_name, 'Right elbow angle', r_shoulder_elbow_angle, c)

                if l_shoulder_elbow_angle is not None:
                    if l_shoulder_elbow_angle > 20:
                        armrest_score += 1
                        c = 'red'
                        self.description = self.description + f'Left elbow is NOT INLINE with left shoulder - ' \
                                                              f'Angle between left shoulder_elbow and vertical axis:' \
                                                              f' {l_shoulder_elbow_angle}\n'
                    else:
                        c = 'green'
                        self.description = self.description + f'Left elbow is INLINE with left shoulder - ' \
                                                              f'Angle between left shoulder_elbow and vertical axis: ' \
                                                              f'{l_shoulder_elbow_angle}\n'
                    self.draw_lines_between_pairs(axis, l_shoulder_elbow_points, c, True)
                    self.put_text_add_description(axis, self.output_path, self.file_name, 'Left elbow angle', l_shoulder_elbow_angle, c, 10, 90)

            else:

                print("Not Enough Info")
                self.put_text_add_description(axis, self.output_path, self.file_name, "Not Enough Info boud sholders being wide", None, 'red')
                armrest_score = None
        #import pudb; pu.db
        #plt.show()
        return armrest_score



        #        r_neck_shoulder_elbow = self.get_r_neck_shoulder_elbow_angle()
        #        r_neck_shoulder_elbow_points = [[self.pose_detector.Neck, self.pose_detector.RShoulder],
        #                                        [self.pose_detector.RShoulder, self.pose_detector.RElbow]]
        #        l_neck_shoulder_elbow = self.get_l_neck_shoulder_elbow_angle()
        #        l_neck_shoulder_elbow_points = [[self.pose_detector.Neck, self.pose_detector.LShoulder],
        #                                        [self.pose_detector.LShoulder, self.pose_detector.LElbow]]

        #        if r_neck_shoulder_elbow:
        #            if r_neck_shoulder_elbow > 120:
        #                armrest_score += 1
        #                self.draw_lines_between_pairs(r_neck_shoulder_elbow_points, False)
        #                self.description = self.description + f'TOO WIDE right elbow - ' \
        #                                                      f'right neck_shoulder_elbow angle: {r_neck_shoulder_elbow}\n'
        #            else:
        #                self.draw_lines_between_pairs(r_neck_shoulder_elbow_points)
        #                self.description = self.description + f'Right elbow is NOT TOO WIDE - ' \
        #                                                      f'right neck_shoulder_elbow angle: {r_neck_shoulder_elbow}\n'

        #        if l_neck_shoulder_elbow:
        #            if l_neck_shoulder_elbow > 120:
        #                armrest_score += 1
        #                self.draw_lines_between_pairs(l_neck_shoulder_elbow_points, False)
        #                self.description = self.description + f'TOO WIDE left elbow - ' \
        #                                                      f'left neck_shoulder_elbow angle: {l_neck_shoulder_elbow}\n'
        #            else:
        #                self.draw_lines_between_pairs(l_neck_shoulder_elbow_points)
        #                self.description = self.description + f'Left elbow is NOT TOO WIDE - ' \
        #                                                      f'left neck_shoulder_elbow angle: {l_neck_shoulder_elbow}\n'
        #    else:
        #        print("Not Enough Info")
        #        self.put_text_add_description(self.output_path, self.file_name, "Not Enough Info aboud sholders being wide", None, 'red')

        #
        #return armrest_score

    def get_backrest_score(self):
        self.get_blure_image()
        axis = self.axs_side[1]
        self.axs_side[1].set_title('Backrest Score')
        self.axs_side[1].axis('off')
        axis.imshow(self.blured_image)
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
                if mid_shoulder_hip_knee < 95:
                    backrest_score = 2
                    c = 'red'
                    self.description = self.description + f'Back rest is BENT FORWARD from right side - ' \
                                                          f'right shoulder_hip_knee angle: {mid_shoulder_hip_knee}\n'
                elif mid_shoulder_hip_knee > 110:
                    backrest_score = 2
                    self.description = self.description + f'Back rest is BENT BACKWARD from right side - ' \
                                                          f'right shoulder_hip_knee angle: {mid_shoulder_hip_knee}\n'
                    c = 'red'
                else:
                    c = 'green'
                    self.description = self.description + f'Back rest is NORMAL from right side - ' \
                                                          f'right shoulder_hip_knee angle: {mid_shoulder_hip_knee}\n'
            
            self.draw_lines_between_pairs(axis, mid_shoulder_hip_knee_points, c, True)
            self.put_text_add_description(axis, self.output_path, self.file_name, 'back angle', mid_shoulder_hip_knee, c, 10, 30)

        else:
            print("Not Enough Info")
            self.put_text_add_description(axis, self.output_path, self.file_name, "Not Enough Info", None, 'red')
            backrest_score = None
        #import pudb; pu.db
        return backrest_score
        

        
        #if self.camera_view_point == "front":
        #    shoulders_neck_angle = self.get_shoulders_neck_angle()
        #    if shoulders_neck_angle:
        #        if shoulders_neck_angle < 160:
        #            backrest_score += 1
        #return backrest_score

    def get_monitor_score(self):
        monitor_score = 1
        self.get_blure_image()
        axis = self.axs_side[2]
        self.axs_side[2].set_title('monitor_score')
        self.axs_side[2].axis('off')
        axis.imshow(self.blured_image)
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
        else:
            print("Not Enough Info")
            self.put_text_add_description(axis, self.output_path, self.file_name, "Not Enough Info", None, 'red')
            monitor_score = None
        
        #import pudb; pu.db
        return monitor_score 





            #r_ear_eye_shoulder_angle = self.get_r_ear_eye_shoulder_angle()
            #r_ear_eye_shoulder_points = [[self.pose_detector.REye, self.pose_detector.REar],
            #                             [self.pose_detector.REar, self.pose_detector.RShoulder]]
            #if r_ear_eye_shoulder_angle:
            #    if r_ear_eye_shoulder_angle < 75:
            #        self.draw_lines_between_pairs(r_ear_eye_shoulder_points, False)
            #        self.description = self.description + f'Neck is BENT DOWNWARD from side view - ' \
            #                                              f'right ear_eye_shoulder angle: {r_ear_eye_shoulder_angle} \n'
            #    elif r_ear_eye_shoulder_angle > 125:
            #        self.draw_lines_between_pairs(r_ear_eye_shoulder_points, False)
            #        self.description = self.description + f'Neck is BENT UPWARD from side view - ' \
            #                                              f'right ear_eye_shoulder angle: {r_ear_eye_shoulder_angle} \n'

            #l_ear_eye_shoulder_angle = self.get_l_ear_eye_shoulder_angle()
            #l_ear_eye_shoulder_points = [[self.pose_detector.LEye, self.pose_detector.LEar],
            #                             [self.pose_detector.LEar, self.pose_detector.LShoulder]]
            #if l_ear_eye_shoulder_angle:
            #    if l_ear_eye_shoulder_angle < 75:
            #        self.draw_lines_between_pairs(l_ear_eye_shoulder_points, False)
            #        self.description = self.description + f'Neck is BENT DOWNWARD from side view - ' \
            #                                              f'left ear_eye_shoulder angle: {l_ear_eye_shoulder_angle} \n'
            #    elif l_ear_eye_shoulder_angle > 125:
            #        self.draw_lines_between_pairs(l_ear_eye_shoulder_points, False)
            #        self.description = self.description + f'Neck is BENT UPWARD from side view - ' \
            #                                              f'left ear_eye_shoulder angle: {l_ear_eye_shoulder_angle} \n'

        #if self.camera_view_point == "front":
        #    # neck rule front
        #    if self.points[self.pose_detector.LEye] and self.points[self.pose_detector.REye] and \
        #            self.points[self.pose_detector.LShoulder] and self.points[self.pose_detector.RShoulder]:
        #        v1 = np.array(self.points[self.pose_detector.LEye]) - np.array(self.points[self.pose_detector.REye])
        #        v2 = np.array(self.points[self.pose_detector.LShoulder]) - np.array(self.points[self.pose_detector.RShoulder])
        #        angle_between_shoulders_and_eyes = self.get_angle_between_lines(v1, v2)
        #        two_eyes_vertical_axis_angle = self.get_angle_between_vector_and_vertical_axis(
        #            np.array(self.points[self.pose_detector.LEye]) - np.array(self.points[self.pose_detector.REye]))
        #        two_shoulders_vertical_axis_angle = self.get_angle_between_vector_and_vertical_axis(
        #            np.array(self.points[self.pose_detector.LShoulder]) - np.array(self.points[self.pose_detector.RShoulder]))
        #        two_eyes_points = [[self.pose_detector.LEye, self.pose_detector.REye]]
        #        two_shoulders_points = [[self.pose_detector.LShoulder, self.pose_detector.RShoulder]]
        #        if math.fabs(two_eyes_vertical_axis_angle - two_shoulders_vertical_axis_angle) > 30:
        #            if 30 < angle_between_shoulders_and_eyes < 90:
        #                self.description = self.description + f'Neck is BENT LEFTWARD from front view - ' \
        #                                                      f'angle_between_shoulders_and_eyes: {angle_between_shoulders_and_eyes} \n'
        #            elif 120 < angle_between_shoulders_and_eyes < 150:
        #                self.description = self.description + f'Neck is BENT RIGHTWARD from front view - ' \
        #                                                      f'angle_between_shoulders_and_eyes: {angle_between_shoulders_and_eyes} \n'
        #            self.draw_lines_between_pairs(two_eyes_points, False)
        #            self.draw_lines_between_pairs(two_shoulders_points, False)
        #        else:
        #            self.description = self.description + f'Neck is not BENT RIGHTWARD or leftward - ' \
        #                                                  f'angle_between_shoulders_and_eyes: {angle_between_shoulders_and_eyes} \n'
        #            self.draw_lines_between_pairs(two_eyes_points)
        #            self.draw_lines_between_pairs(two_shoulders_points)
        #return monitor_score

    def get_phone_score(self):
        phone_score = 1
        self.get_blure_image()
        axis = self.axs_front[1]
        self.axs_front[1].set_title('Phone Score')
        self.axs_front[1].axis('off')
        axis.imshow(self.blured_image)

        #r_hip_shoulder_ear_angle = self.get_r_hip_shoulder_ear_angle()
        #r_hip_shoulder_ear_points = [self.pose_detector.RHip, self.pose_detector.RShoulder, self.pose_detector.REar]
        #l_hip_shoulder_ear_angle = self.get_l_hip_shoulder_ear_angle()
        #l_hip_shoulder_ear_points = [self.pose_detector.LHip, self.pose_detector.LShoulder, self.pose_detector.LEar]

        #import pudb; pu.db
        angle_between_shoulders_and_eyes = self.get_angle_between_shoulders_and_eyes()


        ##if self.points[self.pose_detector.LEye] and self.points[self.pose_detector.REye] and \
        ##        self.points[self.pose_detector.LShoulder] and self.points[self.pose_detector.RShoulder]:
        ##    v1 = np.array(self.points[self.pose_detector.LEye]) - np.array(self.points[self.pose_detector.REye])
        ##    v2 = np.array(self.points[self.pose_detector.LShoulder]) - np.array(self.points[self.pose_detector.RShoulder])
        ##    angle_between_shoulders_and_eyes = self.get_angle_between_lines(v1, v2)
            #two_eyes_vertical_axis_angle = self.get_angle_between_vector_and_vertical_axis(
            #    np.array(self.points[self.pose_detector.LEye]) - np.array(self.points[self.pose_detector.REye]))
            #two_shoulders_vertical_axis_angle = self.get_angle_between_vector_and_vertical_axis(
            #    np.array(self.points[self.pose_detector.LShoulder]) - np.array(self.points[self.pose_detector.RShoulder]))
        two_eyes_points = [self.pose_detector.LEye, self.pose_detector.REye]
        two_shoulders_points = [self.pose_detector.LShoulder, self.pose_detector.RShoulder]
        #    if math.fabs(two_eyes_vertical_axis_angle - two_shoulders_vertical_axis_angle) > 30:
        if 30 < angle_between_shoulders_and_eyes < 90:
            c = 'red'
            self.description = self.description + f'Neck is BENT LEFTWARD from front view - ' \
                                                  f'angle_between_shoulders_and_eyes: {angle_between_shoulders_and_eyes} \n'
        elif 120 < angle_between_shoulders_and_eyes < 150:
            c = 'red'
            self.description = self.description + f'Neck is BENT RIGHTWARD from front view - ' \
                                                  f'angle_between_shoulders_and_eyes: {angle_between_shoulders_and_eyes} \n'
        #self.draw_lines_between_pairs(two_eyes_points, False)
        #self.draw_lines_between_pairs(two_shoulders_points, False)
        else:
            c = 'green'
            self.description = self.description + f'Neck is not BENT RIGHTWARD or leftward - ' \
                                                  f'angle_between_shoulders_and_eyes: {angle_between_shoulders_and_eyes} \n'
        
        self.draw_lines_between_pairs(axis, two_eyes_points, c)
        self.draw_lines_between_pairs(axis, two_shoulders_points, c)
        self.put_text_add_description(axis, self.output_path, self.file_name, 'angle between 2 \n eyes and 2 shoulders', angle_between_shoulders_and_eyes, c)
        #return monitor_score

        shoulders_distance = None
        wrists_distance = None
        wrists_horizontal_axis_angle = None
        two_wrists_points = [[self.pose_detector.LWrist, self.pose_detector.RWrist]]

        if self.camera_view_point == "front":
            if self.points[self.pose_detector.RShoulder] and self.points[self.pose_detector.LShoulder]:
                shoulders_distance = self.calculate_distance_between_two_points(self.points[self.pose_detector.RShoulder],
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



    def get_angle_between_shoulders_and_eyes(self):
        if self.points[self.pose_detector.LEye] and self.points[self.pose_detector.REye] and \
                self.points[self.pose_detector.LShoulder] and self.points[self.pose_detector.RShoulder]:
            v1 = np.array(self.points[self.pose_detector.LEye]) - np.array(self.points[self.pose_detector.REye])
            v2 = np.array(self.points[self.pose_detector.LShoulder]) - np.array(self.points[self.pose_detector.RShoulder])
            angle = self.get_angle_between_lines(v1, v2)
            if angle is None:
                return None
            else:
                return round(angle, 2)
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

    def get_r_shoulder_elbow_angle(self):
        angle = self.get_angle_between_vector_and_vertical_axis(
                        np.array(self.points[self.pose_detector.RElbow]) - np.array(self.points[self.pose_detector.RShoulder]))
        return angle

    def get_l_shoulder_elbow_angle(self):
        angle = self.get_angle_between_vector_and_vertical_axis(
                        np.array(self.points[self.pose_detector.LElbow]) - np.array(self.points[self.pose_detector.LShoulder]))
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

    def get_mid_shoulder_hip_knee_angle(self, MidHip, MidShoulder):
        angle = self.get_angle_between_vector_and_horizontal_axis(
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
            #import pudb; pu.db
            if angle is None:
                return None
            else:
                return round(angle, 2)
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

                
        axis.plot(x, y, c=color)
            
    
    def get_blure_image(self):
        #import pudb; pu.db
        img_copy = self.image.copy()
        img = np.array(img_copy)[:, :, ::-1]
        face_blur_provider = FaceBlurring()
        self.blured_image = face_blur_provider.blur_face(self.points, self.pose_detector.Nose, self.pose_detector.LEye, 
                self.pose_detector.REye, self.pose_detector.LEar, self.pose_detector.REar, img)


    def put_text_add_description(self, axis, output_directory, file_name, rule_name, angle, color, x=10, y=30):
        axis.text(x, y, f'{rule_name}: {angle}', color=color)

        text_file = open(f'{output_directory}/log.txt', "a")
        text_file.write(f'Description of {file_name}:\n{self.description}'
                        f'f"*************************************************************************\n')
        text_file.close()
