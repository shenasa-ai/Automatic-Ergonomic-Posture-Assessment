import argparse
import os
import cv2
import pandas as pd

from pose_detector import PoseDetector
from openpose_detector import OpenPoseDetector
from mediapipe_pose_detector import MediapipePoseDetector
from openpifpaf_pose_detector import OpenpifpafPoseDetector
from rule_provider import RuleProvider

deep_model = "Openpifpaf" #"Mediapipe" #"Openpifpaf"  #"openpose" 

parser = argparse.ArgumentParser()
parser.add_argument('--input_path', default='/home/ali/Desktop/Automatic-Ergonomic-Posture-Assessment/input/main_input/front', help='path to input directory')
parser.add_argument('--output_path', default=f'../output/{deep_model}_front', help='path to output directory')
parser.add_argument('--frame_rate', default=10, help='video frame rate')
parser.add_argument('--front_labels_path', default='front_labels.csv', help='front labels path')
parser.add_argument('--side_labels_path', default='side_labels.csv', help='side labels path')


args = parser.parse_args()


def assess_posture(root_dir, camera_view_point, pose_detector, rosa_rule_provider):
    for file in sorted(os.listdir(root_dir)):
        file_name = os.fsdecode(file)
        #file_name = os.path.splitext(file_name)[0]
        image = cv2.imread(f'{root_dir}/{file_name}')
        # if file == 'side_30.jpg':
            # import pudb; pu.db
        resized_image = pose_detector.preprocess_image(image)
        points = pose_detector.get_joint_points()
        position_status = rosa_rule_provider.get_posture_status(resized_image, points, file_name, camera_view_point, args.output_path, args.front_labels_path, args.side_labels_path)
        #rosa_rule_provider.save_image(position_status, args.output_path, file_name)
        print('*******************************************************************************************')
    pd.DataFrame({k: pd.Series(v) for k, v in rosa_rule_provider.result.items()}).to_csv(f'./../output/pred_{deep_model}_front.csv', index=False)


def main():
    os.makedirs(args.output_path, exist_ok=True)

    input_directory =   os.fsencode(args.input_path).decode("utf-8")
    pose_detector = None
    if deep_model == "openpose":
        pose_detector = OpenPoseDetector()
    elif deep_model == "Openpifpaf":
        pose_detector = OpenpifpafPoseDetector()
    elif deep_model == "Mediapipe":
        pose_detector = MediapipePoseDetector()
    rosa_rule_provider = RuleProvider(pose_detector)
    if os.path.exists(f'{args.output_path}/log.txt'):
        os.remove(f'{args.output_path}/log.txt')
    sub_dirs = [x[0] for x in os.walk(input_directory)]
    for subdir in sub_dirs:
        if 'side' in subdir:
            assess_posture(subdir, 'side', pose_detector, rosa_rule_provider)
        elif 'front' in subdir:
            assess_posture(subdir, 'front', pose_detector, rosa_rule_provider)


if __name__ == '__main__':
    main()
