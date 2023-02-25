import argparse
import os
import glob

import cv2
from src.pose_detector import PoseDetector
from src.rosa_rule_provider import RosaRuleProvider

parser = argparse.ArgumentParser()
parser.add_argument('--path', default='../input/test.mov', help='path to input file')
parser.add_argument('--save_path', default='../output', help='path to save output')
parser.add_argument('--frame_rate', default=10, help='video frame rate')

args = parser.parse_args()


def main():
    os.makedirs(args.save_path, exist_ok=True)
    pose_detector = PoseDetector()

    # image
    img_name = '2'
    image = cv2.imread(f'../input/{img_name}.jpg')
    resized_image, dim = pose_detector.resize(image)
    output = pose_detector.forward_model()
    points = pose_detector.get_joint_points(output)
    rosa_rule_provider = RosaRuleProvider(resized_image)
    rosa_rule_provider.display_joint_points(points)
    position_status = rosa_rule_provider.is_correct_position(points, img_name)
    rosa_rule_provider.save_image(img_name, position_status)

    # # video
    # cap = cv2.VideoCapture(args.path)
    # counter = -1
    # while cap.isOpened():
    #     counter += 1
    #     ret, frame = cap.read()
    #     if ret:
    #         if counter % args.frame_rate == 0:
    #             output = pose_detector.forward_model(frame)
    #             points = pose_detector.get_joint_points(output)
    #             pose_detector.draw_skeleton(points, counter)
    #             pose_detector.is_correct(points, counter)
    #     else:
    #         cap.release()
    #         cv2.destroyAllWindows()
    #         img_array = []
    #         for filename in sorted(glob.glob(os.path.join(args.save_path, '*.png'))):
    #             img = cv2.imread(filename)
    #             height, width, layers = img.shape
    #             size = (width, height)
    #             img_array.append(img)
    #
    #         out = cv2.VideoWriter('results.avi', cv2.VideoWriter_fourcc(*'DIVX'), 3, size)
    #
    #         for i in range(len(img_array)):
    #             out.write(img_array[i])
    #         out.release()


if __name__ == '__main__':
    main()