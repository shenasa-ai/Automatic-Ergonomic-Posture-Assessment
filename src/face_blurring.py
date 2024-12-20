import cv2
import math
import numpy as np


class FaceBlurring:
    def round_tuple_elements(self, input_tuple):
        if input_tuple:
            input_list = list(input_tuple)
            input_list[0] = round(input_list[0])
            input_list[1] = round(input_list[1])
            return tuple(input_list)

    def blur_face(self, points, nose_idx, l_eye_idx, r_eye_idx, l_ear_idx, r_ear_idx, pil_im):
        image = np.array(pil_im)
        # adding a condition for checking if any point has not been detected exit the loop
        if (((points[r_eye_idx] or points[l_ear_idx]) or
             (points[r_ear_idx] or points[l_eye_idx])) and points[nose_idx]):
            # if right eye or left eye exists and nose is found
            if (points[r_eye_idx] or points[l_eye_idx]) and points[nose_idx]:
                # rounding the Points
                points[r_eye_idx] = self.round_tuple_elements(points[r_eye_idx])
                points[l_eye_idx] = self.round_tuple_elements(points[l_eye_idx])

                # if right eye or left eye exists and nose is found
                points[nose_idx] = self.round_tuple_elements(points[nose_idx])

                # making coordination of one eye the same with other eye
                if points[l_eye_idx]:
                    points[r_eye_idx] = tuple(list(points[l_eye_idx]))
                else:
                    points[l_eye_idx] = tuple(list(points[r_eye_idx]))
                # calculation radius of circle
                radius = math.sqrt(((points[l_eye_idx][0] - points[nose_idx][0]) ** 2) + ((points[l_eye_idx][1] -
                                                                                           points[nose_idx][1]) ** 2))

                if points[l_eye_idx][0] == points[nose_idx][0] and points[l_eye_idx][1] == points[nose_idx][1]:
                    radius = 10
                else:
                    radius = int(round(radius))
                # center of the circle
                center = (round(points[nose_idx][0])), round(points[nose_idx][1])

                # if righ ear or left one exists and nose is found
            elif (points[r_ear_idx] or points[l_ear_idx]) and points[nose_idx]:
                # making coordination of one ear the same with other ear
                if points[l_ear_idx]:
                    points[r_ear_idx] = tuple(list(points[l_ear_idx]))
                else:
                    points[l_ear_idx] = tuple(list(points[r_ear_idx]))

                # rounding the Points
                points[r_ear_idx] = self.round_tuple_elements(points[r_ear_idx])
                points[l_ear_idx] = self.round_tuple_elements(points[l_ear_idx])
                points[nose_idx] = self.round_tuple_elements(points[nose_idx])

                # calculation radius of circle
                radius = math.sqrt(((points[l_ear_idx][0] - points[nose_idx][0]) ** 2) +
                                   ((points[l_ear_idx][1] - points[nose_idx][1]) ** 2))

                if points[l_ear_idx][0] == points[nose_idx][0] and points[l_ear_idx][1] == points[nose_idx][1]:
                    radius = 10
                else:
                    radius = int(round(radius))
                # center of the circle
                center = (round(points[nose_idx][0]), round(points[nose_idx][1]))
                # bluring the face
            x = int(round(points[nose_idx][0])) - radius if (int(round(points[nose_idx][0])) - radius) > 0 else 0
            y = int(round(points[nose_idx][1])) - radius if (int(round(points[nose_idx][1])) - radius) > 0 else 0
            w = radius * 3
            h = radius * 3
            roi = image[y:y + h, x:x + w]
            # applying a gaussian blur over this new rectangle area
            if len(roi) != 0:
                roi = cv2.GaussianBlur(roi, (23, 23), 30)
                # impose this blurred image on original image to get final image
                image[y:y + roi.shape[0], x:x + roi.shape[1]] = roi
        return image


# if __name__ == '__main__':
#     pose_detector = OpenPoseDetector()
#     file_name = os.fsdecode("../input/front/20230227_160945.jpg")
#     image = cv2.imread(file_name)
#     resized_image = pose_detector.preprocess_image(image)
#     points = pose_detector.get_joint_points()
#     face_blur = FaceBlurProvider()
#     blured_image = face_blur.blur_face(points, 0, 15, 14, 17, 16, resized_image)
#     img = np.array(blured_image)[:, :, ::-1]
#     plt.imshow(img)
#     plt.axis('off')
#     plt.savefig(f'../output/blured_images/ttt.jpg')
