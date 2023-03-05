from abc import ABCMeta, abstractmethod


class PoseDetector(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def get_joint_points(self) -> []:
        pass

    @abstractmethod
    def preprocess_image(self, image):
        pass
