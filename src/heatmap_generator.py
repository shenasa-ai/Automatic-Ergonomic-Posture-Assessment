import os
import cv2
import numpy as np


class HeatmapGenerator:
    """
    Generates an abstract, pose-detector-agnostic risk heatmap overlay on top
    of a posture image (similar to the green/red glow example: safe body
    regions are highlighted green, risky regions are highlighted red/orange).

    This class is fully decoupled from any specific scoring logic. It only
    needs:
      - the original image,
      - the list of detected joint points,
      - the pose_detector object (used only to look up named joint indices,
        e.g. pose_detector.Neck, pose_detector.RHip, ...),
      - a dict of {region_name: score}.

    It can be reused by any rule provider / scoring pipeline, regardless of
    which specific rules produced the scores, as long as the region names
    used in `scores` exist in `region_joints`.
    """

    # Generic body regions -> joint attribute names that locate them.
    # A region is drawn if at least one of its joints was detected.
    DEFAULT_REGION_JOINTS = {
        'neck': ['Neck', 'Nose'],
        'back': ['RShoulder', 'LShoulder', 'RHip', 'LHip'],
        'trunk': ['RShoulder', 'LShoulder', 'RHip', 'LHip'],
        'chair': ['RHip', 'RKnee', 'RAnkle', 'LHip', 'LKnee', 'LAnkle'],
        'arm': ['RElbow', 'RShoulder', 'LElbow', 'LShoulder'],
        'monitor': ['REar', 'LEar', 'Nose'],
    }

    # score -> BGR color. 1 = safe (green), 2 = moderate (orange), 3 = risk (red)
    DEFAULT_SCORE_COLORS = {
        1: (0, 200, 0),
        2: (0, 165, 255),
        3: (0, 0, 255),
    }

    def __init__(
            self,
            region_joints=None,
            score_colors=None,
            blur_ksize=41,
            alpha=0.45
    ):
        self.region_joints = region_joints or self.DEFAULT_REGION_JOINTS
        self.score_colors = score_colors or self.DEFAULT_SCORE_COLORS
        self.blur_ksize = blur_ksize
        self.alpha = alpha

    def _score_to_color(self, score):
        if score is None:
            return None
        levels = sorted(self.score_colors.keys())
        clamped = max(min(int(round(score)), levels[-1]), levels[0])
        return self.score_colors[clamped]

    def _region_points(self, points, pose_detector, joint_names):
        coords = []

        for name in joint_names:
            idx = getattr(pose_detector, name, None)
            if idx is None:
                continue

            if idx >= len(points):
                continue

            pt = points[idx]
            if pt is not None:
                coords.append(tuple(pt))

        return coords
    def generate(self, image, points, pose_detector, scores,
                 output_path=None, file_name=None):
        """
        Builds a soft, colored "risk glow" overlay and blends it onto the
        original image. Does not mutate `image`.

        image: BGR numpy array
        points: list of (x, y) joint coordinates, indexable by pose_detector's
                named joint attributes
        pose_detector: object exposing named joint indices
        scores: dict {region_name: score}; region_name should be a key in
                self.region_joints (regions with no matching joints or a
                None score are simply skipped)
        output_path / file_name: if both given, saves the result as
                f'{output_path}/{file_name}_heatmap.JPG'

        Returns the heatmap image (BGR numpy array).
        """
        h, w = image.shape[:2]
        color_layer = np.zeros_like(image, dtype=np.float32)
        mask = np.zeros((h, w), dtype=np.float32)

        for region, score in scores.items():
            color = self._score_to_color(score)
            joint_names = self.region_joints.get(region)
            if color is None or not joint_names:
                continue
            joint_points = self._region_points(
                points,
                pose_detector,
                joint_names
            )

            if len(joint_points) == 0:
                continue

            region_mask = np.zeros((h, w), dtype=np.float32)

            joint_radius = 18  # try 12-20

            for pt in joint_points:
                cv2.circle(
                    region_mask,
                    pt,
                    joint_radius,
                    1.0,
                    thickness=-1
                )
            mask = np.maximum(mask, region_mask)
            for c in range(3):
                color_layer[:, :, c] = np.where(region_mask > 0, color[c], color_layer[:, :, c])

        k = self.blur_ksize if self.blur_ksize % 2 == 1 else self.blur_ksize + 1
        mask = cv2.GaussianBlur(mask, (k, k), 0)
        color_layer = cv2.GaussianBlur(color_layer, (k, k), 0)

        mask_3c = np.repeat(mask[:, :, None], 3, axis=2)
        blended = image.astype(np.float32) * (1 - self.alpha * mask_3c) + color_layer * (self.alpha * mask_3c)
        heatmap_image = np.clip(blended, 0, 255).astype(np.uint8)

        if output_path and file_name:
            os.makedirs(output_path, exist_ok=True)
            cv2.imwrite(f'{output_path}/{file_name}_heatmap.JPG', heatmap_image)

        return heatmap_image
