"""
Paper Image Generator for Ergonomic Posture Assessment.

This script processes high-resolution input images and generates publication-ready
diagrams for scientific papers. NO TEXT is drawn on any generated image:
  1. Joint Extracted Image: Detected body keypoints with high-res color-coded markers.
  2. Vector Extracted Image: Kinematic skeleton vectors connecting anatomical joints.
  3. Angle Calculated Image: Posture angles (Knee, Trunk, Neck, Elbow) shown only with
     geometric arcs and reference axes (arc color = ergonomic status: green / orange).
     Numeric angle values are printed to the console instead of the image.
  4. (Optional) Combined Multi-Panel Figure: Composite figure summarizing all steps.

The model used is automatically read from `src/main.py` (deep_model) or can be specified via CLI.
All visual annotations are dynamically scaled to the original high-resolution image.

The background photo (the person) can be faded with --bg_dim / --bg_mode so that
joints, vectors and angle annotations stand out more clearly.
"""

import os
import sys
import math
import argparse
import re
import cv2
import numpy as np
from PIL import Image
from pose_detector import PoseDetector
from openpose_detector import OpenPoseDetector
from mediapipe_pose_detector import MediapipePoseDetector
from openpifpaf_pose_detector import OpenpifpafPoseDetector
from rule_provider import RuleProvider
from face_blurring import FaceBlurring  # adjust the module name if your file is named differently

# Add src to python path for importing detectors
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, "")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)


def get_default_model_from_main() -> str:
    """Reads the active deep_model variable from src/main.py."""
    main_py_path = os.path.join(src_dir, "main.py")
    if os.path.exists(main_py_path):
        try:
            with open(main_py_path, "r", encoding="utf-8") as f:
                content = f.read()
            match = re.search(r'^\s*deep_model\s*=\s*["\']([^"\']+)["\']', content, re.MULTILINE)
            if match:
                return match.group(1)
        except Exception as e:
            print(f"Warning: Could not parse deep_model from main.py: {e}")
    return "Openpifpaf"


def load_pose_detector(model_name: str):
    """Instantiates the appropriate PoseDetector based on model_name."""
    name_lower = model_name.strip().lower()
    if name_lower in ["openpifpaf", "pifpaf"]:
        from openpifpaf_pose_detector import OpenpifpafPoseDetector
        return OpenpifpafPoseDetector(), "Openpifpaf"
    elif name_lower in ["mediapipe", "mp"]:
        from mediapipe_pose_detector import MediapipePoseDetector
        return MediapipePoseDetector(), "Mediapipe"
    elif name_lower in ["yolo", "yolov8"]:
        from yolo_pose_detector import YoloPoseDetector
        return YoloPoseDetector(), "Yolo"
    elif name_lower in ["openpose"]:
        from openpose_detector import OpenPoseDetector
        return OpenPoseDetector(), "OpenPose"
    else:
        print(f"Unknown model '{model_name}'. Defaulting to Openpifpaf.")
        from openpifpaf_pose_detector import OpenpifpafPoseDetector
        return OpenpifpafPoseDetector(), "Openpifpaf"


def extract_high_res_keypoints(pose_detector, original_bgr: np.ndarray, model_type: str):
    """
    Runs detection and maps keypoints accurately back to original high-resolution image coordinates.
    """
    orig_h, orig_w = original_bgr.shape[:2]

    preprocessed = pose_detector.preprocess_image(original_bgr)
    raw_points = pose_detector.get_joint_points()

    if hasattr(pose_detector, "image"):
        det_img = pose_detector.image
        if isinstance(det_img, Image.Image):
            proc_w, proc_h = det_img.size
        elif isinstance(det_img, np.ndarray):
            proc_h, proc_w = det_img.shape[:2]
        else:
            proc_h = 400
            proc_w = int(orig_w * (400.0 / orig_h))
    else:
        proc_h = 400
        proc_w = int(orig_w * (400.0 / orig_h))

    scale_x = orig_w / float(proc_w)
    scale_y = orig_h / float(proc_h)

    high_res_points = []
    for pt in raw_points:
        if pt is not None and len(pt) >= 2 and pt[0] is not None and pt[1] is not None:
            orig_x = int(round(pt[0] * scale_x))
            orig_y = int(round(pt[1] * scale_y))
            orig_x = max(0, min(orig_w - 1, orig_x))
            orig_y = max(0, min(orig_h - 1, orig_y))
            high_res_points.append((orig_x, orig_y))
        else:
            high_res_points.append(None)

    if model_type == "Openpifpaf" or len(high_res_points) == 17:
        names = {
            0: "Nose", 1: "L-Eye", 2: "R-Eye", 3: "L-Ear", 4: "R-Ear",
            5: "L-Shoulder", 6: "R-Shoulder", 7: "L-Elbow", 8: "R-Elbow",
            9: "L-Wrist", 10: "R-Wrist", 11: "L-Hip", 12: "R-Hip",
            13: "L-Knee", 14: "R-Knee", 15: "L-Ankle", 16: "R-Ankle"
        }
    elif model_type == "Mediapipe" or len(high_res_points) == 33:
        names = {
            0: "Nose", 2: "L-Eye", 5: "R-Eye", 7: "L-Ear", 8: "R-Ear",
            11: "L-Shoulder", 12: "R-Shoulder", 13: "L-Elbow", 14: "R-Elbow",
            15: "L-Wrist", 16: "R-Wrist", 23: "L-Hip", 24: "R-Hip",
            25: "L-Knee", 26: "R-Knee", 27: "L-Ankle", 28: "R-Ankle"
        }
    else:
        names = {i: f"Joint-{i}" for i in range(len(high_res_points))}

    return high_res_points, names


def apply_face_blur(original_bgr: np.ndarray, points: list, model_type: str) -> np.ndarray:
    """
    Blurs the face using the project's FaceBlurring class.
    Works on the original high-resolution image with the high-res keypoints, and returns a
    new blurred image (the input image is not modified). Falls back to the unblurred image
    if the face keypoints are missing or blurring fails.
    """
    if model_type == "Openpifpaf" or len(points) == 17:
        nose_idx, l_eye_idx, r_eye_idx, l_ear_idx, r_ear_idx = 0, 1, 2, 3, 4
    elif model_type == "Mediapipe" or len(points) == 33:
        nose_idx, l_eye_idx, r_eye_idx, l_ear_idx, r_ear_idx = 0, 2, 5, 7, 8
    else:
        nose_idx, l_eye_idx, r_eye_idx, l_ear_idx, r_ear_idx = 0, 1, 2, 3, 4

    if len(points) <= max(nose_idx, l_eye_idx, r_eye_idx, l_ear_idx, r_ear_idx):
        print("Warning: not enough keypoints for face blurring, skipping.")
        return original_bgr.copy()

    try:
        blurrer = FaceBlurring()
        blurred = blurrer.blur_face(points, nose_idx, l_eye_idx, r_eye_idx,
                                    l_ear_idx, r_ear_idx, original_bgr)
        return np.ascontiguousarray(blurred, dtype=np.uint8)
    except Exception as e:
        print(f"Warning: face blurring failed ({e}), continuing without blur.")
        return original_bgr.copy()


# ---------------------------------------------------------------------------
# High-Resolution Rendering Utilities
# ---------------------------------------------------------------------------

def get_drawing_scale(image_shape):
    """Calculates scaling multiplier based on image dimensions."""
    h, w = image_shape[:2]
    return max(w, h) / 1000.0


def dim_background(img_bgr: np.ndarray, strength: float = 0.6, mode: str = "light") -> np.ndarray:
    """
    Washes out the person/photo so overlays (joints, vectors, arcs) stand out.
    strength: 0.0 = original image, 1.0 = fully blank.
    mode: "light" blends toward white (person looks pale/ghosted),
          "dark" blends toward black.
    """
    strength = max(0.0, min(1.0, strength))
    if strength == 0.0:
        return img_bgr.copy()
    target_value = 0 if mode == "dark" else 255
    target = np.full_like(img_bgr, target_value)
    return cv2.addWeighted(img_bgr, 1.0 - strength, target, strength, 0)


def draw_styled_joint(img, pt, color=(0, 255, 255), radius=8, outline_color=(15, 15, 15), outline_thick=2):
    """Draws a crisp joint point with outer ring for high visual clarity."""
    if pt is None:
        return
    x, y = int(pt[0]), int(pt[1])
    cv2.circle(img, (x, y), radius + outline_thick, outline_color, -1, cv2.LINE_AA)
    cv2.circle(img, (x, y), radius, color, -1, cv2.LINE_AA)
    cv2.circle(img, (x, y), max(2, radius // 4), (255, 255, 255), -1, cv2.LINE_AA)


def draw_dashed_line(img, pt1, pt2, color=(0, 0, 255), thickness=2, dash_length=12):
    """Draws an anti-aliased dashed reference line between two points in red."""
    p1 = np.array(pt1, dtype=float)
    p2 = np.array(pt2, dtype=float)
    dist = np.linalg.norm(p2 - p1)
    if dist == 0:
        return
    num_dashes = int(dist / (2 * dash_length))
    for i in range(num_dashes + 1):
        start = p1 + (p2 - p1) * (2 * i * dash_length / dist)
        end = p1 + (p2 - p1) * (min(dist, (2 * i + 1) * dash_length) / dist)
        s_pt = (int(round(start[0])), int(round(start[1])))
        e_pt = (int(round(end[0])), int(round(end[1])))
        cv2.line(img, s_pt, e_pt, color, thickness, cv2.LINE_AA)


def draw_styled_vector(img, pt1, pt2, color=(0, 255, 0), thickness=3, show_arrow=False, arrow_size=12):
    """Draws an anti-aliased vector line with dark contrast outline and optional directional arrow."""
    if pt1 is None or pt2 is None:
        return
    p1 = (int(round(pt1[0])), int(round(pt1[1])))
    p2 = (int(round(pt2[0])), int(round(pt2[1])))

    cv2.line(img, p1, p2, (15, 15, 15), thickness + 2, cv2.LINE_AA)
    cv2.line(img, p1, p2, color, thickness, cv2.LINE_AA)

    if show_arrow:
        v = np.array(p2, dtype=float) - np.array(p1, dtype=float)
        v_len = np.linalg.norm(v)
        if v_len > arrow_size * 1.5:
            u = v / v_len
            perp = np.array([-u[1], u[0]])
            tip = np.array(p1, dtype=float) + v * 0.65
            left = tip - u * arrow_size + perp * (arrow_size * 0.5)
            right = tip - u * arrow_size - perp * (arrow_size * 0.5)
            pts = np.array([tip, left, right], dtype=np.int32)
            cv2.fillPoly(img, [pts], color, cv2.LINE_AA)
            cv2.polylines(img, [pts], isClosed=True, color=(15, 15, 15), thickness=1, lineType=cv2.LINE_AA)


def draw_angle_arc(img, vertex, p1, p2, radius=40, color=(0, 255, 255), thickness=2):
    """Draws a smooth circular arc between two vectors originating at vertex."""
    if vertex is None or p1 is None or p2 is None:
        return
    vx, vy = float(vertex[0]), float(vertex[1])

    ang1 = math.degrees(math.atan2(p1[1] - vy, p1[0] - vx)) % 360
    ang2 = math.degrees(math.atan2(p2[1] - vy, p2[0] - vx)) % 360

    diff = (ang2 - ang1) % 360
    if diff > 180:
        start_ang, end_ang = ang2, ang1 + 360
    else:
        start_ang, end_ang = ang1, ang2

    center = (int(round(vx)), int(round(vy)))
    axes = (int(round(radius)), int(round(radius)))

    cv2.ellipse(img, center, axes, 0, start_ang, end_ang, (15, 15, 15), thickness + 2, cv2.LINE_AA)
    cv2.ellipse(img, center, axes, 0, start_ang, end_ang, color, thickness, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Vector and Angle Math Calculations
# ---------------------------------------------------------------------------

def calculate_angle_3pts(p1, p2, p3):
    """Calculates angle in degrees at vertex p2 between rays p2->p1 and p2->p3."""
    if p1 is None or p2 is None or p3 is None:
        return None
    v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]], dtype=float)
    v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]], dtype=float)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return None
    dot = np.dot(v1, v2) / (norm1 * norm2)
    dot = max(-1.0, min(1.0, dot))
    angle = math.degrees(math.acos(dot))
    return round(angle, 1)


def calculate_angle_vector_axis(v, axis_vector=np.array([0, -1])):
    """Calculates angle in degrees between vector v and reference axis (e.g. upward vertical)."""
    if v is None:
        return None
    v_arr = np.array(v, dtype=float)
    norm = np.linalg.norm(v_arr)
    axis_norm = np.linalg.norm(axis_vector)
    if norm == 0 or axis_norm == 0:
        return None
    dot = np.dot(v_arr, axis_vector) / (norm * axis_norm)
    dot = max(-1.0, min(1.0, dot))
    angle = math.degrees(math.acos(dot))
    return round(angle, 1)


# ---------------------------------------------------------------------------
# Image 1: Joint Extracted Image (no text)
# ---------------------------------------------------------------------------

def generate_joint_extracted_image(original_bgr: np.ndarray, points: list) -> np.ndarray:
    """
    Generates Image 1: Detected body joint points overlaid at original high resolution
    with distinct color-coded anatomical markers. No text is drawn.
    """
    img = original_bgr.copy()
    scale = get_drawing_scale(img.shape)
    radius = max(6, int(round(7.5 * scale)))
    outline_thick = max(1, int(round(2 * scale)))

    palette_bgr = {
        "head": (255, 240, 0),      # Cyan
        "l_arm": (0, 165, 255),     # Orange
        "r_arm": (0, 215, 255),     # Gold
        "torso": (50, 235, 50),     # Neon Green
        "l_leg": (220, 50, 220),    # Magenta
        "r_leg": (255, 170, 0),     # Sky Blue
        "default": (0, 255, 255)    # Yellow
    }

    for idx, pt in enumerate(points):
        if pt is None:
            continue
        c_bgr = palette_bgr["default"]
        if idx in [0, 1, 2, 3, 4, 7, 8]:
            c_bgr = palette_bgr["head"]
        elif idx in [5, 7, 9, 11, 13, 15]:
            c_bgr = palette_bgr["l_arm"]
        elif idx in [6, 8, 10, 12, 14, 16]:
            c_bgr = palette_bgr["r_arm"]
        elif idx in [11, 12, 23, 24]:
            c_bgr = palette_bgr["torso"]
        elif idx in [13, 15, 25, 27]:
            c_bgr = palette_bgr["l_leg"]
        elif idx in [14, 16, 26, 28]:
            c_bgr = palette_bgr["r_leg"]
        draw_styled_joint(img, pt, color=c_bgr, radius=radius, outline_thick=outline_thick)

    return img


# ---------------------------------------------------------------------------
# Image 2: Vector Extracted Image (no text)
# ---------------------------------------------------------------------------

def generate_vector_extracted_image(original_bgr: np.ndarray, points: list,
                                    detector_type: str = "Openpifpaf") -> np.ndarray:
    """
    Generates Image 2: Kinematic skeleton vectors connecting anatomical joints
    overlaid at original high resolution with directional arrows. No text is drawn.
    """
    img = original_bgr.copy()
    scale = get_drawing_scale(img.shape)
    thick = max(2, int(round(3.5 * scale)))
    radius = max(5, int(round(6.5 * scale)))
    arrow_size = max(8, int(round(11 * scale)))

    if detector_type == "Openpifpaf" or len(points) == 17:
        segments = [
            # Head & Neck
            {"pair": (0, 1), "color": (255, 220, 0)},
            {"pair": (1, 3), "color": (255, 220, 0)},
            {"pair": (3, 5), "color": (0, 230, 255)},
            # Torso
            {"pair": (5, 6), "color": (50, 220, 50)},
            {"pair": (5, 11), "color": (0, 255, 128)},
            {"pair": (6, 12), "color": (0, 255, 128)},
            {"pair": (11, 12), "color": (50, 220, 50)},
            # Upper Limbs
            {"pair": (5, 7), "color": (0, 165, 255)},
            {"pair": (7, 9), "color": (0, 140, 255)},
            {"pair": (6, 8), "color": (0, 215, 255)},
            {"pair": (8, 10), "color": (0, 195, 255)},
            # Lower Limbs
            {"pair": (11, 13), "color": (220, 60, 220)},
            {"pair": (13, 15), "color": (240, 90, 240)},
            {"pair": (12, 14), "color": (255, 170, 0)},
            {"pair": (14, 16), "color": (255, 140, 0)},
        ]
    elif detector_type == "Mediapipe" or len(points) == 33:
        segments = [
            {"pair": (7, 11), "color": (0, 230, 255)},
            {"pair": (11, 12), "color": (50, 220, 50)},
            {"pair": (11, 23), "color": (0, 255, 128)},
            {"pair": (12, 24), "color": (0, 255, 128)},
            {"pair": (23, 24), "color": (50, 220, 50)},
            {"pair": (11, 13), "color": (0, 165, 255)},
            {"pair": (13, 15), "color": (0, 140, 255)},
            {"pair": (12, 14), "color": (0, 215, 255)},
            {"pair": (14, 16), "color": (0, 195, 255)},
            {"pair": (23, 25), "color": (220, 60, 220)},
            {"pair": (25, 27), "color": (240, 90, 240)},
            {"pair": (24, 26), "color": (255, 170, 0)},
            {"pair": (26, 28), "color": (255, 140, 0)},
        ]
    else:
        segments = [
            {"pair": (5, 6), "color": (50, 220, 50)},
            {"pair": (5, 7), "color": (0, 165, 255)},
            {"pair": (7, 9), "color": (0, 165, 255)},
            {"pair": (11, 13), "color": (220, 60, 220)},
            {"pair": (13, 15), "color": (220, 60, 220)},
        ]

    for seg in segments:
        i, j = seg["pair"]
        if i < len(points) and j < len(points):
            p1, p2 = points[i], points[j]
            if p1 is not None and p2 is not None:
                draw_styled_vector(img, p1, p2, color=seg["color"], thickness=thick,
                                   show_arrow=True, arrow_size=arrow_size)

    for pt in points:
        if pt is not None:
            draw_styled_joint(img, pt, color=(240, 240, 240), radius=radius, outline_thick=max(1, thick // 2))

    return img


# ---------------------------------------------------------------------------
# Image 3: Angle Calculated Image (no text)
# ---------------------------------------------------------------------------

def generate_angle_calculated_image(original_bgr: np.ndarray, points: list, detector_type: str = "Openpifpaf",
                                    light_bg: bool = True) -> np.ndarray:
    """
    Generates Image 3: Postural angles (Neck, Trunk, Knee, Elbow) shown with geometric arcs
    and reference axes only. Arc color encodes status (green = optimal, orange = risk).
    No text is drawn on the image; the numeric values are printed to the console.
    light_bg: True when the background is faded toward white (uses dark skeleton lines for contrast).
    """
    img = original_bgr.copy()
    ref_line_color = (70, 70, 70) if light_bg else (210, 210, 210)
    scale = get_drawing_scale(img.shape)
    thick = max(2, int(round(3 * scale)))
    radius = max(5, int(round(6 * scale)))
    arc_radius = max(26, int(round(40 * scale)))

    if detector_type == "Openpifpaf" or len(points) == 17:
        l_ear, r_ear = points[3], points[4]
        l_sho, r_sho = points[5], points[6]
        l_elb, r_elb = points[7], points[8]
        l_wri, r_wri = points[9], points[10]
        l_hip, r_hip = points[11], points[12]
        l_kne, r_kne = points[13], points[14]
        l_ank, r_ank = points[15], points[16]
    else:
        l_ear, r_ear = points[7], points[8]
        l_sho, r_sho = points[11], points[12]
        l_elb, r_elb = points[13], points[14]
        l_wri, r_wri = points[15], points[16]
        l_hip, r_hip = points[23], points[24]
        l_kne, r_kne = points[25], points[26]
        l_ank, r_ank = points[27], points[28]

    ear = l_ear if l_ear is not None else r_ear
    sho = l_sho if l_sho is not None else r_sho
    elb = l_elb if l_elb is not None else r_elb
    wri = l_wri if l_wri is not None else r_wri
    hip = l_hip if l_hip is not None else r_hip
    kne = l_kne if l_kne is not None else r_kne
    ank = l_ank if l_ank is not None else r_ank

    OK_COLOR = (0, 220, 0)
    RISK_COLOR = (0, 100, 255)
    red_dash = (0, 0, 255)

    # 1. Knee Angle (Hip - Knee - Ankle)
    if hip is not None and kne is not None and ank is not None:
        knee_angle = calculate_angle_3pts(hip, kne, ank)
        if knee_angle is not None:
            is_optimal = (85.0 <= knee_angle <= 105.0)
            color = OK_COLOR if is_optimal else RISK_COLOR
            draw_styled_vector(img, kne, hip, color=ref_line_color, thickness=thick)
            draw_styled_vector(img, kne, ank, color=ref_line_color, thickness=thick)
            draw_angle_arc(img, kne, hip, ank, radius=arc_radius, color=color, thickness=thick)
            print(f"  Knee  angle: {knee_angle:.1f} deg ({'OPTIMAL' if is_optimal else 'RISK'})")

    # 2. Trunk Angle (Shoulder - Hip relative to horizontal)
    if sho is not None and hip is not None:
        v_horiz_ref = (hip[0] - int(130 * scale), hip[1])
        trunk_angle = calculate_angle_3pts(sho, hip, v_horiz_ref)
        if trunk_angle is not None:
            is_optimal = (90.0 <= trunk_angle <= 112.0)
            color = OK_COLOR if is_optimal else RISK_COLOR
            draw_styled_vector(img, hip, sho, color=ref_line_color, thickness=thick)
            draw_dashed_line(img, hip, v_horiz_ref, color=red_dash, thickness=max(1, thick - 1))
            v_vert_ref = (hip[0], hip[1] - int(130 * scale))
            draw_dashed_line(img, hip, v_vert_ref, color=red_dash, thickness=max(1, thick - 1))
            draw_angle_arc(img, hip, sho, v_horiz_ref, radius=arc_radius, color=color, thickness=thick)
            print(f"  Trunk angle: {trunk_angle:.1f} deg ({'OPTIMAL' if is_optimal else 'RISK'})")

    # 3. Neck Angle (Ear - Shoulder vs Upward Vertical)
    if ear is not None and sho is not None:
        v_neck = np.array([ear[0] - sho[0], ear[1] - sho[1]], dtype=float)
        neck_vert_angle = calculate_angle_vector_axis(v_neck, np.array([0, -1]))
        vert_ref_point = (sho[0], sho[1] - int(130 * scale))
        if neck_vert_angle is not None:
            is_optimal = (neck_vert_angle <= 22.0)
            color = OK_COLOR if is_optimal else RISK_COLOR
            draw_styled_vector(img, sho, ear, color=ref_line_color, thickness=thick)
            draw_dashed_line(img, sho, vert_ref_point, color=red_dash, thickness=max(1, thick - 1))
            draw_angle_arc(img, sho, ear, vert_ref_point, radius=int(arc_radius * 0.8), color=color, thickness=thick)
            print(f"  Neck  angle: {neck_vert_angle:.1f} deg ({'OPTIMAL' if is_optimal else 'BENT FORWARD'})")

    # 4. Elbow Angle (Shoulder - Elbow - Wrist)
    if sho is not None and elb is not None and wri is not None:
        elbow_angle = calculate_angle_3pts(sho, elb, wri)
        if elbow_angle is not None:
            is_optimal = (85.0 <= elbow_angle <= 115.0)
            color = OK_COLOR if is_optimal else RISK_COLOR
            draw_styled_vector(img, elb, sho, color=ref_line_color, thickness=thick)
            draw_styled_vector(img, elb, wri, color=ref_line_color, thickness=thick)
            draw_angle_arc(img, elb, sho, wri, radius=int(arc_radius * 0.85), color=color, thickness=thick)
            print(f"  Elbow angle: {elbow_angle:.1f} deg ({'OPTIMAL' if is_optimal else 'RISK'})")

    # Draw joint points on top
    for pt in points:
        if pt is not None:
            draw_styled_joint(img, pt, color=(255, 255, 255), radius=radius, outline_thick=max(1, thick // 2))

    return img


# ---------------------------------------------------------------------------
# Multi-Panel Combined Figure
# ---------------------------------------------------------------------------

def generate_multi_panel_figure(orig_bgr: np.ndarray, img_joints: np.ndarray,
                                img_vectors: np.ndarray, img_angles: np.ndarray) -> np.ndarray:
    """
    Generates a 2x2 composite figure suitable for academic paper publication.
    The first panel is the untouched original photo.
    """
    h, w = orig_bgr.shape[:2]
    target_w = 1200
    scale_factor = target_w / float(w)
    target_h = int(h * scale_factor)

    p1 = cv2.resize(orig_bgr, (target_w, target_h), interpolation=cv2.INTER_AREA)
    p2 = cv2.resize(img_joints, (target_w, target_h), interpolation=cv2.INTER_AREA)
    p3 = cv2.resize(img_vectors, (target_w, target_h), interpolation=cv2.INTER_AREA)
    p4 = cv2.resize(img_angles, (target_w, target_h), interpolation=cv2.INTER_AREA)

    top_row = np.hstack([p1, p2])
    bot_row = np.hstack([p3, p4])
    return np.vstack([top_row, bot_row])


# ---------------------------------------------------------------------------
# Main Pipeline Function
# ---------------------------------------------------------------------------

def process_image(input_image_path: str, output_directory: str, model_name: str = None,
                  save_composite: bool = True, bg_dim: float = 0.6, bg_mode: str = "light",
                  blur_face: bool = True):
    """
    Executes the full paper diagram generation pipeline on a high-resolution image.
    Generates and saves the images separately (no text on any of them).

    bg_dim:  background fade strength, 0.0 (original photo) to 1.0 (blank).
    bg_mode: "light" fades the person toward white (default), "dark" fades toward black.
    """
    if not os.path.exists(input_image_path):
        raise FileNotFoundError(f"Input image not found: {input_image_path}")

    os.makedirs(output_directory, exist_ok=True)

    if model_name is None or model_name.strip() == "":
        model_name = get_default_model_from_main()
    print(f"[*] Selected Pose Detection Model: {model_name}")

    orig_bgr = cv2.imread(input_image_path)
    if orig_bgr is None:
        raise ValueError(f"Failed to read image at: {input_image_path}")
    orig_h, orig_w = orig_bgr.shape[:2]
    print(f"[*] Input image loaded: {input_image_path} (High-Resolution: {orig_w}x{orig_h})")

    # Detection always runs on the untouched original image
    pose_detector, resolved_model = load_pose_detector(model_name)
    high_res_points, keypoint_names = extract_high_res_keypoints(pose_detector, orig_bgr, resolved_model)
    detected_count = sum(1 for p in high_res_points if p is not None)
    print(f"[*] Successfully extracted {detected_count}/{len(high_res_points)} body joints.")

    # Blur the face (detection already ran on the unblurred image, so keypoints stay accurate)
    if blur_face:
        orig_bgr = apply_face_blur(orig_bgr, high_res_points, resolved_model)
        print("[*] Face blurred.")

    # Faded background used only for the annotated figures
    bg_bgr = dim_background(orig_bgr, bg_dim, bg_mode)
    print(f"[*] Background fade: strength={bg_dim}, mode={bg_mode}")

    base_name = os.path.splitext(os.path.basename(input_image_path))[0]

    # 0. Save the original image with the blurred face (no annotations, no fading)
    blurred_out_path = None
    if blur_face:
        blurred_out_path = os.path.join(output_directory, f"{base_name}_0_original_blurred.png")
        cv2.imwrite(blurred_out_path, orig_bgr)
        print(f"  -> Saved: {blurred_out_path}")

    # 1. Joint Extracted Image
    print("[1/3] Generating 1 - Joint Extracted Image...")
    img_joints = generate_joint_extracted_image(bg_bgr, high_res_points)
    joint_out_path = os.path.join(output_directory, f"{base_name}_1_joint_extracted.png")
    cv2.imwrite(joint_out_path, img_joints)
    print(f"  -> Saved: {joint_out_path}")

    # 2. Vector Extracted Image
    print("[2/3] Generating 2 - Vector Extracted Image...")
    img_vectors = generate_vector_extracted_image(bg_bgr, high_res_points, resolved_model)
    vector_out_path = os.path.join(output_directory, f"{base_name}_2_vector_extracted.png")
    cv2.imwrite(vector_out_path, img_vectors)
    print(f"  -> Saved: {vector_out_path}")

    # 3. Angle Calculated Image
    print("[3/3] Generating 3 - Angle Calculated Image...")
    img_angles = generate_angle_calculated_image(bg_bgr, high_res_points, resolved_model,
                                                 light_bg=(bg_mode == "light"))
    angle_out_path = os.path.join(output_directory, f"{base_name}_3_angle_calculated.png")
    cv2.imwrite(angle_out_path, img_angles)
    print(f"  -> Saved: {angle_out_path}")

    # 4. (Optional) Composite 2x2 Multi-Panel Figure
    composite_out_path = None
    if save_composite:
        print("[Bonus] Generating Multi-Panel Composite Paper Diagram...")
        grid = generate_multi_panel_figure(orig_bgr, img_joints, img_vectors, img_angles)
        composite_out_path = os.path.join(output_directory, f"{base_name}_all_steps_composite.png")
        cv2.imwrite(composite_out_path, grid)
        print(f"  -> Saved: {composite_out_path}")

    print("\n=================================================================")
    print("       High-Resolution Paper Diagrams Generated Successfully!    ")
    print("=================================================================")
    if blurred_out_path:
        print(f"  0. Original (Blurred Face) : {blurred_out_path}")
    print(f"  1. Joint Extracted Image   : {joint_out_path}")
    print(f"  2. Vector Extracted Image  : {vector_out_path}")
    print(f"  3. Angle Calculated Image  : {angle_out_path}")
    if composite_out_path:
        print(f"  4. 2x2 Composite Figure    : {composite_out_path}")
    print("=================================================================\n")

    return joint_out_path, vector_out_path, angle_out_path, composite_out_path


def main():
    default_input = os.path.join(current_dir, "../PaperImages", "sample.png")
    default_output = os.path.join(current_dir, "../PaperImages", "output")
    default_model = get_default_model_from_main()

    parser = argparse.ArgumentParser(description="Generate publication-ready ergonomic posture diagrams (text-free).")
    parser.add_argument("--input", default=default_input, help="Path to high-resolution input image")
    parser.add_argument("--output", default=default_output, help="Path to output directory")
    parser.add_argument("--model", default=default_model, help=f"Pose model to use (default from main.py: {default_model})")
    parser.add_argument("--no_composite", action="store_true", help="Skip generating composite 2x2 figure")
    parser.add_argument("--bg_dim", type=float, default=0.6,
                        help="How much the person/photo is faded, 0-1 (0 = original photo, 1 = blank)")
    parser.add_argument("--bg_mode", choices=["light", "dark"], default="light",
                        help="Fade the person toward white (light) or black (dark)")
    parser.add_argument("--no_blur", action="store_true", help="Do not blur the face")

    args = parser.parse_args()
    process_image(args.input, args.output, args.model,
                  save_composite=not args.no_composite,
                  bg_dim=args.bg_dim, bg_mode=args.bg_mode,
                  blur_face=not args.no_blur)


if __name__ == "__main__":
    main()