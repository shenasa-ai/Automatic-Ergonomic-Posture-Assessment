"""
Paper Image Generator for Ergonomic Posture Assessment.

This script processes high-resolution input images and generates publication-ready
diagrams for scientific papers:
  1. Joint Extracted Image: Detected body keypoints with high-res markers and non-overlapping labels.
  2. Vector Extracted Image: Kinematic skeleton vectors connecting anatomical joints.
  3. Angle Calculated Image: Posture angles (Knee, Trunk, Neck, Elbow) with geometric arcs,
     reference axes, numerical degree values, and ROSA ergonomic status badges.
  4. (Optional) Combined Multi-Panel Figure: Composite figure summarizing all steps.

The model used is automatically read from `src/main.py` (deep_model) or can be specified via CLI.
All visual annotations are dynamically scaled to the original high-resolution image.
"""

import os
import sys
import math
import argparse
import re
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pose_detector import PoseDetector
from openpose_detector import OpenPoseDetector
from mediapipe_pose_detector import MediapipePoseDetector
from openpifpaf_pose_detector import OpenpifpafPoseDetector
from rule_provider import RuleProvider

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


# ---------------------------------------------------------------------------
# High-Resolution Rendering & Font Utilities
# ---------------------------------------------------------------------------

def get_drawing_scale(image_shape):
    """Calculates scaling multiplier based on image dimensions."""
    h, w = image_shape[:2]
    return max(w, h) / 1000.0


def get_font(font_size=18, bold=False):
    """Loads a crisp TrueType font for clean paper typography."""
    font_candidates = [
        "arialbd.ttf" if bold else "arial.ttf",
        "calibrib.ttf" if bold else "calibri.ttf",
        "segoeuib.ttf" if bold else "segoeui.ttf",
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    ]
    for candidate in font_candidates:
        try:
            return ImageFont.truetype(candidate, int(font_size))
        except Exception:
            continue
    return ImageFont.load_default()


def draw_pil_badge(draw, text, pos, font, text_color=(255, 255, 255, 255),
                   bg_color=(25, 30, 40, 220), border_color=(200, 200, 200, 255),
                   padding=6, radius=6, img_size=None):
    """Draws a rounded semi-transparent badge with crisp anti-aliased text and boundary clamping."""
    bbox = draw.textbbox((0, 0), text, font=font)
    t_w = bbox[2] - bbox[0]
    t_h = bbox[3] - bbox[1]
    
    x, y = pos
    x1 = x - padding
    y1 = y - padding
    x2 = x + t_w + padding
    y2 = y + t_h + padding
    
    # Boundary clamping
    if img_size is not None or hasattr(draw, "_image"):
        w, h = img_size if img_size else draw._image.size
        margin = 15
        if x2 > w - margin:
            shift = x2 - (w - margin)
            x -= shift
            x1 -= shift
            x2 -= shift
        if x1 < margin:
            shift = margin - x1
            x += shift
            x1 += shift
            x2 += shift
        if y2 > h - margin:
            shift = y2 - (h - margin)
            y -= shift
            y1 -= shift
            y2 -= shift
        if y1 < margin:
            shift = margin - y1
            y += shift
            y1 += shift
            y2 += shift

    # Draw rounded rectangle
    draw.rounded_rectangle([x1, y1, x2, y2], radius=radius, fill=bg_color, outline=border_color, width=1)
    # Draw text
    draw.text((x, y), text, font=font, fill=text_color)
    return x1, y1, x2, y2


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
# Image 1: Joint Extracted Image
# ---------------------------------------------------------------------------

def generate_joint_extracted_image(original_bgr: np.ndarray, points: list, keypoint_names: dict) -> np.ndarray:
    """
    Generates Image 1: Detected body joint points overlaid at original high resolution
    with distinct anatomical markers, outer rings, and non-overlapping label callouts.
    """
    orig_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(orig_rgb).convert("RGBA")
    overlay = Image.new("RGBA", pil_img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    
    scale = get_drawing_scale(original_bgr.shape)
    radius = max(6, int(round(7.5 * scale)))
    outline_thick = max(1, int(round(2 * scale)))
    font = get_font(max(12, int(round(14 * scale))), bold=True)
    title_font = get_font(max(16, int(round(20 * scale))), bold=True)
    badge_pad = max(3, int(round(4.5 * scale)))
    
    palette_bgr = {
        "head": (255, 240, 0),      # Cyan
        "l_arm": (0, 165, 255),     # Orange
        "r_arm": (0, 215, 255),     # Gold
        "torso": (50, 235, 50),     # Neon Green
        "l_leg": (220, 50, 220),    # Magenta
        "r_leg": (255, 170, 0),     # Sky Blue
        "default": (0, 255, 255)    # Yellow
    }
    
    def get_joint_color_rgba(idx):
        bgr = palette_bgr["default"]
        if idx in [0, 1, 2, 3, 4, 7, 8]:
            bgr = palette_bgr["head"]
        elif idx in [5, 7, 9, 11, 13, 15]:
            bgr = palette_bgr["l_arm"]
        elif idx in [6, 8, 10, 12, 14, 16]:
            bgr = palette_bgr["r_arm"]
        elif idx in [11, 12, 23, 24]:
            bgr = palette_bgr["torso"]
        elif idx in [13, 15, 25, 27]:
            bgr = palette_bgr["l_leg"]
        elif idx in [14, 16, 26, 28]:
            bgr = palette_bgr["r_leg"]
        return (bgr[2], bgr[1], bgr[0], 255)

    # First draw styled joint circles on OpenCV layer
    bgr_copy = original_bgr.copy()
    for idx, pt in enumerate(points):
        if pt is not None:
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
            draw_styled_joint(bgr_copy, pt, color=c_bgr, radius=radius, outline_thick=outline_thick)
            
    pil_img = Image.fromarray(cv2.cvtColor(bgr_copy, cv2.COLOR_BGR2RGB)).convert("RGBA")
    draw = ImageDraw.Draw(overlay)
    
    # Specific non-overlapping offsets for joints in side-view to ensure pristine paper readability
    offset_dict = {
        0: (-int(75 * scale), -int(20 * scale)),   # Nose: left
        1: (-int(60 * scale), -int(40 * scale)),   # L-Eye: up-left
        3: (int(15 * scale), -int(30 * scale)),    # L-Ear: up-right
        5: (int(16 * scale), int(14 * scale)),     # L-Shoulder: down-right
        6: (-int(105 * scale), -int(35 * scale)),  # R-Shoulder: up-left
        7: (int(16 * scale), int(6 * scale)),      # L-Elbow: right
        8: (-int(95 * scale), -int(25 * scale)),   # R-Elbow: left
        9: (int(15 * scale), int(10 * scale)),     # L-Wrist: right
        10: (-int(95 * scale), -int(20 * scale)),  # R-Wrist: left
        11: (int(16 * scale), int(10 * scale)),    # L-Hip: right
        12: (-int(75 * scale), -int(25 * scale)),  # R-Hip: left
        13: (int(16 * scale), int(10 * scale)),    # L-Knee: right
        14: (-int(85 * scale), -int(25 * scale)),  # R-Knee: left
        15: (int(16 * scale), int(8 * scale)),     # L-Ankle: right
        16: (-int(85 * scale), -int(25 * scale)),  # R-Ankle: left
    }
    
    for idx, pt in enumerate(points):
        if pt is not None:
            name = keypoint_names.get(idx, f"J{idx}")
            label_text = f"{idx}: {name}"
            dx, dy = offset_dict.get(idx, (int(15 * scale), -int(10 * scale)))
            pos = (pt[0] + dx, pt[1] + dy)
            border_rgba = get_joint_color_rgba(idx)
            draw_pil_badge(draw, label_text, pos, font, text_color=(255, 255, 255, 255),
                           bg_color=(20, 24, 32, 210), border_color=border_rgba,
                           padding=badge_pad, radius=max(3, int(4 * scale)))
                           
    combined = Image.alpha_composite(pil_img, overlay).convert("RGB")
    return cv2.cvtColor(np.array(combined), cv2.COLOR_RGB2BGR)


# ---------------------------------------------------------------------------
# Image 2: Vector Extracted Image
# ---------------------------------------------------------------------------

def generate_vector_extracted_image(original_bgr: np.ndarray, points: list, keypoint_names: dict,
                                    detector_type: str = "Openpifpaf") -> np.ndarray:
    """
    Generates Image 2: Kinematic skeleton vectors connecting anatomical joints
    overlaid at original high resolution with directional arrows and segment grouping.
    """
    img = original_bgr.copy()
    scale = get_drawing_scale(img.shape)
    thick = max(2, int(round(3.5 * scale)))
    radius = max(5, int(round(6.5 * scale)))
    arrow_size = max(8, int(round(11 * scale)))
    
    if detector_type == "Openpifpaf" or len(points) == 17:
        segments = [
            # Head & Neck
            {"pair": (0, 1), "color": (255, 220, 0), "name": "Nose-Eye"},
            {"pair": (1, 3), "color": (255, 220, 0), "name": "Eye-Ear"},
            {"pair": (3, 5), "color": (0, 230, 255), "name": "Ear-Shoulder (Neck)"},
            # Torso
            {"pair": (5, 6), "color": (50, 220, 50), "name": "Bi-Shoulder"},
            {"pair": (5, 11), "color": (0, 255, 128), "name": "L-Trunk"},
            {"pair": (6, 12), "color": (0, 255, 128), "name": "R-Trunk"},
            {"pair": (11, 12), "color": (50, 220, 50), "name": "Bi-Pelvis"},
            # Upper Limbs (Left & Right)
            {"pair": (5, 7), "color": (0, 165, 255), "name": "L-UpperArm"},
            {"pair": (7, 9), "color": (0, 140, 255), "name": "L-ForeArm"},
            {"pair": (6, 8), "color": (0, 215, 255), "name": "R-UpperArm"},
            {"pair": (8, 10), "color": (0, 195, 255), "name": "R-ForeArm"},
            # Lower Limbs (Left & Right)
            {"pair": (11, 13), "color": (220, 60, 220), "name": "L-Thigh"},
            {"pair": (13, 15), "color": (240, 90, 240), "name": "L-Shank"},
            {"pair": (12, 14), "color": (255, 170, 0), "name": "R-Thigh"},
            {"pair": (14, 16), "color": (255, 140, 0), "name": "R-Shank"},
        ]
    elif detector_type == "Mediapipe" or len(points) == 33:
        segments = [
            {"pair": (7, 11), "color": (0, 230, 255), "name": "Neck"},
            {"pair": (11, 12), "color": (50, 220, 50), "name": "Shoulders"},
            {"pair": (11, 23), "color": (0, 255, 128), "name": "L-Trunk"},
            {"pair": (12, 24), "color": (0, 255, 128), "name": "R-Trunk"},
            {"pair": (23, 24), "color": (50, 220, 50), "name": "Pelvis"},
            {"pair": (11, 13), "color": (0, 165, 255), "name": "L-UpperArm"},
            {"pair": (13, 15), "color": (0, 140, 255), "name": "L-ForeArm"},
            {"pair": (12, 14), "color": (0, 215, 255), "name": "R-UpperArm"},
            {"pair": (14, 16), "color": (0, 195, 255), "name": "R-ForeArm"},
            {"pair": (23, 25), "color": (220, 60, 220), "name": "L-Thigh"},
            {"pair": (25, 27), "color": (240, 90, 240), "name": "L-Shank"},
            {"pair": (24, 26), "color": (255, 170, 0), "name": "R-Thigh"},
            {"pair": (26, 28), "color": (255, 140, 0), "name": "R-Shank"},
        ]
    else:
        segments = [
            {"pair": (5, 6), "color": (50, 220, 50), "name": "Torso"},
            {"pair": (5, 7), "color": (0, 165, 255), "name": "Arm"},
            {"pair": (7, 9), "color": (0, 165, 255), "name": "Arm"},
            {"pair": (11, 13), "color": (220, 60, 220), "name": "Leg"},
            {"pair": (13, 15), "color": (220, 60, 220), "name": "Leg"}
        ]
        
    for seg in segments:
        i, j = seg["pair"]
        if i < len(points) and j < len(points):
            p1, p2 = points[i], points[j]
            if p1 is not None and p2 is not None:
                draw_styled_vector(img, p1, p2, color=seg["color"], thickness=thick,
                                   show_arrow=True, arrow_size=arrow_size)
                                   
    for idx, pt in enumerate(points):
        if pt is not None:
            draw_styled_joint(img, pt, color=(240, 240, 240), radius=radius, outline_thick=max(1, thick // 2))
            
    return img


# ---------------------------------------------------------------------------
# Image 3: Angle Calculated Image
# ---------------------------------------------------------------------------

def generate_angle_calculated_image(original_bgr: np.ndarray, points: list, detector_type: str = "Openpifpaf") -> np.ndarray:
    """
    Generates Image 3: Calculated ergonomic postural angles (Neck, Trunk, Knee, Elbow)
    with geometric arcs, reference axes, degree values (°), and ROSA ergonomic status badges.
    """
    img = original_bgr.copy()
    scale = get_drawing_scale(img.shape)
    thick = max(2, int(round(3 * scale)))
    radius = max(5, int(round(6 * scale)))
    arc_radius = max(26, int(round(40 * scale)))
    
    if detector_type == "Openpifpaf" or len(points) == 17:
        nose = points[0]
        l_ear = points[3]
        r_ear = points[4]
        l_sho = points[5]
        r_sho = points[6]
        l_elb = points[7]
        r_elb = points[8]
        l_wri = points[9]
        r_wri = points[10]
        l_hip = points[11]
        r_hip = points[12]
        l_kne = points[13]
        r_kne = points[14]
        l_ank = points[15]
        r_ank = points[16]
    else:
        nose = points[0]
        l_ear = points[7]
        r_ear = points[8]
        l_sho = points[11]
        r_sho = points[12]
        l_elb = points[13]
        r_elb = points[14]
        l_wri = points[15]
        r_wri = points[16]
        l_hip = points[23]
        r_hip = points[24]
        l_kne = points[25]
        r_kne = points[26]
        l_ank = points[27]
        r_ank = points[28]
        
    ear = l_ear if l_ear is not None else r_ear
    sho = l_sho if l_sho is not None else r_sho
    elb = l_elb if l_elb is not None else r_elb
    wri = l_wri if l_wri is not None else r_wri
    hip = l_hip if l_hip is not None else r_hip
    kne = l_kne if l_kne is not None else r_kne
    ank = l_ank if l_ank is not None else r_ank
    
    badges_to_render = []
    summary_items = []
    
    # 1. Knee Angle (Hip - Knee - Ankle)
    if hip is not None and kne is not None and ank is not None:
        knee_angle = calculate_angle_3pts(hip, kne, ank)
        if knee_angle is not None:
            is_optimal = (85.0 <= knee_angle <= 105.0)
            status_color_bgr = (0, 220, 0) if is_optimal else (0, 100, 255)
            status_color_rgba = (0, 220, 0, 255) if is_optimal else (255, 100, 0, 255)
            status_text = "OPTIMAL" if is_optimal else "RISK"
            
            draw_styled_vector(img, kne, hip, color=(210, 210, 210), thickness=thick)
            draw_styled_vector(img, kne, ank, color=(210, 210, 210), thickness=thick)
            draw_angle_arc(img, kne, hip, ank, radius=arc_radius, color=status_color_bgr, thickness=thick)
            
            badge_pos = (kne[0] + int(20 * scale), kne[1] + int(15 * scale))
            badge_text = f"Knee Angle: {knee_angle:.1f}° ({status_text})"
            badges_to_render.append((badge_text, badge_pos, status_color_rgba))
            summary_items.append(("Knee Posture (Hip-Knee-Ankle)", f"{knee_angle:.1f}°", status_text, is_optimal))
            
    # 2. Trunk Angle (Shoulder - Hip relative to horizontal & vertical)
    if sho is not None and hip is not None:
        v_horiz_ref = (hip[0] - int(130 * scale), hip[1])
        trunk_angle = calculate_angle_3pts(sho, hip, v_horiz_ref)
        
        if trunk_angle is not None:
            is_optimal = (90.0 <= trunk_angle <= 112.0)
            status_color_bgr = (0, 220, 0) if is_optimal else (0, 100, 255)
            status_color_rgba = (0, 220, 0, 255) if is_optimal else (255, 100, 0, 255)
            status_text = "OPTIMAL" if is_optimal else "RISK"
            
            draw_styled_vector(img, hip, sho, color=(210, 210, 210), thickness=thick)
            draw_dashed_line(img, hip, v_horiz_ref, color=(0, 0, 255), thickness=max(1, thick - 1))
            v_vert_ref = (hip[0], hip[1] - int(130 * scale))
            draw_dashed_line(img, hip, v_vert_ref, color=(0, 0, 255), thickness=max(1, thick - 1))
            draw_angle_arc(img, hip, sho, v_horiz_ref, radius=arc_radius, color=status_color_bgr, thickness=thick)
            
            badge_pos = (hip[0] - int(250 * scale), hip[1] - int(30 * scale))
            badge_text = f"Trunk Angle: {trunk_angle:.1f}° ({status_text})"
            badges_to_render.append((badge_text, badge_pos, status_color_rgba))
            summary_items.append(("Trunk Posture (Backrest Angle)", f"{trunk_angle:.1f}°", status_text, is_optimal))
            
    # 3. Neck Angle (Ear - Shoulder vs Upward Vertical)
    if ear is not None and sho is not None:
        v_neck = np.array([ear[0] - sho[0], ear[1] - sho[1]], dtype=float)
        neck_vert_angle = calculate_angle_vector_axis(v_neck, np.array([0, -1]))
        vert_ref_point = (sho[0], sho[1] - int(130 * scale))
        
        if neck_vert_angle is not None:
            is_optimal = (neck_vert_angle <= 22.0)
            status_color_bgr = (0, 220, 0) if is_optimal else (0, 100, 255)
            status_color_rgba = (0, 220, 0, 255) if is_optimal else (255, 100, 0, 255)
            status_text = "OPTIMAL" if is_optimal else "BENT FORWARD"
            
            draw_styled_vector(img, sho, ear, color=(210, 210, 210), thickness=thick)
            draw_dashed_line(img, sho, vert_ref_point, color=(0, 0, 255), thickness=max(1, thick - 1))
            draw_angle_arc(img, sho, ear, vert_ref_point, radius=int(arc_radius * 0.8), color=status_color_bgr, thickness=thick)
            
            badge_pos = (sho[0] + int(20 * scale), sho[1] - int(75 * scale))
            badge_text = f"Neck Flexion: {neck_vert_angle:.1f}° ({status_text})"
            badges_to_render.append((badge_text, badge_pos, status_color_rgba))
            summary_items.append(("Neck Flexion (Screen/Monitor)", f"{neck_vert_angle:.1f}°", status_text, is_optimal))
            
    # 4. Elbow / Arm Angle (Shoulder - Elbow - Wrist)
    if sho is not None and elb is not None and wri is not None:
        elbow_angle = calculate_angle_3pts(sho, elb, wri)
        if elbow_angle is not None:
            is_optimal = (85.0 <= elbow_angle <= 115.0)
            status_color_bgr = (0, 220, 0) if is_optimal else (0, 100, 255)
            status_color_rgba = (0, 220, 0, 255) if is_optimal else (255, 100, 0, 255)
            status_text = "OPTIMAL" if is_optimal else "RISK"
            
            draw_styled_vector(img, elb, sho, color=(210, 210, 210), thickness=thick)
            draw_styled_vector(img, elb, wri, color=(210, 210, 210), thickness=thick)
            draw_angle_arc(img, elb, sho, wri, radius=int(arc_radius * 0.85), color=status_color_bgr, thickness=thick)
            
            badge_pos = (elb[0] + int(20 * scale), elb[1] - int(10 * scale))
            badge_text = f"Elbow Angle: {elbow_angle:.1f}° ({status_text})"
            badges_to_render.append((badge_text, badge_pos, status_color_rgba))
            summary_items.append(("Armrest / Elbow Angle", f"{elbow_angle:.1f}°", status_text, is_optimal))
            
    # Draw joint points on top
    for pt in points:
        if pt is not None:
            draw_styled_joint(img, pt, color=(255, 255, 255), radius=radius, outline_thick=max(1, thick // 2))
            
    # Render typography and badges using PIL for perfect degree symbol and anti-aliased text
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).convert("RGBA")
    overlay = Image.new("RGBA", pil_img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    
    badge_font = get_font(max(13, int(round(15 * scale))), bold=True)
    
    # Draw Angle Badges
    for b_text, b_pos, b_color in badges_to_render:
        draw_pil_badge(draw, b_text, b_pos, badge_font, text_color=(255, 255, 255, 255),
                       bg_color=(20, 24, 32, 215), border_color=b_color,
                       padding=max(4, int(5 * scale)), radius=max(3, int(4 * scale)))
                       
    combined = Image.alpha_composite(pil_img, overlay).convert("RGB")
    return cv2.cvtColor(np.array(combined), cv2.COLOR_RGB2BGR)


# ---------------------------------------------------------------------------
# Multi-Panel Combined Figure
# ---------------------------------------------------------------------------

def generate_multi_panel_figure(orig_bgr: np.ndarray, img_joints: np.ndarray,
                                img_vectors: np.ndarray, img_angles: np.ndarray) -> np.ndarray:
    """
    Generates a 2x2 composite figure suitable for academic paper publication.
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
    grid = np.vstack([top_row, bot_row])
    return grid


# ---------------------------------------------------------------------------
# Main Pipeline Function
# ---------------------------------------------------------------------------

def process_image(input_image_path: str, output_directory: str, model_name: str = None, save_composite: bool = True):
    """
    Executes the full paper diagram generation pipeline on a high-resolution image.
    Generates and saves the 3 images separately.
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
    
    pose_detector, resolved_model = load_pose_detector(model_name)
    high_res_points, keypoint_names = extract_high_res_keypoints(pose_detector, orig_bgr, resolved_model)
    detected_count = sum(1 for p in high_res_points if p is not None)
    print(f"[*] Successfully extracted {detected_count}/{len(high_res_points)} body joints.")
    
    base_name = os.path.splitext(os.path.basename(input_image_path))[0]
    
    # 1. Generate Joint Extracted Image
    print("[1/3] Generating 1 - Joint Extracted Image...")
    img_joints = generate_joint_extracted_image(orig_bgr, high_res_points, keypoint_names)
    joint_out_path = os.path.join(output_directory, f"{base_name}_1_joint_extracted.png")
    cv2.imwrite(joint_out_path, img_joints)
    print(f"  -> Saved: {joint_out_path}")
    
    # 2. Generate Vector Extracted Image
    print("[2/3] Generating 2 - Vector Extracted Image...")
    img_vectors = generate_vector_extracted_image(orig_bgr, high_res_points, keypoint_names, resolved_model)
    vector_out_path = os.path.join(output_directory, f"{base_name}_2_vector_extracted.png")
    cv2.imwrite(vector_out_path, img_vectors)
    print(f"  -> Saved: {vector_out_path}")
    
    # 3. Generate Angle Calculated Image
    print("[3/3] Generating 3 - Angle Calculated Image...")
    img_angles = generate_angle_calculated_image(orig_bgr, high_res_points, resolved_model)
    angle_out_path = os.path.join(output_directory, f"{base_name}_3_angle_calculated.png")
    cv2.imwrite(angle_out_path, img_angles)
    print(f"  -> Saved: {angle_out_path}")
    
    # 4. (Optional) Generate Composite 2x2 Multi-Panel Figure
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
    print(f"  1. Joint Extracted Image   : {joint_out_path}")
    print(f"  2. Vector Extracted Image  : {vector_out_path}")
    print(f"  3. Angle Calculated Image  : {angle_out_path}")
    if composite_out_path:
        print(f"  4. 2x2 Composite Figure    : {composite_out_path}")
    print("=================================================================\n")
    
    return joint_out_path, vector_out_path, angle_out_path, composite_out_path


def main():
    default_input = os.path.join(current_dir, "../paper_image", "input", "sample.png")
    default_output = os.path.join(current_dir, "../paper_image", "output")
    default_model = get_default_model_from_main()
    
    parser = argparse.ArgumentParser(description="Generate publication-ready ergonomic posture diagrams.")
    parser.add_argument("--input", default=default_input, help="Path to high-resolution input image")
    parser.add_argument("--output", default=default_output, help="Path to output directory")
    parser.add_argument("--model", default=default_model, help=f"Pose model to use (default from main.py: {default_model})")
    parser.add_argument("--no_composite", action="store_true", help="Skip generating composite 2x2 figure")
    
    args = parser.parse_args()
    process_image(args.input, args.output, args.model, save_composite=not args.no_composite)


if __name__ == "__main__":
    main()
