import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List

import cv2
import numpy as np


def draw_skeleton(points, model="mediapipe", image_size=(480, 640), scale=1.0):
    """
    Draw joints and connections on a white background.

    Args:
        points (list/array): list of (x, y) tuples of detected joints
        model (str): model type ("mediapipe", "openpifpaf17", "openpifpaf127", "openpose18", "openpose25")
        image_size (tuple): (height, width) of output image
        scale (float): >1 makes skeleton larger (joints farther apart),
                       <1 makes it smaller (joints closer together)

    Returns:
        white_bg (numpy array): image with skeleton drawn
    """

    edges_dict = {
        "mediapipe": [
            (11, 12), (11, 23), (12, 24), (23, 24),
            (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
            (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),
            (23, 25), (25, 27), (27, 29), (27, 31),
            (24, 26), (26, 28), (28, 30), (28, 32),
        ],
        "openpifpaf17": [
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
            (11, 12), (5, 11), (6, 12),
            (11, 13), (13, 15), (12, 14), (14, 16),
            # (0, 1), (0, 2), (1, 3), (2, 4),
        ],
        "openpifpaf127": [
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
            (11, 12), (5, 11), (6, 12),
            (11, 13), (13, 15), (12, 14), (14, 16),
        ],
        "openpose15": [  # OpenPose MPI model
            (0, 1), (1, 2), (2, 3), (3, 4),  # Right Arm
            (1, 5), (5, 6), (6, 7),  # Left Arm
            (1, 14), (14, 8), (8, 9), (9, 10),  # Right Leg
            (14, 11), (11, 12), (12, 13)  # Left Leg
        ],
        "openpose18": [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (1, 5), (5, 6), (6, 7),
            (1, 8), (1, 11), (8, 9), (9, 10), (11, 12), (12, 13),
            # (0, 14), (0, 15), (14, 16), (15, 17),
        ],
        "openpose25": [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (1, 5), (5, 6), (6, 7),
            (1, 8), (8, 9), (9, 10), (10, 11),
            (8, 12), (12, 13), (13, 14),
            # (0, 15), (0, 16), (15, 17), (16, 18),
            (14, 21), (14, 19), (14, 20),
            (11, 24), (11, 22), (11, 23),
        ]
    }

    h, w = image_size
    white_bg = np.ones((h, w, 3), dtype=np.uint8) * 255

    # Compute center for scaling
    cx, cy = w // 2, h // 2

    # Apply scaling to points
    scaled_points = []
    for (x, y) in points:
        if x > 0 and y > 0:
            x_new = int(cx + (x - cx) * scale)
            y_new = int(cy + (y - cy) * scale)
            scaled_points.append((x_new, y_new))
        else:
            scaled_points.append((x, y))

    # Draw joints + numbers
    for idx, (x, y) in enumerate(scaled_points):
        if x > 0 and y > 0:
            cv2.circle(white_bg, (x, y), 3, (0, 0, 255), -1)
            # cv2.putText(white_bg, str(idx), (x + 6, y - 6),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.2, (255, 0, 0), 1, cv2.LINE_AA)

    # Draw edges
    for (i, j) in edges_dict.get(model, []):
        if i < len(scaled_points) and j < len(scaled_points):
            x1, y1 = scaled_points[i]
            x2, y2 = scaled_points[j]
            if x1 > 0 and y1 > 0 and x2 > 0 and y2 > 0:
                cv2.line(white_bg, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return white_bg


def make_ready4plot(pred_path, act_path, part, usecols):
    # Plot dist of data
    pre = pd.read_csv(pred_path,
                      usecols=usecols).sort_values('image_number').drop(
        ['image_number'],
        axis=1).reset_index(
        drop=True)
    act = pd.read_csv(act_path,
                      usecols=['image_number', part]).sort_values('image_number').reset_index(drop=True)
    df = pd.concat([act, pre], axis=1)

    return df


def plot_data_distribution(df: pd.DataFrame, data_col: str, value_col: str, label_col: str, z_col: str = None):
    """
    Plots the distribution of data points with class labels as colors.
    Supports 2D and 3D plotting.

    Args:
    df (pd.DataFrame): DataFrame containing the data.
    data_col (str): Column name for the x-axis (data names).
    value_col (str): Column name for the y-axis (data values).
    label_col (str): Column name for the data labels (used for color coding).
    z_col (str, optional): Column name for the z-axis (for 3D plotting). If None, a 2D plot is created.
    """
    if z_col is None:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df, x=data_col, y=value_col, hue=label_col, palette='viridis', s=100)
        plt.title('Data Distribution')
        plt.xticks(rotation=45)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.show()
    else:
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection='3d')

        scatter = ax.scatter(df[data_col], df[value_col], df[z_col], c=pd.factorize(df[label_col])[0], cmap='viridis',
                             s=100)
        ax.set_xlabel(data_col)
        ax.set_ylabel(value_col)
        ax.set_zlabel(z_col)
        ax.set_title('Data Distribution')

        legend1 = ax.legend(*scatter.legend_elements(), title=label_col)
        ax.add_artist(legend1)

        plt.show()
