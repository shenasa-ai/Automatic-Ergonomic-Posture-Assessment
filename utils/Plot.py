import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List


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
