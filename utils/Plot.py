import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def make_ready4plot(pred_path, act_path, part):
    # Plot dist of data
    pre = pd.read_csv(pred_path,
                      usecols=['image_number', f'{part}_angle']).sort_values('image_number').drop(['image_number'],
                                                                                                  axis=1).reset_index(
        drop=True)
    act = pd.read_csv(act_path,
                      usecols=['image_number', part]).sort_values('image_number').reset_index(drop=True)
    df = pd.concat([act, pre], axis=1)

    return df


def plot_data_distribution(df: pd.DataFrame, data_col: str, value_col: str, label_col: str):
    """
    Plots the distribution of data points with class labels as colors.

    Args:
    df (pd.DataFrame): DataFrame containing the data.
    data_col (str): Column name for the x-axis (data names).
    value_col (str): Column name for the y-axis (data values).
    label_col (str): Column name for the data labels (used for color coding).
    """

    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x=data_col, y=value_col, hue=label_col, palette='viridis', s=100)

    plt.title('Data Distribution by Class')
    plt.xticks(rotation=45)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()
