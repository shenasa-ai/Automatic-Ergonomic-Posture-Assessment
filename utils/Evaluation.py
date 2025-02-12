from sklearn.metrics import accuracy_score
import pandas as pd
from typing import Optional, List


def calculate_accuracy(
    pred_path: str,
    actual_path: str,
    pred_part: Optional[str] = None,
    actual_part: Optional[str] = None,
    exclude_images=None,
    error_analys=True
) -> float:
    """
    Calculate the accuracy between predicted and actual labels.

    Parameters:
    - pred_path: Path to the CSV file containing predicted labels.
    - actual_path: Path to the CSV file containing actual labels.
    - pred_part: Column name in the predicted CSV to use for accuracy calculation. If None, uses all columns.
    - actual_part: Column name in the actual CSV to use for accuracy calculation. If None, uses all columns.

    Returns:
    - Accuracy score as a float.
    """
    # Load and sort data
    if exclude_images is None:
        exclude_images = []

    pred_df = pd.read_csv(pred_path).sort_values('image_number')
    actual_df = pd.read_csv(actual_path).sort_values('image_number')

    # Ensure the dataframes are aligned
    if pred_df.shape[0] != actual_df.shape[0]:
        raise ValueError("The 'image_number' columns in predicted and actual data do not match.")

    # find Null indices if exist
    null_indices = pred_df[pred_df[actual_part].isnull().values.tolist()]['image_number'].values

    if null_indices is not None:
        exclude_images.extend(null_indices)

    # Exclude specific image numbers if provided
    if exclude_images is not None:
        pred_df = pred_df[~pred_df['image_number'].isin(exclude_images)]
        actual_df = actual_df[~actual_df['image_number'].isin(exclude_images)]
        if exclude_images:
            print(f'{exclude_images} Did not consider in result calculations')

    if error_analys:
        pred_series = pred_df[pred_part].reset_index(drop=True)
        actual_series = actual_df[actual_part].reset_index(drop=True)
        mismatches = actual_series[actual_series != pred_series].index.tolist()
        for img in mismatches:
            img_nmbr = actual_df['image_number'].iloc[img]  # Ensure we use iloc to access by position
            print(img_nmbr)
        pass


    # Calculate accuracy
    if pred_part is None and actual_part is None:
        return accuracy_score(actual_df, pred_df)
    else:
        if pred_part is None or actual_part is None:
            raise ValueError("Both 'pred_part' and 'actual_part' must be provided if one is specified.")
        return accuracy_score(actual_df[actual_part], pred_df[pred_part])