from statsmodels.stats.inter_rater import fleiss_kappa, aggregate_raters
from collections import Counter
import pandas as pd
import numpy as np
import os
from typing import List, Dict


def finalize_image_labels(
        img_path: str,
        lbl_paths: List[str],
        save_path: str = '.',

        rem_img=False,
        columns=None
) -> None:
    """
    Processes image labels from multiple CSV files, calculates Fleiss' Kappa, and saves the final labels.
    Removes images that do not have consistent labeling across all files.

    Parameters:
    - img_path: Path to the directory containing images.
    - lbl_paths: List of paths to CSV files containing labels.
    - save_path: Directory to save the final labels CSV file. Defaults to the current directory.
    - columns: List of column names to process. Defaults to ['file name', 'chair 1-2', 'back support', 'monitor'].
    """
    # Read and process label files
    if columns is None:
        columns = ['file name', 'chair 1-2', 'back support', 'monitor']
    labels = [pd.read_csv(file, usecols=columns) for file in lbl_paths]

    # Identify rows to drop (where any column has a value of 0)

    # Define the conditions to check
    conditions = ['wrong', 'front', 0]
    rows_to_drop = set()

    for df in labels:
        # Find rows that match any of the conditions
        rows_to_drop.update(df[df.map(lambda x: x in conditions).any(axis=1)].index)

    for i, df in enumerate(labels):
        labels[i] = df.drop(list(rows_to_drop))

    # Extract image numbers from file names
    image_numbers = []
    for df in labels:
        file_names = df['file name']
        df.drop('file name', axis=1, inplace=True)
        image_numbers.extend(int(name.split('_')[1]) for name in file_names)

    # Check for consistency in image numbers
    image_counter = Counter(image_numbers)
    if any(count != 3 for count in image_counter.values()):
        print('A difference has been found in image labeling.')
    else:
        print('All pictures are consistently labeled.')

    # Combine ratings for each category
    categories = ['chair 1-2', 'back support', 'monitor']
    combined_data = {cat: pd.concat([df[cat] for df in labels], axis=1).to_numpy() for cat in categories}

    # Prepare results dictionary
    results = {
        'image_number': list(set(image_numbers)),
        'chair': [], 'chair_rates': [],
        'back': [], 'back_rates': [],
        'monitor': [], 'monitor_rates': []
    }

    # Calculate Fleiss' Kappa and populate results
    for i, (cat, data) in enumerate(combined_data.items()):
        data = [[int(lbl) for lbl in x] for x in data]
        aggregated = aggregate_raters(data)[0]
        for j, row in enumerate(aggregated):
            results[list(results.keys())[2 * i + 1]].append(np.argmax(row) + 1)
            results[list(results.keys())[2 * i + 2]].append(data[j])

    # Save final labels to CSV
    sorted_data = pd.DataFrame(results)
    sorted_data.to_csv(save_path, index=False)

    # Remove images without consistent labeling
    if rem_img:
        image_files = sorted(int(file.split('_')[1].split('.')[0]) for file in os.listdir(img_path))
        images_to_remove = set(image_files) - set(sorted_data['image_number'])
        for img in images_to_remove:
            os.remove(f'{img_path}/side_{img}.jpg')
        print(f"Removed {len(images_to_remove)} inconsistent images.")
