import os
import glob
import shutil

path = '../input'
output_path = '../renamed_input'
folders = sorted(os.listdir(path))

for folder in folders:
    os.makedirs(os.path.join(output_path, folder), exist_ok=True)
    images = sorted(glob.glob(os.path.join(path, folder, '*')))
    for i, src in enumerate(images):
        dest = f'{output_path}/{folder}/{folder}_{i}.jpg'
        shutil.copy(src, dest)
