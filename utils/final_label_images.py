from statsmodels.stats.inter_rater import fleiss_kappa, aggregate_raters
from collections import Counter
import pandas as pd
import numpy as np
import os


def FinalizeImgLbl(img_path, lbl_path, save_path: str='.'):
    lbl, img_nmbr = [], []
    columns = ['file name', 'chair 1-2', 'back support', 'monitor']
    for file in lbl_path:
        lbl.append(pd.read_csv(file, usecols=columns))
    must_drop = lbl[1][(lbl[1] == 0).any(axis=1)].index.values.tolist()
    for ind, df in enumerate(lbl):
        lbl[ind] = df.drop(index=must_drop)
        temp = lbl[ind]['file name']
        lbl[ind].drop('file name', axis=1, inplace=True)
        img_nmbr.extend(map(lambda x: int(x.split(sep='_')[1]), temp))
    temp = Counter(img_nmbr)
    for item in temp.keys():
        if temp[item] != 3:
            print('a difference has been found')
    print('pictures are same')
    chair = pd.concat([df['chair 1-2'] for df in lbl], axis=1).to_numpy()
    back = pd.concat([df['back support'] for df in lbl], axis=1).to_numpy()
    monitor = pd.concat([df['monitor'] for df in lbl], axis=1).to_numpy()
    res = {'image_number': list(set(img_nmbr)), 'chair': [], 'chair_rates': [], 'back': [], 'back_rates': [],
           'monitor': [], 'monitor_rates': []}
    inx = 1
    for cat in [chair, back, monitor]:
        agg_tmp = aggregate_raters(cat)[0]
        for i, sub in enumerate(agg_tmp):
            res[list(res.keys())[inx]].append(np.argmax(sub) + 1)
            res[list(res.keys())[inx + 1]].append(cat[i])
        inx += 2
    sorted_data = pd.DataFrame(res)
    sorted_data.to_csv(f'{save_path}/final_labels.csv', index=False)
    images = sorted([int(i.split(sep='_')[1].split(sep='.')[0]) for i in os.listdir(img_path)])
    for img in set(images) - set(sorted_data['image_number']):
        os.remove(f'{img_path}/side_{img}.jpg')
