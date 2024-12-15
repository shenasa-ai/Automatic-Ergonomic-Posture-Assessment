from statsmodels.stats.inter_rater import fleiss_kappa, aggregate_raters
import pandas as pd
import numpy as np
import os


def FleissKappa(path: str):
    lbl, columns = [], ['chair 1-2', 'back support', 'monitor']
    for file in path:
        lbl.append(pd.read_csv(file, usecols=columns))
    must_drop = lbl[1][(lbl[1] == 0).any(axis=1)].index.values.tolist()
    for ind, df in enumerate(lbl):
        lbl[ind] = df.drop(index=must_drop)
    chair = pd.concat([df['chair 1-2'] for df in lbl], axis=1).to_numpy()
    back = pd.concat([df['back support'] for df in lbl], axis=1).to_numpy()
    monitor = pd.concat([df['monitor'] for df in lbl], axis=1).to_numpy()
    res = []
    for cat in [chair, back, monitor]:
        res.append(fleiss_kappa(aggregate_raters(cat)[0], method='fleiss'))
    return res