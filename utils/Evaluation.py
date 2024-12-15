from sklearn.metrics import accuracy_score
import pandas as pd


def accuracy(pred, actual, pred_part: str = None, actual_part: str = None):
    pred = pd.read_csv(pred).sort_values('image_number')
    actual = pd.read_csv(actual).sort_values('image_number')
    if (not pred_part) and (not actual_part):
        return accuracy_score(actual, pred)
    else:
        return accuracy_score(actual[actual_part], pred[pred_part])
