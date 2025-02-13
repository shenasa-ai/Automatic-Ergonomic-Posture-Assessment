from statsmodels.stats.inter_rater import fleiss_kappa

from fleiss_kapa import FleissKappa
from final_label_images import finalize_image_labels
from clear_labels import clean_labels
from Evaluation import calculate_accuracy
from Plot import plot_data_distribution, make_ready4plot
import pandas as pd

'''
1 - run finalize_image_labels
2 - run clean_labels
'''

pred_path = '/home/ali/Desktop/Automatic-Ergonomic-Posture-Assessment/output/pred_Openpifpaf_front.csv'
actual_path = '/home/ali/Desktop/Automatic-Ergonomic-Posture-Assessment/labels/final_labels_front_cleared.csv'

path = ['./../labels/front_aa.csv', './../labels/front_fj.csv', './../labels/ha_checked_front.csv']

# print(FleissKappa(file_paths=path, columns=['arm angle', 'shrogged', 'neck', 'trunck']))

# image2remove = [123, 89, 96, 158, 159, 171, 172, 213, 232, 244, 245, 247, 248, 265, 284]
# exclude_imgs = [23, 24, 25, 26, 53, 64, 65, 66, 67, 68, 176, 177, 178, 179, 181, 182, 183, 184, 185, 186, 187, 253, 254,
#                 306, 307, 308, 309, 310, 311, 312]
# finalize_image_labels(img_path='/home/ali/Desktop/Automatic-Ergonomic-Posture-Assessment/input/main_input/front', lbl_paths=path, save_path='./../labels/final_labels_front.csv', columns = ['file name', 'arm angle', 'shrogged', 'neck', 'trunck'], rem_img=True)
# clean_labels(lbl_paths='./../labels/final_labels_front.csv', img_path='/home/ali/Desktop/Automatic-Ergonomic-Posture-Assessment/input/main_input/front', save_path='./../labels/final_labels_front_cleared.csv')


# Evaluate
print(calculate_accuracy(pred_part='trunck', actual_part='trunck',
                         pred_path=pred_path,
                         actual_path=actual_path))

df = make_ready4plot(pred_path=pred_path, act_path=actual_path, part='trunck')
#
plot_data_distribution(df=df, data_col='trunck_angle_l', value_col='trunck_angle_r', label_col='trunck')
