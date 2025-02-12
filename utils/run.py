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

pred_path = '/home/ali/Desktop/Automatic-Ergonomic-Posture-Assessment/output/pred_Openpifpaf.csv'
actual_path = '/home/ali/Desktop/Automatic-Ergonomic-Posture-Assessment/labels/final_labels_cleared.csv'

# path = ['./../labels/aa.csv', './../labels/fj.csv', './../labels/ha.csv']
# image2remove = [123, 89, 96, 158, 159, 171, 172, 213, 232, 244, 245, 247, 248, 265, 284]
exclude_imgs = [23, 24, 25, 26, 53, 64, 65, 66, 67, 68, 176, 177, 178, 179, 181, 182, 183, 184, 185, 186, 187, 253, 254,
                306, 307, 308, 309, 310, 311, 312]
# finalize_image_labels(img_path='/home/ali/Desktop/Automatic-Ergonomic-Posture-Assessment/input/main_input/side', lbl_paths=['/home/ali/Desktop/Automatic-Ergonomic-Posture-Assessment/labels/ha.csv'], save_path='./../labels/ha_acc.csv', rem_img=True)
# clean_labels(lbl_paths='./../labels/ha_acc.csv', img_path='/home/ali/Desktop/Automatic-Ergonomic-Posture-Assessment/input/main_input/side', save_path='./../labels/ha_acc_cleared.csv')


# Evaluate
print(calculate_accuracy(pred_part='chair', actual_part='chair', exclude_images=exclude_imgs,
                         pred_path=pred_path,
                         actual_path=actual_path))

df = make_ready4plot(pred_path=pred_path, act_path=actual_path, part='chair')

plot_data_distribution(df=df, data_col='image_number', value_col='chair_angle', label_col='chair')
