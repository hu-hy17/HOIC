import cv2
import glob
import os
import json
import numpy as np

root_path = 'J:/code/HandObjInteraction/UIC/results/quanti/Label/Mask/Bottle8/front/Masks/'
label_file_list = glob.glob(root_path + '*.png')
for fn in label_file_list:
    img = cv2.imread(fn)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    result = cv2.bitwise_not(mask)

    img_n = os.path.basename(fn)
    img_n = img_n.replace('_mask_0', '')
    cv2.imwrite(os.path.join(root_path, img_n), result)

