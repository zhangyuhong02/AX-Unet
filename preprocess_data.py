# -*- coding: utf-8 -*-
"""
Created on Sun Jan 23 09:56:22 2022

@author: Yuhong
"""

import sys 
sys.path.append('/home/aistudio/external-libraries')
import numpy as np
import nibabel as nib
from util import * 
from tqdm import tqdm

from util import *
volumes_path = './Pancreas-CT/data'
labels_path = './Pancreas-CT/TCIA_pancreas_labels-02-05-2017/'
preprocess_path = './work/data/preprocess/'

volumes = listdir(volumes_path)
labels = listdir(labels_path)

MIN_BOUND = -1024.
MAX_BOUND = 1024.
def norm_img(image):
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image > 1] = 1.
    image[image < 0] = 0.
    return image



import sys
if './external-libraries' not in sys.path:
    sys.path.append('./external-libraries')
import numpy as np
import nibabel as nib
from util import * 
from tqdm import tqdm
from PIL import Image
from paddle.vision.transforms import functional as F
volumes_path = './Pancreas-CT/data'
labels_path = './Pancreas-CT/TCIA_pancreas_labels-02-05-2017/'
data_preprocess_path = './data/preprocess/data/'
label_preprocess_path = './data/preprocess/label/'
volumes = listdir(volumes_path)
labels = listdir(labels_path)
assert len(volumes) == len(labels)  
count = 0
for i in tqdm(range(len(volumes))):
    count += 1
    if count != 75:continue

    print(os.path.join(volumes_path, volumes[i]))
    
    vol_file = nib.load(os.path.join(volumes_path, volumes[i]))
    lab_file = nib.load(os.path.join(labels_path, labels[i]))

    #读取CT数据
    volume = vol_file.get_fdata()
    label = lab_file.get_fdata()


    #去除噪声
    #分割胰腺参数为1，分割肿瘤参数为2
    volume = np.clip(volume,-1024,1024)
    
    #增强对比度
    #volume = volume.astype('uint8')
    #volume = F.adjust_contrast(volume, 1.1)
    label = clip_label(label, 1)
    volume = volume.astype(np.float16)
    label = label.astype(np.int64)
    volume = norm_img(volume)
    #取第slice_id片和之前之后的两片CT叠加到一起作为模型的输入,标签只去中间的一片
    for slice_id in range(1, volume.shape[2]-1):
        vol = volume[:,:,slice_id-1:slice_id+2]
        lab = np.fliplr(label[:,:,slice_id])
        if np.sum(lab) < 32:
            continue    
        #(512,512)->(512,512,1)
        lab = lab.reshape([lab.shape[0],lab.shape[1],1])
        #WHC -> CWH
        vol = np.swapaxes(vol,0,2)
        lab = np.swapaxes(lab,0,2)
        #数据来自LITS
        np.save(data_preprocess_path+"lits{}-{}.npy".format(volumes[i].rstrip(".nii").lstrip("volume"), slice_id),vol)
        np.save(label_preprocess_path+"lits{}-{}.npy".format(volumes[i].rstrip(".nii").lstrip("volume"), slice_id),lab)





