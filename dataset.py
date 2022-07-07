'''
generate the dataset:
dataset数据准备文件
'''
from email.mime import base
import os 
import cv2
import json
import torch
import numpy as np
from utils import *
from PIL import Image
from torch.utils.data import Dataset,DataLoader


'''
实例化dataset类
'''
def get_dataset(dataset_name,base_dir,split,factor=4,device=torch.device("cuda")):
    return dataset_dict[dataset_name](base_dir,split,factor,device)

'''
实例化dataloader类
'''
def get_dataloader(dataset_name,base_dir,split,factor=4,batch_size=None,shuffle=True,device=torch.device("cuda")):
    d = get_dataset(dataset_name,base_dir,split,factor,device)
    # video render单独设置batchsize,同时不打乱数据集
    if split == 'render':
        batch_size = d.w * d.h
        shuffle = False
    loader = DataLoader(d,batch_size=batch_size,shuffle = shuffle)
    loader.h = d.h
    loader.w = d.w
    loader.near = d.near
    loader.far = d.far
    return loader

# 不确定是否需要此函数
# def cycle(iterable):
#     while True:
#         for x in iterable:
#             yield x



dataset_dict = {
    'blender': Blender,
    'llff': LLFF,
    'multicam': Multicam,
}