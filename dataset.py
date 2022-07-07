'''
generate the dataset:
dataset数据准备文件
'''
from collections import namedtuple
from email.mime import base
from math import radians
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
'''
完整的Mipnerf Dataset类的实现:
继承:Dataset
'''
class MipDataset(Dataset):
    def __init__(self,base_dir,split,spherify=False,near=2,far=6,whith_background=False,factor=1,n_poses=120,radius=None,radii=None,h=None,w=None,device=torch.device("cpu")):
        super(Dataset,self).__init__()
        self.base_dir = base_dir
        self.split = split
        self.spherify = spherify
        self.near = near
        self.far = far
        self.white_background = whith_background
        self.factor = factor
        self.n_poses = n_poses
        self.n_poses_copy = n_poses
        self.radius = radius
        self.radii = radii
        self.h = h
        self.w = w
        self.device = device
        self.rays = None
        self.images = None
        self.load()
    
    def load(self):
        if self.split == "render":
            self.generate_render_rays()
        else:
            self.generate_training_rays()
        self.flatten()  # 还不知道此步骤是干嘛的
        print("Done")
    
    def generate_training_poses(self):
        '''
        纯虚函数:在具体的数据集中进行实现
        '''
        raise ValueError('no generate_training_poses().')
    
    def generate_render_poses(self):
        '''
        自由视角相机位姿生成
        '''
        self.focal = 1200 # 这里不太确定
        self.n_poses = self.n_poses_copy
        if self.spherify:
            self.generate_spherical_poses(self.n_poses) # 生成球形视角函数
        else:
            self.generate_spiral_pose(self.n_poses) #生成环形视角函数
    
    def generate_spherical_poses(self,n_poses=120):
        #TODO:搞明白poses到底是哪里的坐标系
        self.poses = generate_spherical_cam_to_world(self.radius,n_poses)
        self.cam_to_world = self.poses[:,:3,:4]
    
    def generate_spiral_poses(self,n_poses=120):
        #TODO:搞明白这里为什么不用poses了
        self.cam_to_world = generate_spiral_cam_to_world(self.radii,self.focal,n_poses)
    
    def generate_training_rays(self):
        '''
        生成训练射线
        '''
        print('Generating Training Poses')
        self.generate_training_poses()
        print('Generating Training Rays')
        self.generate_rays()
    
    def generate_render_rays(self):
        '''
        生成渲染射线
        '''
        print('Generating Render Poses')
        self.generate_render_poses()
        print('Generating Render Rays')
        self.generate_rays()

    def generate_rays(self):
        '''
        生成射线函数:小孔相机模型
        输入信息: self.h,self.w,self.focal,self.cam_to_world
        '''
        x,y = np.meshgrid(
            np.arange(self.w,dtype=np.float32), # X轴 (矩阵的列)
            np.arange(self.h,dtype=np.float32), # Y轴 (矩阵的行)
            indexing='xy'
        )
        camera_directions = np.stack(
            [(x - self.w * 0.5 + 0.5) / self.focal,
             -(y - self.h * 0.5 + 0.5) / self.focal,
             -np.ones_like(x)],
            axis=-1
        )
        # 旋转射线操作 from camera to the world
        directions = ((camera_directions[None, ..., None, :] * self.cam_to_world[:, None, None, :3, :3]).sum(axis=-1))  # Translate camera frame's origin to the world frame
        origins = np.broadcast_to(self.cam_to_world[:, None, None, :3, -1], directions.shape)
        viewdirs = directions / np.linalg.norm(directions, axis=-1, keepdims=True)

        # 每条射线到X轴近邻的2-范数距离
        dx = np.sqrt(np.sum((directions[:, :-1, :, :] - directions[:, 1:, :, :]) ** 2, -1))
        dx = np.concatenate([dx, dx[:, -2:-1, :]], 1)

        # 论文中提到的细节 像素->半径
        radii = dx[..., None] * 2 / np.sqrt(12)

        ones = np.ones_like(origins[..., :1])

        self.rays = Rays(
            origins=origins,
            directions=directions,
            viewdirs=viewdirs,
            radii=radii,
            lossmult=ones,
            near=ones * self.near,
            far=ones * self.far)
        
        def flatten(self):
            if self.rays is not None:
                self.rays = namedtuple_map(lambda r: torch.tensor(r).float().reshape([-1, r.shape[-1]]), self.rays)
            if self.images is not None:
                self.images = torch.from_numpy(self.images.reshape([-1, 3]))
        








dataset_dict = {
    'blender': Blender,
    'llff': LLFF,
    'multicam': Multicam,
}