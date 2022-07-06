'''
config file:
参数配置文件
'''
import argparse
import torch 
from os import path

def get_config():
    config = argparse.ArgumentParser()

    # load/save data 路径参数
    config.add_argument('--log_dir',type=str,default='log',)

