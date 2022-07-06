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
    config.add_argument('--log_dir',type=str,default='log')
    config.add_argument('--dataset_name',type=str,default='blender')
    config.add_argument("--scene", type=str, default="lego")

    # model hyperparams 模型超参
    config.add_argument("--use_viewdirs", action="store_false")
    config.add_argument("--randomized", action="store_false")
    config.add_argument("--ray_shape", type=str, default="cone")  # llff:"cylinder"
    config.add_argument("--white_bkgd", action="store_false")  # llff:False
    config.add_argument("--override_defaults", action="store_true")
    config.add_argument("--num_levels", type=int, default=2)
    config.add_argument("--num_samples", type=int, default=128) #采样点数量
    config.add_argument("--hidden", type=int, default=256) #隐藏层数量
    config.add_argument("--density_noise", type=float, default=0.0) #体密度高斯噪声
    config.add_argument("--density_bias", type=float, default=-1.0)
    config.add_argument("--rgb_padding", type=float, default=0.001)
    config.add_argument("--resample_padding", type=float, default=0.01)
    # TODO:deg?存疑,待之后代码注释
    config.add_argument("--min_deg", type=int, default=0)
    config.add_argument("--max_deg", type=int, default=16)
    config.add_argument("--viewdirs_min_deg", type=int, default=0)
    config.add_argument("--viewdirs_max_deg", type=int, default=4)

    # loss / optimizer 超参
    config.add_argument("--coarse_weight_decay", type=float, default=0.1)
    config.add_argument("--lr_init", type=float, default=1e-3)
    config.add_argument("--lr_final", type=float, default=5e-5)
    config.add_argument("--lr_delay_steps", type=int, default=2500)
    config.add_argument("--lr_delay_mult", type=float, default=0.1)
    config.add_argument("--weight_decay", type=float, default=1e-5)

    # training 超参
    config.add_argument("--factor", type=int, default=2)
    config.add_argument("--max_steps", type=int, default=10000)
    config.add_argument("--batch_size", type=int, default=2048)
    config.add_argument("--do_eval", action="store_false")
    config.add_argument("--continue_training", action="store_true")
    config.add_argument("--save_every", type=int, default=1000)
    config.add_argument("--device", type=str, default="cuda")

    # 可视化超参
    config.add_argument("--chunks", type=int, default=8192)
    config.add_argument("--model_weight_path", default="log/model.pt")
    config.add_argument("--visualize_depth", action="store_true") #可视化深度
    config.add_argument("--visualize_normals", action="store_true") #可视化法相

    config = config.parse_args()
    # default configs for llff, automatically set if dataset is llff and not override_defaults
    if config.dataset_name == "llff" and not config.override_defaults:
        config.factor = 4
        config.ray_shape = "cylinder"
        config.white_bkgd = False
        config.density_noise = 1.0
    
    config.device = torch.device(config.device)
    base_data_path = "data/nerf_llff_data/"
    if config.dataset_name == "blender":
         base_data_path = "data/nerf_synthetic/"
    elif config.dataset_name == "multicam":
        base_data_path = "data/nerf_multiscale/"
    config.base_dir = path.join(base_data_path, config.scene)

    return config


















