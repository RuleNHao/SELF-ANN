import os
import sys
import time
import torch
import argparse
import numpy as np
import configparser
from pathlib import Path
from datetime import datetime
import torch.distributed as dist
from torch.distributed import init_process_group, destroy_process_group

import train
import inference



def global_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    return

def ddp_setup():
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
    # os.environ["MASTER_ADDR"] = "localhost"
    # os.environ["MASTER_PORT"] = "12356"
    env_dict = {
        key: os.environ[key]
        for key in ("MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE")
    }
    print(f"[{os.getpid()}] Initializing process group with: {env_dict}")
    init_process_group(backend="nccl")
    print(
        f"[{os.getpid()}] world_size = {dist.get_world_size()}, "
        + f"rank = {dist.get_rank()}, backend={dist.get_backend()}"
    )
    
    # init_process_group(backend="nccl")
                       # init_method="tcp://localhost:12355")
                       # init_method=f"file://{Path().cwd().joinpath('sharedfile').resolve()}")
    # print(f"Initialize process group complete # GPU::{rank} ")
    return


def train_routine(args_config):
    
    # 固定种子
    global_seed(args_config.getint("train_params", "seed"))
    
    # 创建通信群
    ddp_setup()
    
    # 训练流程开始
    dataloader = train.PrepareDataLoader(is_train=True,
                                         data_parallel=True,
                                         dataset_path=args_config.get("train_params", "dataset_path"),
                                         process_method=args_config.getint("train_params", "process_method"),
                                         split_method=args_config.getint("train_params", "split_method"),
                                         batch_size=args_config.getint("train_params", "batch_size"),
                                         num_workers=args_config.getint("train_params", "num_workers")
                                        ).prepare_dataloader()
    model = train.DefineModel(args_config.get("train_params", "model_name"), 
                              args_config.getint("train_params", "feature_in"),
                              args_config.get("train_params", "load_model_params_file")
                             ).model
    trainer = train.Trainer(dataloader=dataloader,
                            model_name=args_config.get("train_params", "model_name"),
                            model=model,
                            gpu_id=dist.get_rank(),
                            loss_func_name=args_config.get("train_params", "loss_func_name"),
                            optimizer_name=args_config.get("train_params", "optimizer_name"),
                            lr_scheduler_name=args_config.get("train_params", "lr_scheduler_name"),
                            lr=args_config.getfloat("train_params", "lr"),
                            max_epoch=args_config.getint("train_params", "max_epoch"),
                            save_model_dir=args_config.get("train_params", "save_model_dir"),
                            save_log_file=args_config.get("train_params", "save_log_file"),
                            args_config=args_config
                           )
    trainer.train()
    
    # 销毁通信群
    destroy_process_group()
    return 


def inference_routine(args_config):
    
    # 创建通信群
    ddp_setup()
    
    model = inference.DefineModel(model_name=args_config.get("inference_params", "model_name"),
                                  feature_in=args_config.getint("inference_params", "feature_in"),
                                  load_model_params_file=args_config.get("inference_params", "load_model_params_file")
                                 ).model
    dataloader, test_df = inference.PrepareDataLoader(is_train=False,
                                                      data_parallel=True,
                                                      dataset_path=args_config.get("inference_params", "dataset_path"),
                                                      process_method=args_config.getint("inference_params", "process_method"),
                                                      batch_size=args_config.getint("inference_params", "batch_size"),
                                                      num_workers=args_config.getint("inference_params", "num_workers")
                                                     ).prepare_dataloader()
    tester = inference.Tester(dataloader=dataloader,
                              test_df=test_df,
                              gpu_id=dist.get_rank(),
                              model=model,
                              inference_file=args_config.get("inference_params", "inference_file"),
                              args_config=args_config
                             )
    tester.test()
    
    # 销毁通信群
    destroy_process_group()
    pass


def Other_ML_algorithm():
    pass


def get_bash_args():
    parser = argparse.ArgumentParser(description="simple distributed training job")
    parser.add_argument("--assign", default="DDP_train", type=str, help="Determinate train or inference")
    parser.add_argument("--config_file", default="./configs/config.ini", type=str, help="The file loaded hyper-parameters")
    return parser.parse_args()


if __name__=="__main__":
    print(f"Script start in {datetime.now():%Y-%m-%d %H:%M:%S}")
    
    # bash参数获取
    bash_args = get_bash_args()
    
    # ini文件参数获取
    args_config = configparser.ConfigParser()
    args_config.read(f"{bash_args.config_file}")
    
    if bash_args.assign == "DDP_train":
        train_routine(args_config)
        pass
    
    elif bash_args.assign == "DDP_inference":
        inference_routine(args_config)
        pass
    else:
        pass
    
    print(f"Script end in {datetime.now():%Y-%m-%d %H:%M:%S}")
    pass
        