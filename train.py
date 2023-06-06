"""
@Author: Penghao Tian <rulenhao@mail.ustc.edu.cn>
@Date: 2022/10/6 00:00
@Description: 通过数据并行方法实现模型训练
"""


import os
import sys
import time
import torch
import einops
import argparse
import numpy as np
import pandas as pd
import xarray as xr
import configparser
from torch import nn
from pathlib import Path
from datetime import datetime
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

import model
import dataset


class DefineModel():
    
    def __init__(self, 
                 model_name: str, 
                 feature_in: int,
                 load_model_params_file: str) -> None:
        self.model = self._define_model(model_name, feature_in)
        try:
            params_file = Path(load_model_params_file)
            if params_file.is_file():
                ckpt = torch.load(params_file, map_location="cpu")
                self.model.to("cpu")
                self.model.load_state_dict(ckpt["model_state_dict"])
                pass
            pass
        except:
            pass
        
        return
    
    def _define_model(self, name, feature_in):
        layer = []
        if name == "pure_linear":
            layer.append(model.FirstLinear(feature_in=feature_in))
            layer.append(model.LastLinear())
            pass
        
        elif name == "encode_decode":
            layer.append(model.FirstLinear(feature_in=feature_in))
            layer.append(model.Trans1DTo2DLayerWith1Channel())
            layer.append(model.EncodeDecode())
            layer.append(model.Trans2DTo1DLayer())
            layer.append(model.LastLinear())
            pass
        
        elif name == "resnet_bottleneck_50":
            layer.append(model.FirstLinear(feature_in=feature_in))
            layer.append(model.Trans1DTo2DLayerWith3Channel())
            layer.append(model.ResNet("Bottleneck", [3, 4, 6, 3]))
            layer.append(model.LastLinear())
        else:
            raise OSError("undefine model name")
            pass
        return nn.Sequential(*layer)
    
    pass


class PrepareDataLoader():
    
    def __init__(self, 
                 is_train: bool,
                 data_parallel: bool,
                 dataset_path: str,
                 process_method: int,
                 split_method,
                 batch_size: int,
                 num_workers: int,) -> None:
        self.is_train = is_train
        self.data_parallel = data_parallel
        self.dataset_path = Path(dataset_path)
        self.process_method = process_method
        self.split_method = split_method
        self.batch_size = batch_size
        self.num_workers = num_workers
        pass
    
    def prepare_dataloader(self):
        # 给出dataset
        data_set = dataset.MyDataset(self.is_train, self.dataset_path, self.process_method)
        
        # 给出分割训练/验证的方法
        train_ds, valid_ds = dataset.split_to_train_valid(data_set, split_method=self.split_method)
        train_dl = DataLoader(train_ds, batch_size=self.batch_size,
                              num_workers=self.num_workers, shuffle=False, 
                              pin_memory=True, sampler=DistributedSampler(train_ds))

        valid_dl = DataLoader(valid_ds, batch_size=self.batch_size,
                              num_workers=self.num_workers, shuffle=False, 
                              pin_memory=True, sampler=DistributedSampler(valid_ds))
        
        return (train_dl, valid_dl)
    
    pass


class Trainer():
    
    def __init__(self,
                 dataloader,
                 model_name,
                 model,
                 gpu_id,
                 loss_func_name,
                 optimizer_name,
                 lr_scheduler_name,
                 lr,
                 max_epoch,
                 save_model_dir,
                 save_log_file,
                 args_config
                ):
        self.train_dl, self.valid_dl = dataloader
        self.model_name = model_name
        self.model = model
        self.gpu_id = gpu_id
        self.loss_func_name = loss_func_name
        self.optimizer_name = optimizer_name
        self.lr_scheduler_name = lr_scheduler_name
        self.lr = lr
        self.max_epoch = max_epoch
        self.save_model_dir = Path(save_model_dir)
        self.save_log_file = Path(save_log_file)
        
        self.args_config = args_config
        self._init_train_method()
        self.log = pd.DataFrame(columns=["time", "train_loss", "valid_loss", "valid_correlation"], 
                                index=range(self.max_epoch))
        pass
    
    def _init_train_method(self):
        
        # 同步化BN层
        BN_tag = False
        for mm in self.model.modules():
            if isinstance(mm, nn.BatchNorm2d):
                BN_tag = True
                break
            pass
        if BN_tag:
            self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            pass
        
        # 使用DDP方案
        self.model = self.model.to(self.gpu_id)
        self.model = DDP(self.model, device_ids=[self.gpu_id])
        # 注意要DDP model后再创建优化器
        # 因为DDP保证了模型在各个GPU上参数一致
        # 再创建优化器后 优化器也一致
        
        # 给出loss
        if self.loss_func_name=="L1Loss":
            self.loss_func = nn.L1Loss()
            pass
        elif self.loss_func_name=="MSELoss":
            self.loss_func = nn.MSELoss()
            pass
        elif self.loss_func_name=="CustomLoss01":
            self.loss_func = model.CustomLoss01
        else:
            raise OSError("undefine loss function")
        
        # 给出优化器
        if self.optimizer_name=="Adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
            pass
        elif self.optimizer_name=="Adamw":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr,
                                              weight_decay=0.00025)
            pass
        elif self.optimizer_name=="NAdam":
            self.optimizer = torch.optim.NAdam(self.model.parameters(), lr=self.lr)
            pass
        elif self.optimizer_name=="SGD":
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr,
                                             momentum=0.9)
            pass
        elif self.optimizer_name=="RMSprop":
            self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=self.lr)
            pass
        else:
            raise OSError("undefine optimizer")
        
        # 给出lr调度方案
        if self.lr_scheduler_name=="StepLR":
            self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 
                                                                step_size=50, gamma=0.5)
            pass
        elif self.lr_scheduler_name=="CosineAnnealingLR":
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer,
                                                                           T_max=5, 
                                                                           eta_min=5e-6)
            pass
        elif self.lr_scheduler_name=="ReduceLROnPlateau":
            self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                                           factor=0.5,
                                                                           patience=5,
                                                                           )
            pass
        else:
            self.lr_scheduler = None
            pass
        
        return 
    
    
    def _run_train_batch(self, source, targets):
        self.optimizer.zero_grad()
        output = self.model(source)
        loss = self.loss_func(output, targets)
        loss.backward()
        self.optimizer.step()
        return loss
    
    def _run_train_epoch(self, epoch):
        """
        训练一个epoch 收集所有GPU上的Loss求均值
        """
        self.model.train()
        train_loss = []
        for source, targets in self.train_dl:
            source = source.to(self.gpu_id)
            targets = targets.to(self.gpu_id)
            train_loss.append(self._run_train_batch(source, targets))
            pass
        train_loss = torch.stack(train_loss, axis=0)
        dist.all_reduce(train_loss, dist.ReduceOp.AVG)
        if self.gpu_id == 0:
            self.log.loc[epoch, "time"] = f"{datetime.now():%Y-%m-%d %H:%M:%S}"
            self.log.loc[epoch, "train_loss"] = torch.mean(train_loss).to("cpu").detach().numpy()
            pass
        return
    
    @torch.no_grad()
    def _run_valid_epoch(self, epoch):
        """
        验证集上求loss
        """
        self.model.eval()
        valid_loss = []
        valid_cor = []
        for source, targets in self.valid_dl:
            source = source.to(self.gpu_id)
            targets = targets.to(self.gpu_id)
            output = self.model(source)
            loss = self.loss_func(output, targets).item()
            # 给出pearson corrcoef
            correlation = torch.corrcoef(torch.concat([output.detach().cpu().T, targets.detach().cpu().T]))[0,1]
            valid_loss.append(loss)
            valid_cor.append(correlation)
            pass
        valid_loss = torch.stack(valid_loss, axis=0)
        valid_cor = torch.stack(valid_cor, axis=0)
        dist.all_reduce(valid_loss, dist.ReduceOp.AVG)
        dist.all_reduce(valid_cor, dist.ReduceOp.AVG)
        if self.gpu_id == 0:
            self.log.loc[epoch, "valid_loss"] = torch.mean(valid_loss).to("cpu").detach().numpy()
            self.log.loc[epoch, "valid_correlation"] = torch.mean(valid_cor).to("cpu").detach().numpy()
            pass
        return torch.mean(valid_loss)
    
    def train(self):
        for epoch in range(self.max_epoch):
            self.train_dl.sampler.set_epoch(epoch)
            self.valid_dl.sampler.set_epoch(epoch)
            self._run_train_epoch(epoch)
            valid_loss = self._run_valid_epoch(epoch)
            if self.lr_scheduler_name=="ReduceLROnPlateau":
                self.lr_scheduler.step(valid_loss)
                pass
            elif self.lr_scheduler_name=="CosineAnnealingLR":
                self.lr_scheduler.step()
                pass
            elif self.lr_scheduler_name=="StepLR":
                self.lr_scheduler.step()
                pass
            else:
                pass
            if self.gpu_id == 0:
                self._save_log()
                if ((self.max_epoch - epoch) >= 0) & ((self.max_epoch - epoch) <= 20):
                    self._save_checkpoint(epoch)
                    pass
                # if epoch >= 150:
                #     self._save_checkpoint(epoch)
                #     pass
                pass
            pass
        return
    
    def _save_checkpoint(self, epoch):
        save_dict = {}
        save_dict["info"] = {sec: list(self.args_config[sec].items()) for sec in self.args_config.sections()}
        save_dict["epoch"] = epoch
        save_dict["model_state_dict"] = self.model.module.state_dict()
        # save_dict["optimizer_state_dict"] = self.optimizer.state_dict()
        ckpt_name = (f"{self.model_name}_" +
                     f"{self.loss_func_name}_" +
                     f"{self.optimizer_name}_" +
                     f"{self.lr_scheduler_name}_"+
                     f"{self.args_config.getint('train_params', 'seed')}_"+
                     f"{self.lr:1.0E}_"+
                     f"{epoch:03}.ckpt")
        torch.save(save_dict, self.save_model_dir.joinpath(ckpt_name))
        print(f"Epoch {epoch} | Training checkpoint saved at {self.save_model_dir.joinpath(ckpt_name)}")
        return 
    
    def _save_log(self):
        self.log.to_csv(self.save_log_file, mode="w")
        print(f"Log file saved in {self.save_log_file}")
    
    pass

