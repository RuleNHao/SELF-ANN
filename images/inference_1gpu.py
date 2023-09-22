"""
用单个GPU推断模型
数据预处理方法选择在dataset.py中定义的子方法
模型定义在model.py中的class中
"""


import os
import sys
import time
import torch
import einops
import numpy as np
import pandas as pd
import xarray as xr
import configparser
from torch import nn
from pathlib import Path
from datetime import datetime
from torch.utils.data import DataLoader

import model
import dataset


class DefineModel():
    
    def __init__(self, 
                 model_name: str, 
                 feature_in: int,
                 load_model_params_file: str) -> None:
        self.model = self._define_model(model_name, feature_in)
        params_file = Path(load_model_params_file)
        if params_file.is_file():
            ckpt = torch.load(params_file, map_location="cpu")
            self.model.to("cpu")
            self.model.load_state_dict(ckpt["model_state_dict"])
        
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
                 dataset_path: str,
                 process_method: int,
                 batch_size: int,
                 num_workers: int,) -> None:
        self.is_train = is_train
        self.dataset_path = Path(dataset_path)
        self.process_method = process_method
        self.batch_size = batch_size
        self.num_workers = num_workers
        pass
    
    def prepare_dataloader(self):
        # 给出dataset
        data_set = dataset.MyDataset(self.is_train, self.dataset_path, self.process_method)
        
        # 给出相应的dl
        test_dl = DataLoader(data_set, batch_size=self.batch_size,
                             num_workers=self.num_workers, shuffle=False, 
                             pin_memory=True)
        
        # 返回dl和含有原label的dataframe
        return test_dl, data_set.dataframe
    
    pass


class Tester():
    
    def __init__(self,
                 dataloader,
                 test_df,
                 gpu_id,
                 model,
                 inference_file,
                 args_config) -> None:
        self.dataloader = dataloader
        self.test_df = test_df
        self.model = model
        self.gpu_id = gpu_id
        self.inference_file = Path(inference_file)
        self._init_train_method()
        self.args_config = args_config
        pass
    
    def _init_train_method(self):
        
        self.model = self.model.to(self.gpu_id)
        return 
    
    @torch.no_grad()
    def test(self):
        """
        测试集给出结果
        """
        self.model.eval()
        prediction_list = []
        target_list = []
        for source, targets in self.dataloader:
            source = source.to(self.gpu_id)
            targets = targets.to(self.gpu_id)
            output = self.model(source)
            prediction_list.append(output)
            target_list.append(targets)
            pass
        prediction = torch.concat(prediction_list, axis=0)
        target = torch.concat(target_list, axis=0)
        
        target = target.to("cpu")
        prediction = prediction.to("cpu")
        self.test_df.loc[:, "target_s4max"] = target
        self.test_df.loc[:, "test_s4max"] = prediction
        # print(f"target: {target[5678:5688]}")
        # print(f"prediction: {prediction[5678:5688]}")
        print(f"{torch.corrcoef(torch.concat([target.T, prediction.T]))[0,1]}")
        self.test_df.to_hdf(self.inference_file, key="data", mode="w")
        pass
        return
