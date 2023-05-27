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


class SequentialDistributedSampler(torch.utils.data.sampler.Sampler):
    """
    Distributed Sampler that subsamples indicies sequentially,
    making it easier to collate all results at the end.
    Even though we only use this sampler for eval and predict (no training),
    which means that the model params won't have to be synced (i.e. will not hang
    for synchronization even if varied number of forward passes), we still add extra
    samples to the sampler to make it evenly divisible (like in `DistributedSampler`)
    to make it easy to `gather` or `reduce` resulting tensors at the end of the loop.
    """

    def __init__(self, dataset, batch_size, rank=None, num_replicas=None):
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = torch.distributed.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.batch_size = batch_size
        self.num_samples = int(np.ceil(len(self.dataset) * 1.0 / self.batch_size / self.num_replicas)) * self.batch_size
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        # add extra samples to make it evenly divisible
        indices += [indices[-1]] * (self.total_size - len(indices))
        # subsample
        indices = indices[self.rank * self.num_samples : (self.rank + 1) * self.num_samples]
        return iter(indices)

    def __len__(self):
        return self.num_samples
    
    pass


class PrepareDataLoader():
    
    def __init__(self, 
                 is_train: bool,
                 data_parallel: bool,
                 dataset_path: str,
                 process_method: int,
                 batch_size: int,
                 num_workers: int,) -> None:
        self.is_train = is_train
        self.data_parallel = data_parallel
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
                             pin_memory=True, 
                             sampler=SequentialDistributedSampler(data_set, batch_size=self.batch_size))
        
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
        
        # 同步化BN层
#         BN_tag = False
#         for mm in self.model.modules():
#             if isinstance(mm, nn.BatchNorm2d):
#                 BN_tag = True
#                 break
#             pass
#         if BN_tag:
#             self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
#             pass
        
        # 使用DDP方案
        self.model = self.model.to(self.gpu_id)
        self.model = DDP(self.model, device_ids=[self.gpu_id])
        return 
    
    @staticmethod
    def distributed_concat(tensor, num_total_examples):
        output_tensors = [tensor.clone() for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(output_tensors, tensor)
        concat = torch.cat(output_tensors, dim=0)
        # truncate the dummy elements added by SequentialDistributedSampler
        # 因为sampler中为了确保所有张量大小一致 把数据延长了
        # 现在要去掉多余数据
        return concat[:num_total_examples]
    
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
        prediction = self.distributed_concat(torch.concat(prediction_list, axis=0),
                                             len(self.dataloader.sampler.dataset))
        target = self.distributed_concat(torch.concat(target_list, axis=0),
                                         len(self.dataloader.sampler.dataset))
        
        if self.gpu_id == 0:
            self.test_df.loc[:, "target_s4max"] = target.to("cpu")
            self.test_df.loc[:, "test_s4max"] = prediction.to("cpu")
            self.test_df.to_hdf(self.inference_file, key="data", mode="w")
            pass
        return
