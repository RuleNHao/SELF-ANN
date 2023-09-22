"""
推断所有数据
2008-2014
不区分训练集和测试集

如果采用不同的处理方法
需要自行更改方法
"""


import os
import sys
import time
import torch
import einops
import numpy as np
import pandas as pd
import xarray as xr
import argparse
import configparser
from torch import nn
from pathlib import Path
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

import model


class AllDataset(Dataset):
    """
    与Dataset一致
    但是返回所有的正则化数据
    
    处理方法是固定的
    """
    def __init__(self,
                 file_path: str,
                 ) -> None:
        self.file_path = Path(file_path)
        self._init_dataset()
        pass
    
    def _init_dataset(self):
        assert self.file_path.is_file()
        raw_df = pd.read_hdf(self.file_path)
        alt_series = raw_df.loc[:, "alt"]
        s4max_series = raw_df.loc[:, "s4max"]
        used_index = (alt_series>=80)&(alt_series<=130)&(s4max_series>=0)&(s4max_series<=5)
        used_df = raw_df.loc[used_index, :]
        used_df.reset_index(drop=True, inplace=True)
        
        # self._process_method_2(used_df)
        self._process_method_3(used_df)
        return
    
    def _process_method_3(self, raw_df):
        """
        全体数据: 2008-1 ~ 2014-12
        训练: 全部的75%
        验证: 全部的5%
        测试: 全部数据中均匀的随机选取20%
        feature: year, month, day, hour, alt, lat, lon, dst, f107, 
                u_5, u_10, u_50, geopential_5/10/50,
                temperature_200/500
        label: s4max
        """
        
        # 将时间变量分割为year month day hour
        time_series = raw_df.loc[:, "time"]
        year_month_day_hour = np.zeros((len(time_series),4))
        for i, tt in enumerate(time_series):
            year_month_day_hour[i,0] = np.array(tt.year, dtype=np.float32)
            year_month_day_hour[i,1] = np.array(tt.month, dtype=np.float32)
            year_month_day_hour[i,2] = np.array(tt.day, dtype=np.float32)
            year_month_day_hour[i,3] = np.array((tt.hour + tt.minute/60 + tt.second/3600),
                                                dtype=np.float32)
            pass
        time_df = pd.DataFrame(year_month_day_hour, columns=["year", "month", "day", "hour"])
        raw_df = pd.concat([time_df, raw_df], axis=1)
        
        used_feature = ["year", "month", "day", "hour", "alt", "lat",
                        "lon", "dst", "f107", "u_component_of_wind_5",
                        "u_component_of_wind_10", "u_component_of_wind_50",
                        "geopotential_5", "geopotential_10", "geopotential_50",
                        "temperature_200", "temperature_500", "s4max"]
        
        # 读取标准化参数
        scaler = StandardScaler()
        standard_params = np.load("./data/process_3_standard_params.npy")
        scaler.mean_ = standard_params[0]
        scaler.scale_ = standard_params[1]
        
        all_used_feature_df = raw_df.loc[:, used_feature]
        
        self.scaled_all_used_feature = scaler.transform(all_used_feature_df.to_numpy()[:, :-1]).astype(np.float32)
        
        self.dataframe = raw_df.loc[:, ["time", "year", "month", "day", "hour", 
                                        "alt", "lat", "lon", "lct", "s4max"]]
        self.scaler = scaler
        return
        
    def _process_method_2(self, raw_df):
        """
        训练: 2008-1 ~ 2013-6 2014-6 ~ 2014-12
        测试: 2013-6 ~ 2014-6
        feature: year, month, day, hour, alt, lat, lon, dst, f107, 
                u_5, u_10, u_50, geopential_5/10/50,
                temperature_200/500
        label: s4max
        """
        
        # 将时间变量分割为year month day hour
        time_series = raw_df.loc[:, "time"]
        year_month_day_hour = np.zeros((len(time_series),4))
        for i, tt in enumerate(time_series):
            year_month_day_hour[i,0] = np.array(tt.year, dtype=np.float32)
            year_month_day_hour[i,1] = np.array(tt.month, dtype=np.float32)
            year_month_day_hour[i,2] = np.array(tt.day, dtype=np.float32)
            year_month_day_hour[i,3] = np.array((tt.hour + tt.minute/60 + tt.second/3600),
                                                dtype=np.float32)
            pass
        time_df = pd.DataFrame(year_month_day_hour, columns=["year", "month", "day", "hour"])
        raw_df = pd.concat([time_df, raw_df], axis=1)
        
        
        # 给定训练和测试时间
        train_time_index = (((time_series>=pd.Timestamp(2008,1,1,0,0,0))&
                            (time_series<pd.Timestamp(2013,6,1,0,0,0)))|
                            ((time_series>=pd.Timestamp(2014,6,1,0,0,0))&
                            (time_series<pd.Timestamp(2015,1,1,0,0,0))))
        # 选择给定的feature
        used_feature = ["year", "month", "day", "hour", "alt", "lat",
                        "lon", "dst", "f107", "u_component_of_wind_5",
                        "u_component_of_wind_10", "u_component_of_wind_50",
                        "geopotential_5", "geopotential_10", "geopotential_50",
                        "temperature_200", "temperature_500", "s4max"]
        
        
        train_data = raw_df.loc[train_time_index, used_feature].to_numpy()
        # 拟合训练数据
        scaler = StandardScaler()
        scaler.fit(train_data[:,:-1])
        
        all_used_feature_df = raw_df.loc[:, used_feature]
        
        self.scaled_all_used_feature = scaler.transform(all_used_feature_df.to_numpy()[:, :-1]).astype(np.float32)
        
        self.dataframe = raw_df
        self.scaler = scaler
        
        # 保存拟合参数
        # result[0] mean
        # result[1] std
        # np.save(f"./web_app/standard_params_{self.process_method}.npy",
        #         np.concatenate([scaler.mean_[None,None,:], 
        #                         scaler.scale_[None,None,:]], axis=0))
        return
        
    def __len__(self):
        return self.scaled_all_used_feature.shape[0]
    
    def __getitem__(self, index):
        feature = torch.from_numpy(self.scaled_all_used_feature[index, :])
        return feature
    pass


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
                 dataset_path: str,
                 batch_size: int,
                 num_workers: int,) -> None:
        self.dataset_path = Path(dataset_path)
        self.batch_size = batch_size
        self.num_workers = num_workers
        pass
    
    def prepare_dataloader(self):
        # 给出dataset
        data_set = AllDataset(self.dataset_path)
        
        # 给出相应的dl
        data_dl = DataLoader(data_set, batch_size=self.batch_size,
                             num_workers=self.num_workers, shuffle=False, 
                             pin_memory=True)
        
        # 返回dl和含有原label的dataframe
        return data_dl, data_set.dataframe
    
    pass


class Tester():
    
    def __init__(self,
                 dataloader,
                 test_df,
                 gpu_id,
                 model,
                 save_file,
                 ) -> None:
        self.dataloader = dataloader
        self.test_df = test_df
        self.model = model
        self.gpu_id = gpu_id
        self.save_file = Path(save_file)
        self.model = self.model.to(self.gpu_id)
        pass
    
    @torch.no_grad()
    def test(self):
        """
        测试集给出结果
        """
        self.model.eval()
        prediction_list = []
        for source in self.dataloader:
            source = source.to(self.gpu_id)
            output = self.model(source)
            prediction_list.append(output)
            pass
        prediction = torch.concat(prediction_list, axis=0).cpu()
        
        self.test_df.loc[:, "test_s4max"] = prediction
        self.test_df.to_hdf(self.save_file, key="data", mode="w")
        return
    pass


def get_bash_args():
    parser = argparse.ArgumentParser(description="simple loop inference job")
    parser.add_argument("--gpu_id", default=0, type=int, help="Select the GPU device for inference")
    return parser.parse_args()


def main(load_params_path, gpu_id):
    model = DefineModel(model_name="resnet_bottleneck_50",
                        feature_in=17,
                        load_model_params_file=load_params_path
                       ).model
    dataloader, test_df = PrepareDataLoader(dataset_path="./data/s4max_process_lower.h5",
                                            batch_size=4096,
                                            num_workers=4
                                           ).prepare_dataloader()
    tester = Tester(dataloader=dataloader,
                    test_df=test_df,
                    gpu_id=gpu_id,
                    model=model,
                    save_file=load_params_path.parent.joinpath(f"Infer_2008_2014_{load_params_path.stem}.h5"),
                   )
    tester.test()


if __name__=="__main__":
    print(f"Script start in {datetime.now():%Y-%m-%d %H:%M:%S}")
    print("Start ...")
    # bash参数获取
    bash_args = get_bash_args()
    load_params_path = Path("./hanhai22/config_02/resnet_bottleneck_50_L1Loss_NAdam_StepLR_3406_2E-04_099.ckpt")
    
    # params_path_list = [f"./hanhai20/config_01/resnet_bottleneck_50_L1Loss_NAdam_None_3406_2E-04_{n}.ckpt" for n in range(150,200)]
    
    # params_path_list = [f"./config_07/resnet_bottleneck_50_L1Loss_NAdam_None_3406_1E-04_{n}.ckpt" for n in range(180,200)]
    
    # params_path_list = [f"./config_08/resnet_bottleneck_50_L1Loss_NAdam_None_3406_2E-04_{n}.ckpt" for n in range(180,200)]
    main(load_params_path, bash_args.gpu_id)
    
    print(f"Script end in {datetime.now():%Y-%m-%d %H:%M:%S}")
    pass

