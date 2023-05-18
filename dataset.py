import torch
import einops
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from torch.utils.data import Dataset, random_split
from sklearn.preprocessing import StandardScaler


class MyDataset(Dataset):
    
    def __init__(self,
                 is_train: bool,
                 file_path: str,
                 process_method: int
                 ) -> None:
        self.is_train = is_train
        self.file_path = Path(file_path)
        self.process_method = process_method
        self._init_dataset()
        pass
    
    def _init_dataset(self):
        assert self.file_path.is_file()
        df = pd.read_hdf(self.file_path)
        if self.process_method == 0:
            self._process_method_0(df)
            pass
        elif self.process_method == 1:
            self._process_method_1(df)
            pass
        elif self.process_method == 2:
            self._process_method_2(df)
            pass
        return
    
    def _process_method_0(self, raw_df):
        """
        训练: 2008-1 ~ 2013-6 2014-6 ~ 2014-12
        测试: 2013-6 ~ 2014-6
        feature: year, month, day, hour, alt, lat, lon, dst, f107
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
        
        test_time_index = ((time_series>=pd.Timestamp(2013,6,1,0,0,0))&
                           (time_series<pd.Timestamp(2014,6,1,0,0,0)))
        
        train_df = raw_df.loc[train_time_index, :]
        test_df = raw_df.loc[test_time_index, :]
        
        train_df.reset_index(drop=True, inplace=True)
        test_df.reset_index(drop=True, inplace=True)
        
        train_data = train_df.loc[:, ["year", "month", "day", "hour",
                                      "alt", "lat", "lon", "dst", "f107", 
                                      "s4max"]].to_numpy()
        test_data = test_df.loc[:, ["year", "month", "day", "hour", 
                                    "alt", "lat", "lon", "dst", "f107",
                                    "s4max"]].to_numpy()
        
        train_feature = train_data[:, :-1]
        train_label = train_data[:, [-1]]
        test_feature = test_data[:, :-1]
        test_label = test_data[:, [-1]]
        
        # 拟合训练数据
        scaler = StandardScaler()
        scaler.fit(train_feature)
        
        if self.is_train:
            scaled_feature = scaler.transform(train_feature)
            train_data = np.concatenate([scaled_feature, train_label], axis=1).astype(np.float32)
            self.data = train_data
            return 
        else:
            scaled_feature = scaler.transform(test_feature)
            test_data = np.concatenate([scaled_feature, test_label], axis=1).astype(np.float32)
            self.data = test_data
            self.dataframe = test_df
            return 
    
    def _process_method_1(self, raw_df):
        """
        ######### TODO more data process method
        训练: 2008-1 ~ 2013-6 2014-6 ~ 2014-12
        测试: 2013-6 ~ 2014-6
        feature: year, month, day, hour, alt, lat, lon, dst, f107, 
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
        
        test_time_index = ((time_series>=pd.Timestamp(2013,6,1,0,0,0))&
                           (time_series<pd.Timestamp(2014,6,1,0,0,0)))
        
        train_df = raw_df.loc[train_time_index, :]
        test_df = raw_df.loc[test_time_index, :]
        
        train_df.reset_index(drop=True, inplace=True)
        test_df.reset_index(drop=True, inplace=True)
        
        train_data = train_df.loc[:, ["year", "month", "day", "hour",
                                      "alt", "lat", "lon", "dst", "f107", 
                                      "s4max"]].to_numpy()
        test_data = test_df.loc[:, ["year", "month", "day", "hour", 
                                    "alt", "lat", "lon", "dst", "f107",
                                    "s4max"]].to_numpy()
        
        train_feature = train_data[:, :-1]
        train_label = train_data[:, [-1]]
        test_feature = test_data[:, :-1]
        test_label = test_data[:, [-1]]
        
        # 拟合训练数据
        scaler = StandardScaler()
        scaler.fit(train_feature)
        
        if self.is_train:
            scaled_feature = scaler.transform(train_feature)
            train_data = np.concatenate([scaled_feature, train_label], axis=1).astype(np.float32)
            self.data = train_data
            return 
        else:
            scaled_feature = scaler.transform(test_feature)
            test_data = np.concatenate([scaled_feature, test_label], axis=1).astype(np.float32)
            self.data = test_data
            self.dataframe = test_df
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
        
        test_time_index = ((time_series>=pd.Timestamp(2013,6,1,0,0,0))&
                           (time_series<pd.Timestamp(2014,6,1,0,0,0)))
        
        train_df = raw_df.loc[train_time_index, :]
        test_df = raw_df.loc[test_time_index, :]
        
        train_df.reset_index(drop=True, inplace=True)
        test_df.reset_index(drop=True, inplace=True)
        
        # 选择给定的feature
        used_feature = ["year", "month", "day", "hour", "alt", "lat",
                        "lon", "dst", "f107", "u_component_of_wind_5",
                        "u_component_of_wind_10", "u_component_of_wind_50",
                        "geopotential_5", "geopotential_10", "geopotential_50",
                        "temperature_200", "temperature_500", "s4max"]
        ################################################
        # 改变一下dst和f107值
        # test_df.loc[:, ["dst", "f107"]] = test_df.loc[:, ["dst", "f107"]] * 2
        
        ###############################################
        train_data = train_df.loc[:, used_feature].to_numpy()
        test_data = test_df.loc[:, used_feature].to_numpy()
        
        train_feature = train_data[:, :-1]
        train_label = train_data[:, [-1]]
        test_feature = test_data[:, :-1]
        test_label = test_data[:, [-1]]
        
        # 拟合训练数据
        scaler = StandardScaler()
        scaler.fit(train_feature)
        
        # 保存拟合参数
        # result[0] mean
        # result[1] std
        # np.save(f"./web_app/standard_params_{self.process_method}.npy",
        #         np.concatenate([scaler.mean_[None,None,:], 
        #                         scaler.scale_[None,None,:]], axis=0))
        
        
        if self.is_train:
            scaled_feature = scaler.transform(train_feature)
            train_data = np.concatenate([scaled_feature, train_label], axis=1).astype(np.float32)
            self.data = train_data
            return 
        else:
            scaled_feature = scaler.transform(test_feature)
            test_data = np.concatenate([scaled_feature, test_label], axis=1).astype(np.float32)
            self.data = test_data
            self.dataframe = test_df
            return 
    
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, index):
        feature = torch.from_numpy(self.data[index, :-1])
        label = torch.from_numpy(self.data[index, [-1]])
        return (feature, label)
    pass


def split_to_train_valid(data_set, split_method=0):
    """
    分割为训练集和验证集
    """
    if split_method==0:
        # 随机选1/10为验证集
        n_train = int(len(data_set)*0.9)
        n_valid = len(data_set) - n_train
        dataset_train, dataset_valid = random_split(data_set, [n_train, n_valid])
        pass
    
    else:
        pass
    
    return dataset_train, dataset_valid
    

if __name__=="__main__":
    obj = MyDataset(is_train=False, 
                    file_path="./data/s4max_process_lower.h5",
                    process_method=2)
    print("Done")
    