"""
@Author: Penghao Tian <rulenhao@mail.ustc.edu.cn>
@Data: 2023/6/6 19:41
@Desription: 
preprocess method
1. 根据数字测高仪的foEs时空信息给出相应的低层大气信息和地磁/太阳活动信息
2. 根据以上信息和超参数进行模型推断
"""


import os
import sys
import time
import torch
import einops
import scipy
import argparse
import numpy as np
import pandas as pd
import xarray as xr
import configparser
from torch import nn
from pathlib import Path
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from numba import njit, prange, objmode

import model as models


class foEsPreprocess():
    """
    基于foEs的数据构造数据集
    包含低层大气信息和太阳地磁
    
    通过给定的foEs的时空信息
    循环给出低层大气和太阳地磁信息
    """
    def __init__(self, foEs_file_path,
                 foEs_lat,
                 foEs_lon,
                 dst_process_dir, 
                 era_process_dir,
                 save_file_name,
                 variable_dict):
        self.station_lat = float(foEs_lat)
        self.station_lon = float(foEs_lon)
        self.era_process_dir = era_process_dir
        self.var_dict = {kk: [int(i) for i in variable_dict[kk]] for kk in variable_dict.keys()}
        self.save_file_name = save_file_name
        self.foEs = self._load_foEs(foEs_file_path)
        self.df_dst_f107 = self._load_dst_f107(dst_process_dir)
        pass
    
    def _load_foEs(self, file_path):
        file = Path(file_path)
        assert file.is_file()
        df = pd.read_csv(file, sep="\s+", header=None, 
                         names=["year", "month", "day", "doy", "hour", "foEs", "alt"])
        
        time_df = pd.DataFrame(columns=["time"])
        for i in range(len(df)):
            year = df.loc[i,  "year"]
            month = df.loc[i, "month"]
            day = df.loc[i, "day"]
            hour = df.loc[i, "hour"]
            time = pd.Timestamp(year, month, day, int(hour))
            time_df.loc[i, "time"] = time
            pass
        df = pd.concat([time_df, df], axis=1)
        df.loc[:, "time"] = pd.to_datetime(df.loc[:, "time"])
        
        # 选择给定的时间范围
        # 2008~2014
        time_series = df.loc[:, "time"]
        select_index = (time_series>=pd.Timestamp(2008,1,1))&(time_series<pd.Timestamp(2015,1,1))
        df = df.loc[select_index, :]
        df.reset_index(inplace=True, drop=True)
        return df
    
    def _load_dst_f107(self, dst_dir):
        file = Path(dst_dir, "omni_hourly_alldata_smallfilled.h5")
        df = pd.read_hdf(file)
        return df
    
    def _find_era_nc_file(self, variable, year, month):
        """
        给出时间和变量
        返回ERA5文件绝对路径
        """
        file_name = Path(self.era_process_dir,
                         f"{year}", 
                         f"era5_reanalysis_{variable}_{year}_{month:02}.nc")
        return file_name
    
    @staticmethod
    def get_variable_name(xr_dataset, variable):
        """
        把物理量字符转化为nc文件中的变量名
        """
        return list(xr_dataset.data_vars)[0]
    
    def _read_era(self, variable, time, lat, lon):
        """
        根据foEs的时间和空间信息选择同一时空的ERA5文件
        之后读取相应ERA5数据
        返回给定ERA5低层大气变量的不同pressure level的数据
        """
        year = time.year
        month = time.month
        day = time.day
        hour = time.hour
        minute = time.minute
        time_stamp = pd.Timestamp(year, month, day, hour, minute)
        
        opt_time = [time_stamp.floor("h"), time_stamp.ceil("h")]
        # 设定时间边界
        if (opt_time[0].year==2019) | (opt_time[1].year==2019):
            opt_time[0] = pd.Timestamp(2018,12,31,23,0)
            opt_time[1] = pd.Timestamp(2018,12,31,23,0)
            pass
        opt_level = self.var_dict[variable]
        
        opt_lat = [np.floor(lat), np.ceil(lat)]
        
        # 设定经度边界
        opt_lon = [np.floor(lon) + 180, np.ceil(lon) + 180]
        if opt_lon[0] == 360:
            opt_lon[0] = 0
            pass
        if opt_lon[1] == 360:
            opt_lon[1] = 0
            pass
        
        fname = self._find_era_nc_file(variable, year, month)
        # 判断每个时间点前后是否在同一年
        # 因为有可能出现2007-12-31 23:20这种情况
        # 此时计算前后两个小时平均值就需要读取2008年1月的数据
        if opt_time[0].month != opt_time[1].month:
            new_year = opt_time[1].year
            new_month = opt_time[1].month
            new_fname = self._find_era_nc_file(variable, new_year, new_month)
            
            with xr.open_dataset(fname) as df1:
                pass
            var_name = self.get_variable_name(df1, variable)
            df_sel1 = df1.sel(time=opt_time[0],
                              level=opt_level,
                              latitude=opt_lat,
                              longitude=opt_lon)
            var1 = df_sel1[var_name].to_numpy()
            var1 = einops.reduce(var1, "level lat lon -> level", "mean")
            
            with xr.open_dataset(new_fname) as df2:
                pass
            df_sel2 = df2.sel(time=opt_time[1],
                              level=opt_level, 
                              latitude=opt_lat,
                              longitude=opt_lon)
            var2 = df_sel2[var_name].to_numpy()
            var2 = einops.reduce(var2, "level lat lon -> level", "mean")
            var = var1 + ((var2-var1)/(opt_time[1].value-opt_time[0].value))*(time_stamp.value-opt_time[0].value)
            pass
        else:
            with xr.open_dataset(fname) as df:
                pass
            var_name = self.get_variable_name(df, variable)
            
            df_sel = df.sel(time=opt_time,
                            level=opt_level,
                            latitude=opt_lat,
                            longitude=opt_lon)
            
            var = df_sel[var_name].to_numpy()
            var = einops.reduce(var, "time level lat lon -> time level", "mean")
            if np.array_equal(var[0], var[1]):
                var = var[0]
                pass
            else:
                var = var[0] + ((var[1]-var[0])/(opt_time[1].value-opt_time[0].value))*(time_stamp.value-opt_time[0].value)
            pass
        var = var.astype(np.float32)
        df_columns = [f"{variable}_{pl}" for pl in opt_level]
        dframe = pd.DataFrame(einops.rearrange(var, "level -> 1 level"), columns=df_columns)
        return dframe
    
    def _calculate_era_from_foEs(self, time, lat, lon):
        """
        对每个大气变量进行循环 计算相应的均值
        """
        df_var_list = []
        for variable in self.var_dict.keys():
            df_var_list.append(self._read_era(variable, time, lat, lon))
            pass
        
        df_var = pd.concat(df_var_list, axis=1)
        return df_var
    
    def _calculate_dst_f107_from_foEs(self, time):
        """
        根据foEs的信息选择Dst F10.7
        利用线性插值
        """
        # 选择Dst和F107
        opt_time = [time.floor("h"), time.ceil("h")]
        
        # 处理边界问题
        # 截至到2019年
        if (opt_time[0]==2019) | (opt_time[1].year==2019):
            opt_time[0] = pd.Timestamp(2018,12,31,23,0)
            opt_time[1] = pd.Timestamp(2018,12,31,23,0)
            pass
        
        # 按线性插值给出相应的dst和F10.7
        # y = y1 + ((y2-y1)/(x2-x1))*(x-x1)
        dst_array = self.df_dst_f107["Dst (nT)"].loc[opt_time].to_numpy(copy=True)
        f107_array = self.df_dst_f107["f10.7"].loc[opt_time].to_numpy(copy=True)
        if np.array_equal(dst_array[0], dst_array[1]):
            dst = dst_array[0]
            pass
        else:
            dst = (dst_array[0] + 
                   ((dst_array[1]-dst_array[0])/(opt_time[1].value-opt_time[0].value))*
                   (time.value-opt_time[0].value))
            pass
        
        if np.array_equal(f107_array[0], f107_array[1]):
            f107 = f107_array[0]
            pass
        else:
            f107 = (f107_array[0] + 
                   ((f107_array[1]-f107_array[0])/(opt_time[1].value-opt_time[0].value))*
                   (time.value-opt_time[0].value))
            pass
        
        df = pd.DataFrame(np.array([[dst, f107]], dtype=np.float32),
                          columns=["dst", "f107"])
        return df
    
    def construct_one_dataset(self, time, lat, lon):
        """
        根据S4max信息构建单个数据
        """
        df_era5 = self._calculate_era_from_foEs(time, lat, lon)
        df_dst_f107 = self._calculate_dst_f107_from_foEs(time)
        df = pd.concat([df_era5, df_dst_f107], axis=1)
        return df
    
    def construct_dataset(self):
        df_lower_list = []
        save_num = 0
        for event in range(len(self.foEs)):
            time = self.foEs.loc[event, "time"]
            lat = self.station_lat
            lon = self.station_lon
            df_lower_list.append(self.construct_one_dataset(time, lat, lon))
            if (event+1)%int(2e5)==0:
                self._save_file(df_lower_list)
                save_num += 1
                print(f"Save file in the period of {save_num}")
                pass
            # if event==10:
            #     break
            pass
        
        foEs_lower = self._save_file(df_lower_list)
        return foEs_lower
    
    def _save_file(self, df_lower_list):
        df_lower = pd.concat(df_lower_list, axis=0, ignore_index=True)
        df = pd.concat([df_lower, self.foEs.iloc[:len(df_lower_list),:]], axis=1)
        df.loc[:, "lat"] = self.station_lat
        df.loc[:, "lon"] = self.station_lon
        df.to_hdf(Path(self.save_file_name), key="data", mode="w",
                  format="fixed")
        return df
    pass


class TransformDataset():
    """
    对foEs数据进行标准化
    选择模型训练时采用的标准化参数
    """
    def __init__(self,
                 df_foEs,
                 std_params_file: str,
                 height_bias
                 ) -> None:
        self.std_params = Path(std_params_file)
        self.height_bias = height_bias
        if (("year" in df_foEs.columns)&
            ("month" in df_foEs.columns)&
            ("day" in df_foEs.columns)&
            ("hour" in df_foEs.columns)):
            pass
        else:
            df_foEs = self._devide_time(df_foEs)
            pass
        self.raw_df_foEs = df_foEs
        self._init_data()
        pass
    
    def _devide_time(self, raw_df):
        """
        将时间变量分割为year month day hour
        """
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
        return raw_df
    
    def _init_data(self):
        
        used_feature = ["year", "month", "day", "hour", "alt", "lat",
                        "lon", "dst", "f107", "u_component_of_wind_5",
                        "u_component_of_wind_10", "u_component_of_wind_50",
                        "geopotential_5", "geopotential_10", "geopotential_50",
                        "temperature_200", "temperature_500"]
        
        ###################################################
        # 如果数据中alt是用的foEs的虚高 这里需要转换
        # 如果用S4max的原始高度 不用转换
        # 根据测高仪虚高和S4max观测的切点高度的关系
        # 调整相应的值
        test_df = self.raw_df_foEs.loc[:, used_feature]
        # 对于Beijing站
        # 线性拟合系数为 60.32226459886587+0.3643283699024251*x
        # 均值为 foEs 112.27517015706808 s4max 101.22731
        # test_df.loc[:, "alt"] = test_df.loc[:, "alt"] * 0.3643283699024251 + 60.32226459886587
        # (112.27517015706808 - 101.22731)
        test_df.loc[:, "alt"] = test_df.loc[:, "alt"] - self.height_bias
        test_data = test_df.to_numpy()
        
        # 读取处理方法3的标准化参数
        assert self.std_params.is_file()
        std_params = np.load(self.std_params)
        scaler = StandardScaler()
        scaler.mean_ = std_params[0]
        scaler.scale_ = std_params[1]
        
        scaled_test_data = scaler.transform(test_data).astype(np.float32)
        
        self.scaled_feature = scaled_test_data
        pass
    
    def __len__(self):
        return self.scaled_feature.shape[0]
    
    def __getitem__(self, index):
        feature = torch.from_numpy(self.scaled_feature[index, :])
        return feature
    pass


class LoadModel():
    
    def __init__(self, 
                 model_name="resnet_bottleneck_50", 
                 feature_in=17,
                 load_model_params_file="./hanhai22/config_02/resnet_bottleneck_50_L1Loss_NAdam_StepLR_3406_2E-04_099.h5") -> None:
        self.load_model_params_file = Path(load_model_params_file)
        assert self.load_model_params_file.is_file()
        self.check_point = torch.load(self.load_model_params_file, map_location="cpu")
        self.model = self._load_model(model_name, feature_in)
        pass
    
    def _load_model(self, name, feature_in):
        layer = []
        if name == "pure_linear":
            layer.append(models.FirstLinear(feature_in=feature_in))
            layer.append(models.LastLinear())
            pass
        
        elif name == "encode_decode":
            layer.append(models.FirstLinear(feature_in=feature_in))
            layer.append(models.Trans1DTo2DLayerWith1Channel())
            layer.append(models.EncodeDecode())
            layer.append(models.Trans2DTo1DLayer())
            layer.append(models.LastLinear())
            pass
        
        elif name == "resnet_bottleneck_50":
            layer.append(models.FirstLinear(feature_in=feature_in))
            layer.append(models.Trans1DTo2DLayerWith3Channel())
            layer.append(models.ResNet("Bottleneck", [3, 4, 6, 3]))
            layer.append(models.LastLinear())
        else:
            raise OSError("undefine model name")
            pass
        
        model = nn.Sequential(*layer)
        
        model.to("cpu")
        model.load_state_dict(self.check_point["model_state_dict"])
        
        return model
    
    pass


class Tester():
    
    def __init__(self,
                 data_set,
                 model,
                 gpu_id) -> None:
        self.data_set = data_set
        self.dataloader = DataLoader(data_set, batch_size=2048, shuffle=False, num_workers=1)
        self.model = model
        self.gpu_id = gpu_id
        self.model.to(gpu_id)
        pass
    
    @torch.no_grad()
    def test(self):
        self.model.eval()
        prediction_list = []
        for source in self.dataloader:
            source = source.to(self.gpu_id)
            output = self.model(source)
            prediction_list.append(output)
            pass
        prediction = torch.concat(prediction_list, axis=0)
        
        prediction = prediction.to("cpu")
        return prediction.detach().numpy()
    pass


def get_bash_args():
    parser = argparse.ArgumentParser(description="simple loop inference job")
    parser.add_argument("--config_file", default="./configs/foEs_inference.ini", type=str, help="The file loaded hyper-parameters")
    parser.add_argument("--gpu_id", default=0, type=int, help="Select the GPU device for inference")
    return parser.parse_args()


def main(foEs_file_path, station_name, foEs_lat, foEs_lon, height_bias):
    
    
    # bash参数获取
    bash_args = get_bash_args()
    
    # ini文件参数获取
    args = configparser.ConfigParser()
    args.read(f"{bash_args.config_file}")
    
    
    # 获取要处理的era5变量
    variables = args.sections()
    variables.remove("init")
    process_era_variable_dict = {}
    for vv in variables:
        if args.getboolean(vv, "opt"):
            plevel = args.get(vv, "pressure_level").replace(" ", "")
            process_era_variable_dict[vv] = plevel.split(",")
            pass
        pass
    
    # 根据foEs原始数据提取低层大气和太阳地磁信息
    data_preprocesser = foEsPreprocess(foEs_file_path=foEs_file_path,
                                       foEs_lat=foEs_lat,
                                       foEs_lon=foEs_lon,
                                       dst_process_dir=args.get("init", "dst_process_dir"),
                                       era_process_dir=args.get("init", "era_process_dir"),
                                       save_file_name=f"./data/ionosonde/{station_name}_with_lower.h5",
                                       variable_dict=process_era_variable_dict)
    
    
    df_foEs_lower = data_preprocesser.construct_dataset()
    
    # 构建测试数据集
    foEs_dataset = TransformDataset(df_foEs=df_foEs_lower,
                                    std_params_file=args.get("init", "standard_params_file"),
                                    height_bias=height_bias
                                   )
    # 构建模型
    model = LoadModel(model_name=args.get("init", "model_name"),
                      feature_in=args.getint("init", "feature_in"),
                      load_model_params_file=args.get("init", "load_model_params_file")
                      ).model
    # 推断
    predict_s4max = Tester(data_set=foEs_dataset,
                           model=model,
                           gpu_id=bash_args.gpu_id).test()
    
    predict_s4max_df = pd.DataFrame(predict_s4max, columns=["test_s4max"])
    total_dataframe = pd.concat([df_foEs_lower, predict_s4max_df], axis=1)
    
    # 保存的文件名字
    load_model_params_file = Path(args.get("init", "load_model_params_file"))
    save_file = Path(load_model_params_file.parent, f"foEs_V3_{station_name}_{load_model_params_file.stem}.h5")
    
    _foEs = total_dataframe.loc[:, "foEs"].to_numpy()
    _test_s4max = total_dataframe.loc[:, "test_s4max"].to_numpy()
    sindex = (_foEs<999)
    _foEs = _foEs[sindex]
    _test_s4max = _test_s4max[sindex]
    R_value, P_value = scipy.stats.spearmanr(_foEs, _test_s4max)
    
    # print(f"Load Model Path: {load_model_params_file}")
    # print(f"Save Path: {save_file.resolve}")
    log = open("./logs/foEs_height_bias.log", "a")
    print(f"{R_value}: {height_bias}", file=log)
    log.close()
    # print(total_dataframe.head(10))
    # print(total_dataframe.info())
    # print(total_dataframe.columns)
    total_dataframe.to_hdf(save_file, key="data", mode="w")
    
    return





if __name__ == "__main__":
    import datetime
    print(f"Script start in {datetime.datetime.now():%Y-%m-%d %H:%M:%S}")
    
    file_list = [(Path("./data/ionosonde/BP440_ALL200606-201612.TXT"), 40.3, 116.2),
                 (Path("./data/ionosonde/MH453_ALL201009-201612.TXT"), 52.0, 122.5),
                 (Path("./data/ionosonde/SA418_ALL200710-201612.TXT"), 18.3, 109.4),
                 (Path("./data/ionosonde/SH427_ALL201204-201612.TXT"), 27.1, 111.3),
                 (Path("./data/ionosonde/WU430_ALL201004-201612.TXT"), 30.5, 114.4)]
    
    # for (file, lat, lon) in file_list:
    #     # if "BP440" in file.stem:
    #     station_name = file.stem.split("_")[0]
    #     main(foEs_file_path=file, station_name=station_name, foEs_lat=lat, foEs_lon=lon)
    #     print(f"{station_name} is Done")
    #     print("====="*10)
    #     break
    #     pass
    hbias_array = np.arange(0, 30, 0.1)
    
    file, lat, lon = file_list[0]
    station_name = file.stem.split("_")[0]
    for hbias in hbias_array:
        main(foEs_file_path=file, station_name=station_name, foEs_lat=lat, foEs_lon=lon, height_bias=hbias)
        pass
    print("Done")
    print(f"Script done in {datetime.datetime.now():%Y-%m-%d %H:%M:%S}")
