from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import xarray as xr
import time
import einops
import configparser



@njit()
def calculate_dis(lat1, lon1, lat2, lon2):
    """
    计算两个点的地球表明距离
    """
    R = 6378.137
    lat1 = np.full_like(lat2, lat1)
    lon1 = np.full_like(lon2, lon1)
    lat1, lon1, lat2, lon2 = map(np.deg2rad, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    dist = R * c
    return dist

def select_data(foEs_file, station_lat, station_lon, s4max_file):
    """
    给定S4max的数据
    按照时空网格将foEs数据和S4max数据匹配
    """
    s4max_df = pd.read_hdf(s4max_file)
    
    foEs_df = pd.read_csv(foEs_file, sep="\s+", header=None, 
                          names=["year", "month", "day", "doy", "UT", "foEs", "h"])
    
    # 加入Timestamp列
    time_df = pd.DataFrame(columns=["time"])
    for i in range(len(foEs_df)):
        year = foEs_df.loc[i,  "year"]
        month = foEs_df.loc[i, "month"]
        day = foEs_df.loc[i, "day"]
        hour = foEs_df.loc[i, "UT"]
        time = pd.Timestamp(year, month, day, int(hour))
        time_df.loc[i, "time"] = time
        pass
    foEs_df = pd.concat([time_df, foEs_df], axis=1)
    
    foEs_result_df = pd.DataFrame(columns=["alt", "s4max", "test_s4max", "foEs_alt", "foEs"], dtype=np.float32)
    foEs_result_df_index = 0
    s4max_res_list = []
    # 限制时空取平均
    for i in range(len(foEs_df)):
        
        # limite time
        tt1 = foEs_df.loc[i, "time"] + pd.Timedelta(0.5, "H")
        tt2 = foEs_df.loc[i, "time"] - pd.Timedelta(0.5, "H")
        
        # limite space
        # 北京站 40.3 116.2
        lat1 = station_lat + 2.5
        lat2 = station_lat - 2.5
        lon1 = station_lon + 2.5
        lon2 = station_lon - 2.5
        
        time_series = s4max_df.loc[:, "time"]
        lat_series = s4max_df.loc[:, "lat"]
        lon_series = s4max_df.loc[:, "lon"]
        
        select_index = ((time_series>=tt2) & (time_series<tt1) &
                        (lat_series>=lat2) & (lat_series<=lat1) &
                        (lon_series>=lon2) & (lon_series<=lon1))
                        # (alt_series>=alt2) & (alt_series<=alt1))
        s4max_res = s4max_df.loc[select_index, :]
        if len(s4max_res) >= 1:
            alt = np.mean(s4max_res.loc[:, "alt"].to_numpy())
            s4max = np.mean(s4max_res.loc[:, "s4max"].to_numpy())
            test_s4max = np.mean(s4max_res.loc[:, "test_s4max"].to_numpy())
            
            foEs_result_df.loc[foEs_result_df_index, "alt"] = alt
            foEs_result_df.loc[foEs_result_df_index, "s4max"] = s4max
            foEs_result_df.loc[foEs_result_df_index, "test_s4max"] = test_s4max
            foEs_result_df.loc[foEs_result_df_index, "foEs_alt"] = foEs_df.loc[i, "h"]
            foEs_result_df.loc[foEs_result_df_index, "foEs"] = foEs_df.loc[i, "foEs"]
            foEs_result_df_index += 1
            pass
        # if foEs_result_df_index >= 10:
        #     break
        pass
    # s4max_res_list_df = pd.concat(s4max_res_list, axis=0, ignore_index=True)
    # output_df = pd.concat([s4max_res_list_df, foEs_result_df], axis=1)
    output_df = foEs_result_df
    return output_df



if __name__=="__main__":
    # s4max_file_path = Path("./data/s4max_process_lower.h5")
    # args = configparser.ConfigParser()
    # args.read(f"{sys.argv[1]}")
    print("Script start ...")
    # s4max_file = Path("./hanhai20/config_01/Infer_2008_2014_resnet_bottleneck_50_L1Loss_NAdam_None_3406_2E-04_159.h5")
    s4max_file = Path("./hanhai22/config_02/Infer_2008_2014_resnet_bottleneck_50_L1Loss_NAdam_StepLR_3406_2E-04_099.h5")
    # s4max_file = Path("./hanhai20/simple_config_01/Infer_resnet_bottleneck_50_L1Loss_NAdam_StepLR_3406_1E-04_049.h5")
    
    
    # infer_file_list = []
    
    # 每个地基观测的文件与s4max_file筛选同一时空
    file_list = [(Path("./data/ionosonde/BP440_ALL200606-201612.TXT"), 40.3, 116.2),
                 (Path("./data/ionosonde/MH453_ALL201009-201612.TXT"), 52.0, 122.5),
                 (Path("./data/ionosonde/SA418_ALL200710-201612.TXT"), 18.3, 109.4),
                 (Path("./data/ionosonde/SH427_ALL201204-201612.TXT"), 27.1, 111.3),
                 (Path("./data/ionosonde/WU430_ALL201004-201612.TXT"), 30.5, 114.4)]
    for (file, lat, lon) in file_list:
        # if "BP440" in file.stem:
        station_name = file.stem.split("_")[0]
        save_file = Path(s4max_file.parent, s4max_file.name.replace("Infer", f"foEs_V1_{station_name}"))
        df = select_data(file, lat, lon, s4max_file)
        print(df.info())
        print(df.columns.to_list())
        print(df.head(10))
        print(f"{station_name} is Done")
        print("====="*10)
        # break
        df.to_hdf(save_file, key="data", mode="w")
        pass
    pass
