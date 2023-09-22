"""
@Author: Penghao Tian <rulenhao@mail.ustc.edu.cn>
@Data: 2023/05/24 17:56
@Desription: 预处理S4max数据, 给出2006~2018年的S4max原始文件

--------------------------------------------------------------------
| year | month | day | hour | time | alt | lat | lon | lct | s4max |
--------------------------------------------------------------------
"""


import sys
import time
import joblib
import einops
import numpy as np
import pandas as pd
import xarray as xr
import configparser
from pathlib import Path
from timezonefinder import TimezoneFinder


class DataPreprocess():
    """
    构建数据集
    """
    
    def __init__(self, 
                 scnlv1_process_dir: str, 
                 start_year: int, 
                 end_year: int,
                 save_dir: str, 
                 save_file_name: str,
                 ) -> None:
        self.scnlv1_process_dir = scnlv1_process_dir
        self.start_year = start_year
        self.end_year = end_year
        self.save_dir = save_dir
        assert Path(self.save_dir).is_dir()
        self.save_file_name = save_file_name
        self.scnlv1_process_files = self._scnlv1_process_file()
        self.timezone_finder = TimezoneFinder(in_memory=True)
        pass
    
    def _scnlv1_process_file(self):
        file_list = []
        for year in range(int(self.start_year), int(self.end_year)+1):
            seek_path = Path(self.scnlv1_process_dir, f"{year}")
            for file in seek_path.rglob("scnLv1_*_nc"):
                file_list.append(file)
                pass
            pass
        
        # 根据进程数分割文件为chunk
        # file_chunk = file_list[self.process_rank:len(file_list):self.world_size]
        return sorted(file_list)
    
    def _read_one_s4max_nc(self, fname):
        """
        读取一个s4max_nc文件 提取相关的时空信息
        """
        with xr.open_dataset(fname, engine="netcdf4") as ds:
            pass
        
        alt = ds.attrs["alttp_s4max"]
        lat = ds.attrs["lattp_s4max"]
        lon = ds.attrs["lontp_s4max"]
        lct = ds.attrs["lcttp_s4max"]
        s4max = ds.attrs["s4max"]
        
        # 去除无效值
        # 出现-999无效值则去除
        if (int(alt)==-999) | (int(lat)==-999) | (int(lon)==-999) | (int(lct)==-999):
            return None
        
        # S4max不在0~5之间去除
        if (int(s4max)<0) | (int(s4max)>5):
            return None
        
        # 根据local time和time zone给出UTC时
        year = ds.attrs["year"]
        month = ds.attrs["month"]
        day = ds.attrs["day"]
        
        lct_part = np.modf(lct, dtype=np.float32)
        decimal = lct_part[0]*3600
        hour = lct_part[1].astype(np.int32)
        minute = (decimal//60).astype(np.int32)
        second = np.floor(decimal%60).astype(np.int32)
        local_zone = self.timezone_finder.timezone_at(lng=lon, lat=lat)
        
        if hour==24:
            hour = 0
            local_time = (pd.Timestamp(year=year, month=month, 
                                       day=day, hour=hour, minute=minute, 
                                       second=second, tz=local_zone) + 
                          pd.Timedelta(1, "day"))
            pass
        else:
            local_time = pd.Timestamp(year=year, month=month, 
                                      day=day, hour=hour, minute=minute, 
                                      second=second, tz=local_zone)
        
        # 把local time转换为正确的s4max位置的UT
        time = local_time.tz_convert("UTC").tz_localize(None) 
        
        df1 = pd.DataFrame([[time]], columns=["time"])
        df2 = pd.DataFrame(np.array([[alt, lat, lon, lct, s4max]], 
                                    dtype=np.float32),
                           columns=["alt", "lat", "lon", "lct", "s4max"])
        df = pd.concat([df1, df2], axis=1)
        
        return df
    
    def read_s4max_nc(self):
        df_list = []
        save_num = 0
        for i, one_s4max_file in enumerate(self.scnlv1_process_files, 1):
            df_list.append(self._read_one_s4max_nc(one_s4max_file))
            # if i>=200:
            #     break
            if i%int(2e6)==0:
                self._save_file(df_list)
                save_num += 1
                print(f"Save file in the period of {save_num}")
                pass
            pass
        # with ThreadPoolExecutor(max_workers=1) as executor:
        #     for i, one_s4max_file in enumerate(self.scnlv1_process_files, 1):
        #         future = executor.submit(self._read_one_s4max_nc, one_s4max_file)
        #         df_list.append(future.result())
        #         # df_list.append(self._read_one_s4max_nc(one_s4max_file))
        #         if i>=2000:
        #             break
        #         if i%int(2e5)==0:
        #             self._save_file(df_list)
        #             save_num += 1
        #             print(f"Save file in the period of {save_num}")
        #             pass
        #         pass
        #     pass
        print(f"Done")
        self._save_file(df_list)
        return
    
    def _save_file(self, df_list):
        raw_df = pd.concat(df_list, axis=0, ignore_index=True)
        
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
        
        raw_df.to_hdf(Path(self.save_dir, self.save_file_name), key="data", mode="w")
        return
    pass


def main():
    
    data_process = DataPreprocess(scnlv1_process_dir="/database9/tianph/ion_dataset/scnLv1", 
                                  start_year=2007,
                                  end_year=2018,
                                  save_dir="/home/tianph/wspace/project_predict_ion/data", 
                                  save_file_name=f"s4max_raw_2007_2018_new.h5",)
    data_process.read_s4max_nc()
    return


if __name__=="__main__":
    from datetime import datetime
    print(f"Start time is {datetime.now()}")
    # GLOBAL_NUM = int(sys.argv[1])
    print("Script start ...")
    t1 = time.time()
    main()
    t2 = time.time()
    print("Data pre-process has completed.")
    print(f"Wall time is {t2-t1}")
    print(f"Complete time is {datetime.now()}")
    pass