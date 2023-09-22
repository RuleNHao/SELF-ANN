"""
@Author: Penghao Tian
@Date: 2022/08/31 22:27
@Description: Implement the download of raw data set
including ERA5, COSMIC RO raw data.
"""

import os
import sys
import shutil
import requests
import tarfile
from concurrent import futures
import configparser
import cdsapi
import calendar


def _RM_ALL(dirs):
    """
    移除所有的文件夹以及子集
    """
    shutil.rmtree(dirs)
    return None


class Download_COSMIC_scnLv1():
    """
    Download raw COSMIC scnLv1 data
    Extract the zip archive of the original web
    and unzip it.
    Web: http://cosmic.ucar.edu
    
    @Attributes:
        file_prefix: 下载的文件标识符号
        save_dir: 保存的目录
        start_year: 开始年份
        end_year: 结束年份
        url: 连接的网址
        threads: 开启多线程下载的线程数
    """
    def __init__(self, save_dir, start_year, end_year, 
                 url, file_prefix, threads):
        self.file_prefix = file_prefix
        self.save_dir = save_dir
        self.start_year = start_year
        self.end_year = end_year
        self.url = url
        self.threads = threads
        pass
    
    def _untar(self, fname, dirs):
        """
        unzip the file with tar.gz suffix
        
        @Args:
            fname: tar.gz file path
            dirs: the save path
        """
        try:
            # 确保解压的文件夹不包含其他文件
            assert len(os.listdir(dirs))==1
            t = tarfile.open(fname)
            t.extractall(path=dirs)
            pass
        except:
            pass
        os.remove(fname)
        return None
    
    def _download_one_file(self, yy, dd):
        year = yy
        doy = str(dd).zfill(3)
        target_url = os.path.join(self.url, 
                                  str(year), str(doy), 
                                  f"{self.file_prefix}_{year}_{doy}.tar.gz")
        r = requests.get(target_url, stream=True)

        save_path = os.path.join(self.save_dir, str(year), str(doy))
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        file_path = os.path.join(save_path, f"{self.file_prefix}_{year}_{doy}.tar.gz")
        with open(file_path, "wb") as f:
            f.write(r.content)
            pass

        self._untar(file_path, save_path)

        # 删除空文件夹, 即没有记录的year/doy数据
        if not os.listdir(save_path):
            os.rmdir(save_path)
            pass
        return 
    
    def download_data(self,):
        """
        采用多线程下载
        """
        with futures.ThreadPoolExecutor(max_workers=self.threads) as executor:
            for yy in range(int(self.start_year), int(self.end_year)+1):
                for dd in range(1, 367):
                    executor.submit(self._download_one_file, yy, dd)
                    pass
                pass
            pass
        return None
    pass


class Download_ERA():
    """
    下载ERA5数据
    使用CDS API
    默认为小时分辨率, 全天
    
    @Attributes:
        save_dir: 保存的文件路径
        start_year: 开始年份
        end_year: 结束年份
        variable_dict: 需要下载的变量及其压力面, 如 {"temperature": ["1", "2"]}
        grid: 网格的分辨率, 自定义为1x1, 默认再分析为0.25x0.25
    """
    def __init__(self, save_dir, start_year, end_year, variable_dict, grid):
        self.save_dir = save_dir
        self.start_year = start_year
        self.end_year = end_year
        assert isinstance(variable_dict, dict)
        self.variable_dict = variable_dict
        assert isinstance(grid, list)
        self.grid = grid
        self.C = cdsapi.Client()
        pass
    
    def _retrieve_one_file(self, year, month, var):
        day =[f"{dd:02}" for dd in range(1, calendar.monthrange(year, month)[1]+1)]
        file_name = f"era5_reanalysis_{var}_{year}_{month:02}.nc"
        file_path = os.path.join(self.save_dir, f"{year}", file_name)
        
        self.C.retrieve(
            'reanalysis-era5-pressure-levels',
            {
            'product_type': 'reanalysis',
            'format': 'netcdf',
            'variable': f"{var}",
            'pressure_level': self.variable_dict.get(var),
            'year': f"{year}",
            'month': f"{month:02}",
            'day': day,
            'time': [
                '00:00', '01:00', '02:00',
                '03:00', '04:00', '05:00',
                '06:00', '07:00', '08:00',
                '09:00', '10:00', '11:00',
                '12:00', '13:00', '14:00',
                '15:00', '16:00', '17:00',
                '18:00', '19:00', '20:00',
                '21:00', '22:00', '23:00',],
            'grid': self.grid, 
            }, 
            file_path)
        return 
    
    def download_era5(self,):
        for year in range(int(self.start_year), 1+int(self.end_year)):
            if not os.path.exists(os.path.join(self.save_dir, str(year))):
                os.mkdir(os.path.join(self.save_dir, str(year)))
                pass
            for month in range(1, 13):
                for var in self.variable_dict.keys():
                    self._retrieve_one_file(year, month, var)
                    pass
                pass
            pass
        return
    pass


def subroutine_download_cosmic(args):
    assert args.has_section("scnLv1")
    secs = args["scnLv1"]
    start_year = secs.getint("start_year")
    end_year = secs.getint("end_year")
    assert start_year <= end_year
    download_client = []
    
    download_client_repro2013 = Download_COSMIC_scnLv1(save_dir=secs.get("save_dir"), 
                                                       file_prefix=secs.get("file_prefix_1"),
                                                       start_year=start_year, end_year=2014,
                                                       url=secs.get("url_2006_2014"), 
                                                       threads=secs.getint("threads"))
        
    download_client_postProc = Download_COSMIC_scnLv1(save_dir=secs.get("save_dir"), 
                                                       file_prefix=secs.get("file_prefix_2"),
                                                       start_year=2014, end_year=end_year,
                                                       url=secs.get("url_2014_2020"), 
                                                       threads=secs.getint("threads"))
    
    download_client.extend([download_client_repro2013, download_client_postProc])
    for dc in download_client:
        dc.download_data()
        pass
    return

def subroutine_download_era(args):
    start_year = args.get("era", "start_year")
    end_year = args.get("era", "end_year")
    _grid = args.get("era", "grid").replace(" ", "")
    grid = [int(i) for i in _grid.split(",")]
    
    variables = args.sections()
    variables.remove("era")
    var_dict = {}
    for vv in variables:
        if args.getboolean(vv, "opt"):
            plevel = args.get(vv, "pressure_level").replace(" ", "")
            var_dict[vv] = plevel.split(",")
            pass
        pass
    download_client = Download_ERA(save_dir=args.get("era", "save_dir"), start_year=start_year,
                                   end_year=end_year, variable_dict=var_dict,
                                   grid=grid)
    download_client.download_era5()
    # print(var_dict)
    return
    
def main():
    args = configparser.ConfigParser()
    args.read(f"{sys.argv[1]}")
    if args.has_section("cosmic"):
        subroutine_download_cosmic(args)
        pass
    elif args.has_section("era"):
        subroutine_download_era(args)
        pass
    else:
        raise OSError("Error configuration file")
        pass
    return
    
if __name__=="__main__":
    main()
    print("Download has completed.")
    pass
