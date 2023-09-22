"""
@Author: Penghao Tian <rulenhao@mail.ustc.edu.cn>
@Date: 2022/10/16 00:59
@Description: 数据可视化
"""


import os
import sys
import scipy
import einops
import netCDF4
import sklearn
import cartopy
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib as mpl
from pathlib import Path
import sklearn.metrics as sklm
import matplotlib.pyplot as plt
from numba import njit, prange, objmode
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression

# IGRF
import igrf


class _PlotBase():
    """
    绘图基本类
    一些特定设置
    """
    def __init__(self, 
                 backend="agg",
                 file_path="./data/s4max_process.h5",
                 ) -> None:
        if backend == "agg":
            self.filetype = "png"
            pass
        elif backend == "svg":
            self.filetype = "svg"
            pass
        elif backend == "pdf":
            self.filetype = "pdf"
            pass
        elif backend == "ps":
            self.filetype = "eps"
            pass
        else:
            raise OSError(f"This is an error backend for {backend}")
        mpl.rcParams["backend"] = backend
        mpl.rcParams["font.family"] = "DejaVu Serif"
        mpl.rcParams["font.style"] = "normal"
        mpl.rcParams["font.weight"] = "normal"
        
        self._load_data(file_path)
        return
    
    def _load_data(self, file_path):
        file_path = Path(file_path)
        df = pd.read_hdf(file_path)
        
        s4max_str = "test_s4max" if "inference" in file_path.stem else "s4max"
        
        time_series = df.loc[:, "time"]
        time_index = (time_series >= pd.Timestamp("2012-1-1")) & (time_series < pd.Timestamp("2013-1-1"))
        use_df = df.loc[time_index, ["time", "alt", "lat", "lon", "lct", s4max_str]]
        use_df.reset_index(drop=True, inplace=True)
        
        alt_series = use_df.loc[:, "alt"]
        alt_index = (alt_series >= 80) & (alt_series <= 130)
        use_df = use_df.loc[alt_index, :]
        data = use_df.loc[:, ["alt", "lat", "lon", "lct", s4max_str]].values.astype(np.float64)
        self.alt = data[:, 0]
        self.lat = data[:, 1]
        self.lon = data[:, 2]
        self.lct = data[:, 3]
        self.s4max = data[:, 4]
        
        self.doy = use_df.loc[:, "time"].dt.day_of_year.values
        
        return
    
    def _save_fig(self):
        picture_path = Path(f"./figure.{self.filetype}")
        if self.filetype=="png":
            self.fig.savefig(picture_path, dpi=450)
            pass
        else:
            self.fig.savefig(picture_path)
        return
    
    pass


class FeatureImportance(_PlotBase):
    
    def __init__(self, 
                 backend: str, 
                 file_path: str,
                 index_name: str,
                 is_pressure: bool
                 ) -> None:
        self.is_pressure = is_pressure
        self.index_name = index_name
        super().__init__(backend, file_path)
        self._private_color = [(158/255, 202/255, 225/255),
                               (8/255, 48/255, 107/255),
                              ]
        pass
    
    def _load_data(self, file_path):
        file_path = Path(file_path)
        assert file_path.is_file()
        df = pd.read_hdf(file_path)
        
        if self.index_name=="mdi":
            if self.is_pressure=="every":
                # 按照压力面排序
                sort_df = df.sort_values(["mdi"], axis="columns")
                cols = sort_df.columns.values.tolist()
                mdi_data = sort_df.loc["mdi",:].values
                self.sorted_name = cols
                self.data = mdi_data
                pass
            elif self.is_pressure=="level":
                # 按照大气分层排序
                atom_df = self._trans2atom(df, "mdi")
                sort_df = atom_df.sort_values(["mdi"], axis="columns")
                cols = sort_df.columns.values.tolist()
                mdi_data = sort_df.loc["mdi",:].values
                self.sorted_name = cols
                self.data = mdi_data
                pass
            elif self.is_pressure=="all":
                # 把整个低层大气平均
                all_df = self._trans2all(df, "mdi")
                sort_df = all_df.sort_values(["mdi"], axis="columns")
                cols = sort_df.columns.values.tolist()
                mdi_data = sort_df.loc["mdi",:].values
                self.sorted_name = cols
                self.data = mdi_data
                
            pass
        elif self.index_name=="permut":
            if self.is_pressure=="every":
                # 按照permutation的mean值排序
                permu_data = df.iloc[1:,:].to_numpy(copy=True)
                permu_mean = permu_data.mean(axis=0)
                new_df = pd.DataFrame(permu_mean[None,:], 
                                      index=["permut"], columns=df.columns)
                new_df = pd.concat([df, new_df], axis=0)

                sort_df = new_df.sort_values(["permut"], axis="columns")
                cols = sort_df.columns.values.tolist()
                permut_data = sort_df.loc["permut",:].values
                self.sorted_name = cols
                
                self.data = permut_data
                pass
            elif self.is_pressure=="level":
                # 按照大气分层排序
                permu_data = df.iloc[1:,:].to_numpy(copy=True)
                permu_mean = permu_data.mean(axis=0)
                new_df = pd.DataFrame(permu_mean[None,:], 
                                      index=["permut"], columns=df.columns)
                new_df = pd.concat([df, new_df], axis=0)
                
                atom_df = self._trans2atom(new_df, "permut")
                
                sort_df = atom_df.sort_values(["permut"], axis="columns")
                cols = sort_df.columns.values.tolist()
                permut_data = sort_df.loc["permut",:].values
                self.sorted_name = cols
                self.data = permut_data
                pass
            elif self.is_pressure=="all":
                # 按照整个大气排序 不分层和压力面
                permu_data = df.iloc[1:,:].to_numpy(copy=True)
                permu_mean = permu_data.mean(axis=0)
                new_df = pd.DataFrame(permu_mean[None,:], 
                                      index=["permut"], columns=df.columns)
                new_df = pd.concat([df, new_df], axis=0)
                
                atom_df = self._trans2all(new_df, "permut")
                
                sort_df = atom_df.sort_values(["permut"], axis="columns")
                cols = sort_df.columns.values.tolist()
                permut_data = sort_df.loc["permut",:].values
                self.sorted_name = cols
                self.data = permut_data
        return

    @staticmethod
    def _trans2all(df, index_name):
        assert (index_name=="mdi")|(index_name=="permut")
        cols = df.columns.tolist()
        data = df.loc[index_name,:].to_numpy()

        new_cols = []
        new_data = []

        i = 0
        for _ in range(9):
            new_cols.append(f"{'_'.join(cols[i].split('_')[:-1])}_all")
            new_data.append(data[i:i+5].mean())
            i += 5
            pass
        nn_df = pd.DataFrame([new_data], columns=new_cols, index=[index_name])
        
        # 剩下的data内包括alt lat lon 这些基本数据
        rest_df = df.loc[[index_name],:].iloc[:, 45:]
        return pd.concat([nn_df, rest_df], axis=1)

    @staticmethod
    def _trans2atom(df, index_name):
        assert (index_name=="mdi")|(index_name=="permut")
        cols = df.columns.tolist()
        data = df.loc[index_name,:].to_numpy()
        
        new_cols = []
        new_data = []
        
        i = 0
        for _ in range(9):
            new_cols.append(f"{'_'.join(cols[i].split('_')[:-1])}_strato")
            new_data.append(data[i:i+3].mean())
            
            new_cols.append(f"{'_'.join(cols[i].split('_')[:-1])}_tropo")
            new_data.append(data[i+3:i+5].mean())
            i += 5
            pass
        nn_df = pd.DataFrame([new_data], columns=new_cols, index=[index_name])
        rest_df = df.loc[[index_name],:].iloc[:, 45:]
        return pd.concat([nn_df, rest_df], axis=1)

    def _draw_axes_0(self, ax):
        
        
        # 数据标准化
        # self.data = (self.data - self.data.min()) / (self.data.max()-self.data.min())
        # self.data = (self.data - self.data.mean()) / self.data.std()
        # self.data
        
        
        # bbox = ax.get_position()
        # ax.set_position([bbox.xmin+0.15, bbox.ymin, bbox.width, bbox.height])
        y = np.arange(len(self.sorted_name))
        contain = ax.barh(y, width=self.data, height=1, left=0, align="center",
                          color=self._private_color[0], edgecolor=self._private_color[1],
                          linewidth=0.9)
        if self.index_name=="mdi":
            ax.text(0.5, 0.965, "Mean Decrease Impurity Feature Importance", fontsize=18,
                    transform=self.fig.transFigure, ha="center", va="center")
            pass
        else:
            ax.set_title(f"Permutation Feature Importance", fontsize=15)
        # ax.set_xlim(0,0.7)
        # ax.set_ylim(-1,
        replace_dict = {"alt": "Altitude", "lct": "Local Time", "lat": "Geographic Latitude",
                        "lon": "Geographic Longitude", "doy": "Day of Year", 
                        "f107": "F10.7 Index", "dst": "Dst", 
                        "u_component_of_wind_strato": "U Component of Wind (S)",
                        "u_component_of_wind_tropo": "U Component of Wind (T)",
                        "geopotential_strato": "Geopotential (S)",
                        "geopotential_tropo": "Geopotential (T)",
                        "temperature_strato": "Temperature (S)",
                        "temperature_tropo": "Temperature (T)",
                        "potential_vorticity_strato": "Potential Vorticity (S)",
                        "potential_vorticity_tropo": "Potential Vorticity (T)",
                        "divergence_strato": "Divergence (S)",
                        "divergence_tropo": "Divergence (T)",
                        "vertical_velocity_strato": "Vertical Velocity (S)",
                        "vertical_velocity_tropo": "Vertical Velocity (T)",
                        "v_component_of_wind_strato": "V Component of Wind (S)",
                        "v_component_of_wind_tropo": "V Component of Wind (T)",
                        "vorticity_strato": "Vorticity (S)",
                        "vorticity_tropo": "Vorticity (T)",
                        "relative_humidity_strato": "Relative Humidity (S)",
                        "relative_humidity_tropo": "Relative Humidity (T)",
                        "year": "Year"}
        ax.set_yticks(ticks=y, 
                      labels=[replace_dict[x] if x in replace_dict else x for x in self.sorted_name], 
                      fontsize=13, fontweight="normal")
        
        
        
        if self.index_name=="mdi":
            for x, yy in zip(contain.datavalues, y):
                ax.text(x+0.001, yy, f"{x:.4f}", ha="left", va="center_baseline",
                        fontsize=13)
                if x>0.1:
                    ax.text(x+0.001, yy, f"{x:.4f}", ha="left", va="center_baseline",
                            fontsize=12, color="r")
        else:
            for x, yy in zip(contain.datavalues, y):
                ax.text(x+0.002, yy, f"{x:.4f}", ha="left", va="center_baseline",
                        fontsize=14)
                if x>0.03:
                    ax.text(x+0.002, yy, f"{x:.4f}", ha="left", va="center_baseline",
                            fontsize=14, color="r")
                    pass
                pass
            pass
        
        # 微调图片
        ax.spines[["right", "top"]].set_visible(False)
        ax.spines["left"].set_bounds(low=-0.5, high=24.5)
        ax.spines["bottom"].set_position(("axes", 0.02))
        
        
        xticks = np.arange(0, 0.1, 0.02)
        ax.set_xticks(ticks=xticks, 
                      labels=[f"{perce:.2f}" for perce in xticks],
                      fontsize=13)
        
        ax.set_xlabel("Gini index", fontsize=15, labelpad=10)
        
        ax.axhline(y=14.5, xmin=0, xmax=0.99, color="red",
                   ls="--", lw=2)
        con = mpl.patches.ConnectionPatch(xyA=(0, 14.5), coordsA=ax.transData,
                                          xyB=(-0.03, 14.5), coordsB=ax.transData,
                                          arrowstyle="-", color="black")
        ax.add_patch(con)
        
        ax.text(0.8, 0.09, "S: Stratosphere\nT: Troposphere", transform=ax.transAxes,
                bbox=dict(facecolor="#F5F5F5", edgecolor="black"), 
                fontsize=14, ha="center", va="center",
                ma="center", linespacing=1.5)
    
    def plot(self):
        figsize = (8, 10)
        dpi = 200
        nrows = 1
        ncols = 1
        axes = []
        
        self.fig = plt.figure(figsize=figsize, dpi=dpi, 
                              facecolor="white")
        gs = self.fig.add_gridspec(nrows=nrows, ncols=ncols, 
                                   left=0.35, right=0.94, bottom=0.05, top=0.97,
                                   wspace=0, hspace=0)
        axes.append(self.fig.add_subplot(gs[0, 0]))
        
        self._draw_axes_0(axes[0])
        
        self._save_fig()
        
        return
    
    def _p_l_o_t_(self):
        # fig = plt.figure(figsize=(12,20), dpi=200)
        # ax = fig.add_subplot(111)
        # bbox = ax.get_position()
        # ax.set_position([bbox.xmin+0.2, bbox.ymin, bbox.width, bbox.height])
        # objdic = ax.boxplot(permut_data, vert=False, whis=(0,100),
        #                     widths=0.88,
        #                     patch_artist=False, positions=y, meanline=True,
        #                     showmeans=True, labels=cols, showcaps=True,
        #                     capprops=dict(lw=1,), whiskerprops=dict(lw=1,),
        #                     medianprops=dict(visible=False),
        #                     meanprops=dict(ls="-", c="#FF0000", lw=0.5));
        # ax.set_title("100 estimators, default parameters, 2007 dataset (100000)\nPermutation, Seed 20220927",
        #              fontfamily="serif", fontsize=15)
        # ax.set_xlim(0,1)
        # ax.set_ylim(-1,48)
        # ax.set_yticks(ticks=y, labels=cols, fontfamily="serif", fontsize=13);
        # fig.savefig("./figure.svg");
        # print("Done")
        pass
    
    pass


class StatisticalDistribution(_PlotBase):
    """
    绘制alt-lct lat-doy alt-lat alt-lon
    观测和模型对比图 以及相关线图
    """
    def __init__(self,
                 backend: str,
                 file_path: str):
        super().__init__(backend, file_path)
        pass
    
    def _load_data(self, file_path):
        file_path = Path(file_path)
        df = pd.read_hdf(file_path)
        
        alt_series = df.loc[:, "alt"]
        select_index = (alt_series >= 80) & (alt_series <= 130)
        select_df = df.loc[select_index, :]
        select_df.reset_index(drop=True, inplace=True)
        
        self.time = select_df.loc[:, "time"]
        self.alt = select_df.loc[:, "alt"].to_numpy()
        self.lct = select_df.loc[:, "lct"].to_numpy()
        self.doy = select_df.loc[:, "time"].dt.day_of_year.to_numpy()
        self.lat = select_df.loc[:, "lat"].to_numpy()
        self.lon = select_df.loc[:, "lon"].to_numpy()
        self.s4max = select_df.loc[:, "s4max"].to_numpy()
        self.test_s4max = select_df.loc[:, "test_s4max"].to_numpy()
        
        return
    
    @staticmethod
    @njit(parallel=True, cache=False)
    def _compute_count(row_coords, col_coords, value, tag=None):
        """
        统计网格内的数值
        网格方向按照直觉性排列
        左上为(row_coords[0,0], col_coords[0,0])
        左下为(row_coords[imax,0], col_coords[imax,0])
        """
        assert len(row_coords.shape) == 1
        assert len(col_coords.shape) == 1
        assert len(value.shape) == 1
        with objmode(row_grid="float64[:,:]", col_grid="float64[:,:]", 
                     delta_row="float64", delta_col="float64"):
            if tag=="alt-lct":
                row_grid, col_grid = np.meshgrid(np.linspace(80,130,100),
                                                 np.linspace(0,24,100),
                                                 indexing="ij")
                pass
            elif tag=="alt-doy":
                row_grid, col_grid = np.meshgrid(np.linspace(80,130,100),
                                                 np.linspace(0,365,100),
                                                 indexing="ij")
                pass
            elif tag=="alt-lat":
                row_grid, col_grid = np.meshgrid(np.linspace(80,130,100),
                                                 np.linspace(-90,90,100),
                                                 indexing="ij")
                pass
            elif tag=="alt-lon":
                row_grid, col_grid = np.meshgrid(np.linspace(80,130,100),
                                                 np.linspace(-180,180,100),
                                                 indexing="ij")
                pass
            else:
                pass
            
            delta_row = np.diff(row_grid, axis=0)[0,0]
            delta_col = np.diff(col_grid, axis=1)[0,0]
            pass
        Z = np.zeros_like(row_grid)
        N = np.zeros_like(row_grid)
        for i in prange(Z.shape[0]):
            for j in prange(Z.shape[1]):
                for data_index in prange(len(value)):
                    rr = row_coords[data_index]
                    cc = col_coords[data_index]
                    vv = value[data_index]
                    
                    rp = row_grid[i, j]
                    cp = col_grid[i, j]
                    if ((rr>=rp-(delta_row/2))&(rr<rp+(delta_row/2))&
                        (cc>=cp-(delta_col/2))&(cc<cp+(delta_col/2))):
                        N[i, j] += 1
                        Z[i, j] += vv
                        pass
                    pass
                pass
            pass
        return (row_grid, col_grid, N, Z)
    
    def plot(self):
        figsize = (12, 5)
        dpi = 200
        nrows = 3
        ncols = 4
        axes = []
        
        self.fig = plt.figure(figsize=figsize, dpi=dpi, 
                              facecolor="white")
        gs = self.fig.add_gridspec(nrows=nrows, ncols=ncols, 
                                   left=0.075, right=0.96, bottom=0.09, top=0.95,
                                   wspace=0.25,)
        for i in range(nrows):
            for j in range(ncols):
                axes.append(self.fig.add_subplot(gs[i, j]))
                pass
            pass
        cmap = "rainbow"
        
        # 0~3为观测的结果
        self._draw_axes_0(axes[0], cmap)
        self._draw_axes_1(axes[1], cmap)
        self._draw_axes_2(axes[2], cmap)
        self._draw_axes_3(axes[3], cmap)
        
        # 4~7为模型结果
        self._draw_axes_4(axes[4], cmap)
        self._draw_axes_5(axes[5], cmap)
        self._draw_axes_6(axes[6], cmap)
        self._draw_axes_7(axes[7], cmap)
        
        # 8~11为线图
        self._draw_axes_8(axes[8])
        self._draw_axes_9(axes[9])
        self._draw_axes_10(axes[10])
        self._draw_axes_11(axes[11])
        
        # 绘制序号
        for index, ax in zip(list("abcdefghijkl"), axes):
            ax.text(0.02, 1.09, f"({index})", transform=ax.transAxes,
                    ha="center", va="center", fontsize=12.5,
                    fontweight="bold")
            pass
        
        # 绘制cbar
        for i in range(4):
            mappable = mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(0.2, 0.6),
                                             cmap=cmap)
            self._add_cbar(axes[i], axes[i+4], mappable)
            pass
        
        # 调整轴的一些细节
        for i, ax in enumerate(axes):
            if (i==1)|(i==2)|(i==3)|(i==5)|(i==6)|(i==7):
                for l in ax.yaxis.get_ticklabels():
                    l.set_visible(False)
                    pass
                pass
            if (i==2)|(i==3):
                ax.xaxis.get_ticklabels()[-1].set_visible(False)
                pass
            if (i!=8)&(i!=9)&(i!=10)&(i!=11):
                ax.xaxis.get_ticklabels()[0].set_visible(False)
            
        self._save_fig()
        return
    
    def _set_alt_tick(self, ax):
        """
        设置高度的刻度细节
        """
        alt = np.linspace(80, 130, num=6, dtype=np.int32)
        altstr = []
        for i, j in enumerate(alt):
            altstr.append(f"{j}")
            pass
        
        lct = mpl.ticker.FixedLocator(alt)
        fmt = mpl.ticker.FixedFormatter(altstr)
        
        ax.yaxis.set_view_interval(80, 130, ignore=True)
        ax.yaxis.set_major_locator(lct)
        ax.yaxis.set_major_formatter(fmt)
        ax.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(n=2))
        ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
        
        ax.tick_params(axis='both', which='both', direction='in', 
                       top=True, right=True)
        pass
    
    def _set_lct_tick(self, ax):
        """
        设定当地时刻度
        """
        lct = np.linspace(0, 24, num=5, dtype=np.int32)
        lctstr = []
        for i, j in enumerate(lct):
            lctstr.append(f"{j}")
            pass
        lct = mpl.ticker.FixedLocator(lct)
        fmt = mpl.ticker.FixedFormatter(lctstr)
        
        ax.xaxis.set_view_interval(0, 24, ignore=True)
        ax.xaxis.set_major_locator(lct)
        ax.xaxis.set_major_formatter(fmt)
        ax.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(n=2))
        ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())

        ax.tick_params(axis='both', which='both', direction='in', 
                          top=True, right=True)
        pass
    
    def _set_doy_tick(self, ax):
        """
        设定DOY刻度
        """
        doy = np.arange(0, 365, 80, dtype=np.int32)
        doystr = []
        for i, j in enumerate(doy):
            doystr.append(f"{j}")
            pass
        lct = mpl.ticker.FixedLocator(doy)
        fmt = mpl.ticker.FixedFormatter(doystr)

        ax.xaxis.set_view_interval(0, 365, ignore=True)
        ax.xaxis.set_major_locator(lct)
        ax.xaxis.set_major_formatter(fmt)
        ax.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(n=2))
        ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())

        ax.tick_params(axis='both', which='both', direction='in', 
                          top=True, right=True)
        pass
    
    def _set_lat_tick(self, ax):
        """
        设置纬度刻度
        """
        lat = np.linspace(-90, 90, num=13, dtype=np.int32)
        latstr = []
        for i, j in enumerate(lat):
            if i%2==0:
                jj = str(j) + "$^{\circ}$"
                latstr.append(f"{jj}")
                pass
            else:
                latstr.append("")
                pass
            pass
        lct = mpl.ticker.FixedLocator(lat)
        fmt = mpl.ticker.FixedFormatter(latstr)

        ax.xaxis.set_view_interval(-90, 90, ignore=True)
        ax.xaxis.set_major_locator(lct)
        ax.xaxis.set_major_formatter(fmt)
        ax.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(n=2))
        ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())

        ax.tick_params(axis='both', which='both', direction='in', 
                          top=True, right=True)
        pass
    
    def _set_lon_tick(self, ax):
        """
        设置经度的刻度细节
        """
        lon = np.linspace(-180, 180, num=19, dtype=np.int32)
        lonstr = []
        for i, j in enumerate(lon):
            if i%3==0:
                jj = str(j) + "$^{\circ}$"
                lonstr.append(f"{jj}")
                pass
            else:
                lonstr.append("")
                pass
            pass
        lct = mpl.ticker.FixedLocator(lon)
        fmt = mpl.ticker.FixedFormatter(lonstr)
        
        ax.xaxis.set_view_interval(-180, 180, ignore=True)
        ax.xaxis.set_major_locator(lct)
        ax.xaxis.set_major_formatter(fmt)
        ax.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(n=2))
        ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
        ax.tick_params(axis='both', which='both', direction='in', 
                       top=True, right=True)
        pass
    
    def _set_s4max_tick(self, ax):
        """
        设定S4max强度刻度
        """
        value = np.arange(0, 1.25, 0.25)
        vstr = [f"{v}" for v in value]
        
        lct = mpl.ticker.FixedLocator(value)
        fmt = mpl.ticker.FixedFormatter(vstr)
        
        ax.yaxis.set_view_interval(0, 1, ignore=True)
        ax.yaxis.set_major_locator(lct)
        ax.yaxis.set_major_formatter(fmt)
        ax.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(n=2))
        ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
        
        ax.tick_params(axis='both', which='both', direction='in', 
                       top=True, right=True)
        pass
    
    def _add_cbar(self, ax1, ax2, mappable):
        """
        在ax1和ax2之间添加colorbar
        mappable为对应的映射参数
        """
        box1 = mpl.transforms.Bbox.from_extents(ax2.get_position().xmax+0.008,
                                                ax2.get_position().ymin,
                                                ax2.get_position().xmax+0.018,
                                                ax1.get_position().ymax)
        cax1 = self.fig.add_axes([.01,.01,.01,.01], position=box1)
        cbar1 = self.fig.colorbar(mappable, cax=cax1)
        cbar1.ax.minorticks_on()
        cbar1.ax.set_yticks([0.2, 0.3, 0.4, 0.5, 0.6])
        cbar1.ax.tick_params(axis="both", which="both", direction="in",
                             labelsize=8)
        cax1.text(0.3, 1.04, "Intensity",
                  va="center", ha="center", transform=cax1.transAxes,
                  fontsize=9)
        # cbar1.ax.yaxis.set_view_interval(vmin, vmax, ignore=True)
        
    def _draw_axes_0(self, ax, cmap):
        # alt-lct
        grid_R, grid_C, grid_count, grid_value = self._compute_count(self.alt, self.lct, 
                                                                     self.s4max,
                                                                     tag="alt-lct")
        grid_count[grid_count==0] = 1
        contourset = ax.pcolormesh(grid_C, grid_R, grid_value/grid_count,
                                   cmap=cmap, norm=mpl.colors.Normalize(0.2,0.6))
        
        self._set_alt_tick(ax)
        self._set_lct_tick(ax)
        ax.text(-0.32, 0.5, "COSMIC Data",
                transform=ax.transAxes, ha="center", va="center",
                rotation="vertical", fontsize=12, fontweight="bold")
        ax.set_ylabel("Altitude (Km)", loc="center", fontsize=10)
        pass
    
    def _draw_axes_1(self, ax, cmap):
        # alt-doy
        grid_R, grid_C, grid_count, grid_value = self._compute_count(self.alt, self.doy, 
                                                                     self.s4max,
                                                                     tag="alt-doy")
        grid_count[grid_count==0] = 1
        contourset = ax.pcolormesh(grid_C, grid_R, grid_value/grid_count,
                                   cmap=cmap, norm=mpl.colors.Normalize(0.2,0.6))
        self._set_alt_tick(ax)
        self._set_doy_tick(ax)
        pass
    
    def _draw_axes_2(self, ax, cmap):
        # alt-lat
        grid_R, grid_C, grid_count, grid_value = self._compute_count(self.alt, self.lat, 
                                                                     self.s4max,
                                                                     tag="alt-lat")
        grid_count[grid_count==0] = 1
        contourset = ax.pcolormesh(grid_C, grid_R, grid_value/grid_count,
                                   cmap=cmap, norm=mpl.colors.Normalize(0.2,0.6))
        self._set_alt_tick(ax)
        self._set_lat_tick(ax)
        pass
    
    def _draw_axes_3(self, ax, cmap):
        # alt-lon
        grid_R, grid_C, grid_count, grid_value = self._compute_count(self.alt, self.lon, 
                                                                     self.s4max,
                                                                     tag="alt-lon")
        grid_count[grid_count==0] = 1
        contourset = ax.pcolormesh(grid_C, grid_R, grid_value/grid_count,
                                   cmap=cmap, norm=mpl.colors.Normalize(0.2,0.6))
        self._set_alt_tick(ax)
        self._set_lon_tick(ax)
        pass
    
    def _draw_axes_4(self, ax, cmap):
        # alt-lct
        grid_R, grid_C, grid_count, grid_value = self._compute_count(self.alt, self.lct, 
                                                                     self.test_s4max,
                                                                     tag="alt-lct")
        grid_count[grid_count==0] = 1
        contourset = ax.pcolormesh(grid_C, grid_R, grid_value/grid_count,
                                   cmap=cmap, norm=mpl.colors.Normalize(0.2,0.6))
        self._set_alt_tick(ax)
        self._set_lct_tick(ax)
        ax.text(-0.32, 0.5, "Model",
                transform=ax.transAxes, ha="center", va="center",
                rotation="vertical", fontsize=12, fontweight="bold")
        ax.set_ylabel("Altitude (Km)", loc="center", fontsize=10)
        pass
    
    def _draw_axes_5(self, ax, cmap):
        # alt-doy
        grid_R, grid_C, grid_count, grid_value = self._compute_count(self.alt, self.doy, 
                                                                     self.test_s4max,
                                                                     tag="alt-doy")
        grid_count[grid_count==0] = 1
        contourset = ax.pcolormesh(grid_C, grid_R, grid_value/grid_count,
                                   cmap=cmap, norm=mpl.colors.Normalize(0.2,0.6))
        self._set_alt_tick(ax)
        self._set_doy_tick(ax)
        pass
    
    def _draw_axes_6(self, ax, cmap):
        # alt-lat
        grid_R, grid_C, grid_count, grid_value = self._compute_count(self.alt, self.lat, 
                                                                     self.test_s4max,
                                                                     tag="alt-lat")
        grid_count[grid_count==0] = 1
        contourset = ax.pcolormesh(grid_C, grid_R, grid_value/grid_count,
                                   cmap=cmap, norm=mpl.colors.Normalize(0.2,0.6))
        self._set_lat_tick(ax)
        self._set_alt_tick(ax)
        pass
    
    def _draw_axes_7(self, ax, cmap):
        # alt-lon
        grid_R, grid_C, grid_count, grid_value = self._compute_count(self.alt, self.lon, 
                                                                     self.test_s4max,
                                                                     tag="alt-lon")
        grid_count[grid_count==0] = 1
        contourset = ax.pcolormesh(grid_C, grid_R, grid_value/grid_count,
                                   cmap=cmap, norm=mpl.colors.Normalize(0.2,0.6))
        self._set_lon_tick(ax)
        self._set_alt_tick(ax)
        pass
    
    def _draw_axes_8(self, ax):
        # alt-lct
        grid_R1, grid_C1, grid_count1, grid_value1 = self._compute_count(self.alt, self.lct, 
                                                                     self.s4max,
                                                                     tag="alt-lct")
        grid_R2, grid_C2, grid_count2, grid_value2 = self._compute_count(self.alt, self.lct, 
                                                                     self.test_s4max,
                                                                     tag="alt-lct")
        mean_s4max = grid_value1.sum(axis=0)/grid_count1.sum(axis=0)
        mean_test_s4max = grid_value2.sum(axis=0)/grid_count2.sum(axis=0)
        line1, = ax.plot(grid_C1[0, :], mean_s4max, color="#c1c1c1")
        line2, = ax.plot(grid_C1[0, :], mean_test_s4max, color="#d32327")
        ax.legend(handles=[line1, line2], labels=["Data", "Prediction"],
                  loc="upper right", fontsize=8,
                  edgecolor="black")
        
        self._set_s4max_tick(ax)
        self._set_lct_tick(ax)
        ax.set_ylabel("S4max intensity", loc="center", fontsize=12)
        ax.set_xlabel("Local Time", loc="center", fontsize=12)
        ax.grid(ls="--")
        pass
    def _draw_axes_9(self, ax):
        # alt-doy
        grid_R1, grid_C1, grid_count1, grid_value1 = self._compute_count(self.alt, self.doy, 
                                                                     self.s4max,
                                                                     tag="alt-doy")
        grid_R2, grid_C2, grid_count2, grid_value2 = self._compute_count(self.alt, self.doy, 
                                                                     self.test_s4max,
                                                                     tag="alt-doy")
        mean_s4max = grid_value1.sum(axis=0)/grid_count1.sum(axis=0)
        mean_test_s4max = grid_value2.sum(axis=0)/grid_count2.sum(axis=0)
        line1, = ax.plot(grid_C1[0, :], mean_s4max, color="#c1c1c1")
        line2, = ax.plot(grid_C1[0, :], mean_test_s4max, color="#d32327")
        ax.legend(handles=[line1, line2], labels=["Data", "Prediction"],
                  loc="upper right", fontsize=8,
                  edgecolor="black")
        
        self._set_s4max_tick(ax)
        self._set_doy_tick(ax)
        ax.set_xlabel("Day of Year", loc="center", fontsize=12)
        ax.grid(ls="--")
        pass
    
    def _draw_axes_10(self, ax):
        # alt-lat
        grid_R1, grid_C1, grid_count1, grid_value1 = self._compute_count(self.alt, self.lat, 
                                                                     self.s4max,
                                                                     tag="alt-lat")
        grid_R2, grid_C2, grid_count2, grid_value2 = self._compute_count(self.alt, self.lat, 
                                                                     self.test_s4max,
                                                                     tag="alt-lat")
        mean_s4max = grid_value1.sum(axis=0)/grid_count1.sum(axis=0)
        mean_test_s4max = grid_value2.sum(axis=0)/grid_count2.sum(axis=0)
        line1, = ax.plot(grid_C1[0, :], mean_s4max, color="#c1c1c1")
        line2, = ax.plot(grid_C1[0, :], mean_test_s4max, color="#d32327")
        ax.legend(handles=[line1, line2], labels=["Data", "Prediction"],
                  loc="upper right", fontsize=8,
                  edgecolor="black")
        
        self._set_s4max_tick(ax)
        self._set_lat_tick(ax)
        
        ax.set_xlabel("Geographic Latitude", loc="center", fontsize=12)
        ax.grid(ls="--")
        pass
    def _draw_axes_11(self, ax):
        # alt-lon
        grid_R1, grid_C1, grid_count1, grid_value1 = self._compute_count(self.alt, self.lon, 
                                                                     self.s4max,
                                                                     tag="alt-lon")
        grid_R2, grid_C2, grid_count2, grid_value2 = self._compute_count(self.alt, self.lon, 
                                                                     self.test_s4max,
                                                                     tag="alt-lon")
        mean_s4max = grid_value1.sum(axis=0)/grid_count1.sum(axis=0)
        mean_test_s4max = grid_value2.sum(axis=0)/grid_count2.sum(axis=0)
        line1, = ax.plot(grid_C1[0, :], mean_s4max, color="#c1c1c1")
        line2, = ax.plot(grid_C1[0, :], mean_test_s4max, color="#d32327")
        ax.legend(handles=[line1, line2], labels=["Data", "Prediction"],
                  loc="upper right", fontsize=8,
                  edgecolor="black")
        
        self._set_s4max_tick(ax)
        self._set_lon_tick(ax)
        ax.set_xlabel("Geographic Longitude", loc="center", fontsize=12)
        ax.grid(ls="--")
        pass
    
    pass


class SeasonDistribution(_PlotBase):
    """
    模型的季节分布和磁纬分布
    """
    def __init__(self,
                 backend: str,
                 file_path: str):
        super().__init__(backend, file_path)
        pass
    
    def _load_data(self, file_path):
        file_path = Path(file_path)
        df = pd.read_hdf(file_path)
        
        alt_series = df.loc[:, "alt"]
        select_index = (alt_series >= 80) & (alt_series <= 130)
        select_df = df.loc[select_index, :]
        #####
        select_df.reset_index(drop=True, inplace=True)
        #####
        self.df = select_df
        return 
    
    def _select_season(self, season=None):
        """
        选择季节
        不同季节给出不同的数据
        """
        # spring: MAM(3,4,5)
        # summer: JJA(6,7,8)
        # autumn: SON(9,10,11)
        # winter: DJF(12,1,2)
        use_month_series = self.df.loc[:, "time"].dt.month
        if season=="spring":
            use_time_index = (use_month_series>=3)&(use_month_series<=5)
            
            # length = len(use_time_index)
            # randix = np.random.randint(0, length, size=int(0.1*length))
            # use_time_index = use_time_index[randix]
            pass
        elif season=="summer":
            use_time_index = (use_month_series>=6)&(use_month_series<=8)
            pass
        elif season=="autumn":
            use_time_index = (use_month_series>=9)&(use_month_series<=11)
            pass
        elif season=="winter":
            use_time_index = ((use_month_series>=1)&(use_month_series<=2))|(use_month_series==12)
            pass
        elif season=="global":
            pass
        else:
            raise OSError("Undefine season")
            pass
        
        try:
            lat = self.df.loc[use_time_index, "lat"].to_numpy()
            lon = self.df.loc[use_time_index, "lon"].to_numpy()
            s4max = self.df.loc[use_time_index, "s4max"].to_numpy()
            test_s4max = self.df.loc[use_time_index, "test_s4max"].to_numpy()
            pass
        except:
            lat = self.df.loc[:, "lat"].to_numpy()
            lon = self.df.loc[:, "lon"].to_numpy()
            s4max = self.df.loc[:, "s4max"].to_numpy()
            test_s4max = self.df.loc[:, "test_s4max"].to_numpy()
            pass
        
        grid_R1, grid_C1, grid_count1, grid_value1 = self._compute_count(lat, lon, s4max)
        grid_R2, grid_C2, grid_count2, grid_value2 = self._compute_count(lat, lon, test_s4max)
        
        # 计算地理纬度和dip纬度平均
        (grid_glat1, N1, Z1) = self._compute_glat_mean(lat, lon, s4max, tag="glat")
        (grid_glat2, N2, Z2) = self._compute_glat_mean(lat, lon, test_s4max, tag="glat")
        
        (grid_dip_lat1, N3, Z3) = self._compute_glat_mean(lat, lon, s4max, tag="dip_lat")
        (grid_dip_lat2, N4, Z4) = self._compute_glat_mean(lat, lon, test_s4max, tag="dip_lat")
        
        glat_mean_s4max = Z1/N1
        glat_mean_test_s4max = Z2/N2
        
        dip_lat_mean_s4max = Z3/N3
        dip_lat_mean_test_s4max = Z4/N4
        
        # 防止出现0为除数
        grid_count1[grid_count1==0] = 1
        grid_count2[grid_count2==0] = 1
        s4max_density = grid_value1/grid_count1
        test_s4max_density = grid_value2/grid_count2
        return (grid_R1, grid_C1, s4max_density, test_s4max_density, 
                grid_glat1, glat_mean_s4max, glat_mean_test_s4max,
                grid_dip_lat1, dip_lat_mean_s4max, dip_lat_mean_test_s4max,)
    
    @staticmethod
    @njit(parallel=True, cache=False)
    def _compute_count(row_coords, col_coords, value, tag="lat-lon"):
        """
        统计网格内的数值
        网格方向按照直觉性排列
        左上为(row_coords[0,0], col_coords[0,0])
        左下为(row_coords[imax,0], col_coords[imax,0])
        """
        assert len(row_coords.shape) == 1
        assert len(col_coords.shape) == 1
        assert len(value.shape) == 1
        with objmode(row_grid="float64[:,:]", col_grid="float64[:,:]", 
                     delta_row="float64", delta_col="float64"):
            if tag=="lat-lon":
                # row_grid, col_grid = np.meshgrid(np.linspace(-90,90,90),
                #                                  np.linspace(-180,180,180),
                #                                  indexing="ij")
                row_grid, col_grid = np.meshgrid(np.arange(-90.,90.1,2.5),
                                                 np.arange(-180.,180.1,2.5),
                                                 indexing="ij")
                pass
            
            delta_row = np.diff(row_grid, axis=0)[0,0]
            delta_col = np.diff(col_grid, axis=1)[0,0]
            pass
        Z = np.zeros_like(row_grid)
        N = np.zeros_like(row_grid)
        for i in prange(Z.shape[0]):
            for j in prange(Z.shape[1]):
                for data_index in prange(len(value)):
                    rr = row_coords[data_index]
                    cc = col_coords[data_index]
                    vv = value[data_index]
                    
                    rp = row_grid[i, j]
                    cp = col_grid[i, j]
                    if ((rr>=rp-(delta_row/2))&(rr<rp+(delta_row/2))&
                        (cc>=cp-(delta_col/2))&(cc<cp+(delta_col/2))):
                        N[i, j] += 1
                        Z[i, j] += vv
                        pass
                    pass
                pass
            pass
        return (row_grid, col_grid, N, Z)
    
    @staticmethod
    @njit(parallel=True)
    def _compute_glat_mean(glat_coords, glon_coords, value, tag="glat"):
        """
        统计给定纬度网格内的value平均值
        """
        assert len(glat_coords.shape)==1
        assert len(glon_coords.shape)==1
        assert len(value.shape)==1
        assert glat_coords.shape==value.shape
        if tag=="glat":
            # grid_glat = np.arange(-89.5,90,1)
            # grid_glat = np.arange(-90+(1.5/2),90,1.5)
            grid_glat = np.arange(-90+.75, 90., 1.5)
            # grid_glat = np.linspace(-90,90,91)
            delta = grid_glat[1] - grid_glat[0]
            N = np.zeros_like(grid_glat)
            Z = np.zeros_like(grid_glat)
            for i in prange(Z.shape[0]):
                for di in prange(len(value)):
                    glati = glat_coords[di]
                    valuei = value[di]
                    grid_glati = grid_glat[i]
                    
                    if (glati>=grid_glati-(delta/2))&(glati<grid_glati+(delta/2)):
                        N[i] += 1
                        Z[i] += valuei
                        pass
                    pass
                pass
            return (grid_glat, N, Z)
        elif tag=="dip_lat":
            # 把地理经纬度转化为dip纬度
            year = 2014
            alt = 100
            with objmode(g="float64[:,:]", h="float64[:,:]"):
                g, h = igrf.ReadCoeffs().get_coeffs(year)
                pass
            dip_lat_coords = np.zeros_like(glat_coords)
            for i in prange(len(value)):
                dip, gd, gi, bh, bx, by, bz, bf = igrf.igrf_value(g, h, 
                                                                  glat_coords[i], glon_coords[i], 
                                                                  alt, year)
                # dip纬度/磁倾角
                dip_lat_coords[i] = dip
                pass
            # dip 纬度网格
            # grid_dip_lat = np.arange(-89.5,90,1)
            grid_dip_lat = np.arange(-90+.75, 90., 1.5)
            # grid_dip_lat = np.linspace(-90,90,91)
            delta = grid_dip_lat[1] - grid_dip_lat[0]
            N = np.zeros_like(grid_dip_lat)
            Z = np.zeros_like(grid_dip_lat)
            for i in prange(Z.shape[0]):
                for di in prange(len(value)):
                    dip_lati = dip_lat_coords[di]
                    valuei = value[di]
                    grid_dip_lati = grid_dip_lat[i]
                    
                    if (dip_lati>grid_dip_lati-(delta/2))&(dip_lati<=grid_dip_lati+(delta/2)):
                        N[i] += 1
                        Z[i] += valuei
                        pass
                    pass
                pass
            return (grid_dip_lat, N, Z)
        
        return (None,)*3
    
    def plot(self):
        figsize = (12, 8)
        dpi = 200
        nrows = 4
        ncols = 4
        axes = []
        
        self.fig = plt.figure(figsize=figsize, dpi=dpi, 
                              facecolor="white")
        gs = self.fig.add_gridspec(nrows=nrows, ncols=ncols, 
                                   left=0.05, right=0.98, bottom=0.06, top=0.95,
                                   wspace=0.15, hspace=0.1)
        for i in range(nrows):
            for j in range(ncols):
                if (j==0)|(j==1):
                    axes.append(self.fig.add_subplot(gs[i, j], projection=cartopy.crs.Robinson()))
                    pass
                else:
                    axes.append(self.fig.add_subplot(gs[i, j]))
                pass
            pass
        season = ["spring", "summer", "autumn", "winter"]
        for i in range(nrows):
            index = slice(4*i, 4*(i+1))
            self._draw_axes_season(*axes[index], season[i], i)
            pass
        
        # 标记序号
        for i, ax in zip("abcdefghijklmnop", axes):
            ax.text(0.02, 1.09, f"({i})", transform=ax.transAxes,
                    ha="center", va="center", fontsize=12.5,
                    fontweight="bold")
            pass
        
        # 调整轴的一些细节
        # 添加说明文字
        for i, ax in enumerate(axes):
            if i==0:
                ax.text(0.5, 1.2, "COSMIC Data",
                        transform=ax.transAxes, ha="center", va="center",
                        rotation="horizontal", fontsize=13, fontweight="bold")
                
                ax.text(-0.1, 0.5, "Spring",
                        transform=ax.transAxes, ha="center", va="center",
                        rotation="vertical", fontsize=12, fontweight="bold")
                pass
            
            if i==1:
                ax.text(0.5, 1.2, "Model",
                        transform=ax.transAxes, ha="center", va="center",
                        rotation="horizontal", fontsize=13, fontweight="bold")
                pass
            
            if i==4:
                ax.text(-0.1, 0.5, "Summer",
                        transform=ax.transAxes, ha="center", va="center",
                        rotation="vertical", fontsize=12, fontweight="bold")
                pass
            
            if i==8:
                ax.text(-0.1, 0.5, "Autumn",
                        transform=ax.transAxes, ha="center", va="center",
                        rotation="vertical", fontsize=12, fontweight="bold")
                pass
            
            if i==12:
                ax.text(-0.1, 0.5, "Winter",
                        transform=ax.transAxes, ha="center", va="center",
                        rotation="vertical", fontsize=12, fontweight="bold")
                pass
            
            if (i==2)|(i==3)|(i==6)|(i==7)|(i==10)|(i==11):
                ax.xaxis.get_ticklabels()[0].set_visible(False)
                pass
            pass
        
        
        for i in range(nrows * ncols):
            # axes[i].text(0.5, 0.5, f"{i}", transform=axes[i].transAxes,
            #              fontsize=25, fontweight="bold")
            pass
        
        # 添加colorbar
        mappable = mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(0., 0.8),
                                         cmap="rainbow")
        self._hang_cbar(axes[12], axes[13], mappable)
        self._save_fig()
        return
    
    def _hang_cbar(self, ax1, ax2, mappable):
        """
        给定两个axes添加colorbar
        mappable为对应的映射参数
        """
        box1 = mpl.transforms.Bbox.from_bounds(ax1.get_position(original=True).xmin+0.06,
                                               ax1.get_position(original=True).ymin-0.02,
                                               0.35, 0.02)
        cax1 = self.fig.add_axes([.01,.01,.01,.01], position=box1)
        cbar1 = self.fig.colorbar(mappable, cax=cax1, extend="both", 
                                  orientation="horizontal")
        cbar1.ax.minorticks_on()
        cbar1.ax.set_xticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
        cbar1.ax.tick_params(axis="both", which="both", direction="in",
                             labelsize=9)
        cax1.text(-0.15, 0.5, f"S4max\nIntensity",
                  va="center", ha="center", transform=cax1.transAxes,
                  fontsize=10)
        pass
    
    def _zoom_axes(self, ax, k):
        """
        拉伸axes的大小
        同时确保是等比放缩
        原图大小: xmin ymin w h
        放缩后: xmin' = xmin - (k - 1) * w / 2
                ymin' = ymin - (k - 1) * h / 2
                w' = k * w
                h' = k * h
        k大于1放大 小于1缩小 中心位置不变
        """
        bbox1 = ax.get_position(original=True)
        x1, y1, w1, h1 = bbox1.xmin, bbox1.ymin, bbox1.width, bbox1.height
        x2 = x1 - (k - 1) * w1 / 2
        y2 = y1 - (k - 1) * h1 / 2
        w2 = k * w1
        h2 = k * h1
        bbox2 = mpl.transforms.Bbox.from_bounds(x2, y2, w2, h2)
        ax.set_position(bbox2)
        return
    
    def _set_lat_tick(self, ax):
        """
        设置纬度刻度
        """
        lat = np.linspace(-90, 90, num=13, dtype=np.int32)
        latstr = []
        for i, j in enumerate(lat):
            if i%2==0:
                jj = str(j) + "$^{\circ}$"
                latstr.append(f"{jj}")
                pass
            else:
                latstr.append("")
                pass
            pass
        lct = mpl.ticker.FixedLocator(lat)
        fmt = mpl.ticker.FixedFormatter(latstr)

        ax.xaxis.set_view_interval(-90, 90, ignore=True)
        ax.xaxis.set_major_locator(lct)
        ax.xaxis.set_major_formatter(fmt)
        ax.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(n=2))
        ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())

        ax.tick_params(axis="both", which="both", direction="in", 
                       top=True, right=True)
        pass
    
    def _set_s4max_tick(self, ax):
        """
        设定S4max强度刻度
        """
        value = np.arange(0, 1.25, 0.25)
        vstr = [f"{v}" for v in value]
        
        lct = mpl.ticker.FixedLocator(value)
        fmt = mpl.ticker.FixedFormatter(vstr)
        
        ax.yaxis.set_view_interval(0, 1, ignore=True)
        ax.yaxis.set_major_locator(lct)
        ax.yaxis.set_major_formatter(fmt)
        ax.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(n=2))
        ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
        
        ax.tick_params(axis="both", which="both", direction="in", 
                       top=True, right=True)
        pass
    
    def _draw_axes_season(self, ax1, ax2, ax3, ax4, season, i):
        (grid_R, grid_C, 
         s4max_density, test_s4max_density, 
         grid_glat, glat_mean_s4max, glat_mean_test_s4max,
         grid_dip_lat, dip_lat_mean_s4max, 
         dip_lat_mean_test_s4max) = self._select_season(season=season)
        
        # 绘制ax1
        # contourset1 = ax1.contourf(grid_C, grid_R, s4max_density, 
        #                            levels=50, cmap="rainbow", 
        #                            norm=mpl.colors.Normalize(0.,0.8),
        #                            transform=cartopy.crs.PlateCarree(),
        #                            transform_first=True)
        ax1.pcolormesh(grid_C, grid_R, s4max_density, cmap="rainbow", 
                       norm=mpl.colors.Normalize(0,0.8),
                       transform=cartopy.crs.PlateCarree(),)
        self._zoom_axes(ax1, 1.122)
        ax1.add_feature(cartopy.feature.COASTLINE)
        ax1.set_global()
        
        # 绘制ax2
        # contourset2 = ax2.contourf(grid_C, grid_R, test_s4max_density, 
        #                            levels=50, cmap="rainbow", 
        #                            norm=mpl.colors.Normalize(0.,0.8),
        #                            transform=cartopy.crs.PlateCarree(),
        #                            transform_first=True)
        ax2.pcolormesh(grid_C, grid_R, test_s4max_density, cmap="rainbow", 
                       norm=mpl.colors.Normalize(0,0.8),
                       transform=cartopy.crs.PlateCarree(),)
                       # transform_first=True)
        self._zoom_axes(ax2, 1.122)
        ax2.add_feature(cartopy.feature.COASTLINE)
        ax2.set_global()
        
        
        
        # 绘制ax3
        # 针对不同季节
        # 调整细节
        if i==3:
            ax3.set_xlabel("Geographic Latitude", fontsize=12)
            ax4.set_xlabel("Geomagnetic Latitude", fontsize=12)
            pass
        
        # 不同季节绘制不同的fill region
        thres = 7
        Hc = "#ffece3"
        Lc = "#e7f4fd"
        Bc = "#bdd2e6"
        if season=="spring":
            ax3.axvspan(xmin=30-thres, xmax=30+thres, color=Hc)
            ax3.axvspan(xmin=60-thres, xmax=60+thres, color=Lc)
            
            ax4.axvspan(xmin=-7.5, xmax=7.5, color=Bc)
            ax4.axvspan(xmin=30-thres, xmax=30+thres, color=Hc)
            ax4.axvspan(xmin=60-thres, xmax=60+thres, color=Lc)
            pass
        
        elif season=="summer":
            ax3.axvspan(xmin=20, xmax=40, color=Hc)
            
            ax4.axvspan(xmin=-7.5, xmax=7.5, color=Bc)
            ax4.axvspan(xmin=30-thres, xmax=30+thres, color=Hc)
            ax4.axvspan(xmin=60-thres, xmax=60+thres, color=Lc)
            pass
        
        elif season=="autumn":
            ax3.axvspan(xmin=60-thres, xmax=60+thres, color=Lc)
            
            ax4.axvspan(xmin=-7.5, xmax=7.5, color=Bc)
            ax4.axvspan(xmin=-30-thres, xmax=-30+thres, color=Hc)
            ax4.axvspan(xmin=60-thres, xmax=60+thres, color=Lc)
            pass
        
        else:
            ax3.axvspan(xmin=-45-thres, xmax=-45+thres, color=Hc)
            ax3.axvspan(xmin=60-thres, xmax=60+thres, color=Lc)
            
            ax4.axvspan(xmin=-7.5, xmax=7.5, color=Bc)
            ax4.axvspan(xmin=-30-thres, xmax=-30+thres, color=Hc)
            
            pass
        #f1bc23
        #2b6a99
        line1, = ax3.plot(grid_glat, glat_mean_s4max, color="#c1c1c1")
        line2, = ax3.plot(grid_glat, glat_mean_test_s4max, color="#d32327")
        self._zoom_axes(ax3, 0.93)
        self._set_s4max_tick(ax3)
        self._set_lat_tick(ax3)
        ax3.grid(ls="--")
        ax3.legend(handles=[line1, line2], labels=["Data", "Prediction"],
                  loc="upper left", fontsize=8,
                  edgecolor="black")
        
        
        # 绘制ax4
        line3, = ax4.plot(grid_dip_lat, dip_lat_mean_s4max, color="#c1c1c1")
        line4, = ax4.plot(grid_dip_lat, dip_lat_mean_test_s4max, color="#d32327")
        self._zoom_axes(ax4, 0.93)
        self._set_s4max_tick(ax4)
        self._set_lat_tick(ax4)
        ax4.grid(ls="--")
        ax4.legend(handles=[line3, line4], labels=["Data", "Prediction"],
                  loc="upper left", fontsize=8,
                  edgecolor="black")
        
        return
    pass


class GlobalDistribution(_PlotBase):
    """
    绘制全球的分布对比
    带有磁场信息
    """
    def __init__(self, backend: str,
                 file_path: str):
        super().__init__(backend, file_path)
        self._global_data()
        file = Path("./vik.txt")
        cdata = np.loadtxt(file)
        self.cmap = mpl.colors.LinearSegmentedColormap.from_list("cc", cdata)
        pass
    
    def _load_data(self, file_path):
        file_path = Path(file_path)
        df = pd.read_hdf(file_path)
        
        alt_series = df.loc[:, "alt"]
        select_index = (alt_series >= 80) & (alt_series <= 130)
        select_df = df.loc[select_index, :]
        select_df.reset_index(drop=True, inplace=True)
        self.df = select_df
        return
    
    def _global_data(self,):
        lat = self.df.loc[:, "lat"].to_numpy()
        lon = self.df.loc[:, "lon"].to_numpy()
        s4max = self.df.loc[:, "s4max"].to_numpy()
        test_s4max = self.df.loc[:, "test_s4max"].to_numpy()
        
        grid_R1, grid_C1, grid_count1, grid_value1 = SeasonDistribution._compute_count(lat, lon, s4max)
        grid_R2, grid_C2, grid_count2, grid_value2 = SeasonDistribution._compute_count(lat, lon, test_s4max)
        
        # 计算地理纬度和dip纬度平均
        (grid_glat1, N1, Z1) = SeasonDistribution._compute_glat_mean(lat, lon, s4max, tag="glat")
        (grid_glat2, N2, Z2) = SeasonDistribution._compute_glat_mean(lat, lon, test_s4max, tag="glat")
        
        (grid_dip_lat1, N3, Z3) = SeasonDistribution._compute_glat_mean(lat, lon, s4max, tag="dip_lat")
        (grid_dip_lat2, N4, Z4) = SeasonDistribution._compute_glat_mean(lat, lon, test_s4max, tag="dip_lat")
        
        glat_mean_s4max = Z1/N1
        glat_mean_test_s4max = Z2/N2
        
        dip_lat_mean_s4max = Z3/N3
        dip_lat_mean_test_s4max = Z4/N4
        
        # 防止出现0为除数
        grid_count1[grid_count1==0] = 1
        grid_count2[grid_count2==0] = 1
        s4max_density = grid_value1/grid_count1
        test_s4max_density = grid_value2/grid_count2
        
        self.grid_R = grid_R1
        self.grid_C = grid_C1
        self.s4max_density = s4max_density
        self.test_s4max_density = test_s4max_density
        self.grid_glat = grid_glat1
        self.glat_mean_s4max = glat_mean_s4max
        self.glat_mean_test_s4max = glat_mean_test_s4max
        self.grid_dip_lat = grid_dip_lat1
        self.dip_lat_mean_s4max = dip_lat_mean_s4max
        self.dip_lat_mean_test_s4max = dip_lat_mean_test_s4max
        
        (self.grid_global_lat, self.grid_global_lon,
         self.global_dip_lat, self.global_decl) = self._compute_global_magnetic()
        
        return
    
    @staticmethod
    @njit(parallel=True)
    def _compute_global_magnetic():
        """
        计算全球的磁场dip纬度和磁偏角
        """
        with objmode(glat="float64[:,:]", glon="float64[:,:]"):
            glat, glon = np.meshgrid(np.arange(-90,90,0.25)[1::2], 
                                     np.arange(-180,180,0.25)[1::2],
                                     indexing="ij")
            pass
        dip_lat = np.zeros_like(glat)
        decl = np.zeros_like(glat)
        alt = 100
        year = 2014
        
        with objmode(g="float64[:,:]", h="float64[:,:]"):
            g, h = igrf.ReadCoeffs().get_coeffs(year)
            pass
        
        for i in prange(glat.shape[0]):
            for j in prange(glat.shape[1]):
                lat = glat[i, j]
                lon = glon[i, j]
                dip, gd, gi, bh, bx, by, bz, bf = igrf.igrf_value(g, h, lat, lon, alt, year)
                dip_lat[i, j] = dip
                decl[i, j] = gd
                pass
            pass
        return (glat, glon, dip_lat, decl)
    
    def plot(self,):
        figsize = (12, 11.5)
        dpi = 200
        nrows = 2
        ncols = 1
        axes = []
        
        self.fig = plt.figure(figsize=figsize, dpi=dpi, 
                              facecolor="white")
        gs = self.fig.add_gridspec(nrows=nrows, ncols=ncols, 
                                   left=0.1, right=0.89, bottom=0.09, top=0.94,)
                                   # wspace=0.3, hspace=0.2)
        
        axes.append(self.fig.add_subplot(gs[0, 0], projection=cartopy.crs.PlateCarree()))
        axes.append(self.fig.add_subplot(gs[1, 0], projection=cartopy.crs.PlateCarree()))
        
        self._draw_axes_0(axes[0])
        self._draw_axes_1(axes[1])
        
        # 添加colorbar
        # 使用on the fly的方法
        mappable = mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(0.15, 0.75),
                                         cmap=self.cmap)
        self._add_cbar(axes[0], axes[1], mappable)
        
        self._save_fig()
        return 
    
    def _zoom_axes(self, ax, k):
        """
        拉伸axes的大小
        同时确保是等比放缩
        原图大小: xmin ymin w h
        放缩后: xmin' = xmin - (k - 1) * w / 2
                ymin' = ymin - (k - 1) * h / 2
                w' = k * w
                h' = k * h
        k大于1放大 小于1缩小 中心位置不变
        """
        bbox1 = ax.get_position(original=True)
        x1, y1, w1, h1 = bbox1.xmin, bbox1.ymin, bbox1.width, bbox1.height
        x2 = x1 - (k - 1) * w1 / 2
        y2 = y1 - (k - 1) * h1 / 2
        w2 = k * w1
        h2 = k * h1
        bbox2 = mpl.transforms.Bbox.from_bounds(x2, y2, w2, h2)
        ax.set_position(bbox2)
        return
    
    def _add_cbar(self, ax1, ax2, mappable):
        """
        在ax1和ax2之间添加colorbar
        mappable为对应的映射参数
        """
        box1 = mpl.transforms.Bbox.from_extents(ax2.get_position().xmax+0.01,
                                                ax2.get_position().ymin,
                                                ax2.get_position().xmax+0.03,
                                                ax1.get_position().ymax)
        cax1 = self.fig.add_axes([.01,.01,.01,.01], position=box1)
        cbar1 = self.fig.colorbar(mappable, cax=cax1, extend="both")
        cbar1.ax.minorticks_on()
        cbar1.ax.set_yticks([0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
        cbar1.ax.tick_params(axis="both", which="both", direction="in",
                             labelsize=15, length=5, width=1.3)
        cax1.text(-0.2, 1.07, "S4max Intensity",
                  va="center", ha="center", transform=cax1.transAxes,
                  fontsize=14)
        pass
    
    
    def _draw_axes_0(self, ax):
        from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
        contourset = ax.contourf(self.grid_C, self.grid_R, self.s4max_density, 
                                 levels=50, cmap=self.cmap, 
                                 norm=mpl.colors.Normalize(0.15, 0.75),
                                 transform=cartopy.crs.PlateCarree())
        CS1 = ax.contour(self.grid_global_lon, self.grid_global_lat,  self.global_dip_lat, 
                         levels=np.arange(-60,61,20), colors=list("wwwrwww"),
                         linestyles="solid", linewidths=1.)
        # CS2 = ax.contourf(self.grid_global_lon, self.grid_global_lat,  self.global_dip_lat, 
        #                   levels=np.arange(-20,21,20), 
        #                   colors=["gray"], alpha=0.5)
        # CS2 = ax.contour(self.grid_global_lon, self.grid_global_lat,  self.global_decl, 
        #                  levels=np.arange(-60,60,20), colors="r",
        #                  linestyles="dashed")
        ax.clabel(CS1, levels=np.arange(-60,61,20), colors="red", fontsize=15)
        self._zoom_axes(ax, 1.1)
        ax.add_feature(cartopy.feature.COASTLINE)
        ax.set_global()
        
        ax.text(-0.06, 0.5, "COSMIC Data",
                transform=ax.transAxes, ha="center", va="center",
                rotation="vertical", fontsize=20, fontweight="bold")
        
        ax.text(0, 1.03, "(a)",
                transform=ax.transAxes, ha="left", va="center",
                rotation="horizontal", fontsize=18, fontweight="bold")
        ax.set_yticks([-90, -60, -30, 0, 30, 60, 90], crs=cartopy.crs.PlateCarree())
        lat_formatter = LatitudeFormatter(direction_label=False)
        ax.yaxis.set_major_formatter(lat_formatter)
        ax.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(n=4))
        ax.tick_params(axis='both', which='both', direction='in', 
                       top=True, right=True)
        ax.tick_params(axis="both", which="both", labelsize=14.9)
        ax.tick_params(axis="both", which="major", length=5, width=1.4)
        ax.tick_params(axis="both", which="minor", length=2.5, width=1.1)
        
        ax.gridlines(crs=cartopy.crs.PlateCarree(), ylocs=np.arange(-90,91,15),
                     xlocs=np.arange(-180, 181, 60), alpha=.5)
        
        # 添加三个方框
        # 同时在方框附近添加圆框数字
        loc1 = (-60, 15, 120, 30)
        loc2 = (100, 15, 60, 30)
        loc3 = (-120, -60, 60, 30)
        rec1 = mpl.patches.FancyBboxPatch((loc1[0], loc1[1]), width=loc1[2], height=loc1[3], transform=ax.transData,
                                          facecolor="none", edgecolor="#b63655", lw=1.5)
        ax.text(x=loc1[0]-5, y=loc1[1]+30, s="1", transform=ax.transData, fontweight="bold",
                fontsize=8, bbox=dict(boxstyle="circle", edgecolor="k", facecolor="white"))
        
        rec2 = mpl.patches.FancyBboxPatch((loc2[0], loc2[1]), width=loc2[2], height=loc2[3], transform=ax.transData,
                                          facecolor="none", edgecolor="#b63655", lw=1.5)
        ax.text(x=loc2[0]-5, y=loc2[1]+30, s="2", transform=ax.transData, fontweight="bold",
                fontsize=8, bbox=dict(boxstyle="circle", edgecolor="k", facecolor="white"))
        
        rec3 = mpl.patches.FancyBboxPatch((loc3[0], loc3[1]), width=loc3[2], height=loc3[3], transform=ax.transData,
                                          facecolor="none", edgecolor="#b63655", lw=1.5)
        ax.text(x=loc3[0]-5, y=loc3[1]+30, s="3", transform=ax.transData, fontweight="bold",
                fontsize=8, bbox=dict(boxstyle="circle", edgecolor="k", facecolor="white"))
        
        ax.add_patch(rec1)
        ax.add_patch(rec2)
        ax.add_patch(rec3)
        pass
    
    def _draw_axes_1(self, ax):
        from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
        contourset = ax.contourf(self.grid_C, self.grid_R, self.test_s4max_density, 
                                 levels=50, cmap=self.cmap,
                                 norm=mpl.colors.Normalize(0.15, 0.75),
                                 transform=cartopy.crs.PlateCarree())
        CS1 = ax.contour(self.grid_global_lon, self.grid_global_lat, self.global_dip_lat, 
                         levels=np.arange(-60,61,20), colors=list("wwwrwww"),
                         linestyles="solid", linewidths=1.)
        ax.clabel(CS1, levels=np.arange(-60,61,20), colors="red", fontsize=15)
        self._zoom_axes(ax, 1.1)
        ax.add_feature(cartopy.feature.COASTLINE)
        ax.set_global()
        
        ax.text(-0.06, 0.5, "Model",
                transform=ax.transAxes, ha="center", va="center",
                rotation="vertical", fontsize=20, fontweight="bold")
        
        ax.text(0, 1.03, "(b)",
                transform=ax.transAxes, ha="left", va="center",
                rotation="horizontal", fontsize=18, fontweight="bold")
        
        ax.set_xticks([-120, -60, 0, 60, 120, 180], crs=cartopy.crs.PlateCarree())
        ax.set_yticks([-90, -60, -30, 0, 30, 60, 90], crs=cartopy.crs.PlateCarree())
        lon_formatter = LongitudeFormatter(direction_label=False)
        lat_formatter = LatitudeFormatter(direction_label=False)
        ax.xaxis.set_major_formatter(lon_formatter)
        ax.yaxis.set_major_formatter(lat_formatter)
        ax.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(n=4))
        ax.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(n=4))
        ax.tick_params(axis='both', which='both', direction='in', 
                       top=True, right=True)
        ax.tick_params(axis="both", which="both", labelsize=14.9)
        ax.tick_params(axis="both", which="major", length=5, width=1.4)
        ax.tick_params(axis="both", which="minor", length=2.5, width=1.1)
        
        ax.gridlines(crs=cartopy.crs.PlateCarree(), ylocs=np.arange(-90,91,15),
                     xlocs=np.arange(-180, 181, 60), alpha=.5)
        
        
        # 添加三个方框
        loc1 = (-60, 15, 120, 30)
        loc2 = (100, 15, 60, 30)
        loc3 = (-120, -60, 60, 30)
        
        rec1 = mpl.patches.FancyBboxPatch((loc1[0], loc1[1]), width=loc1[2], height=loc1[3], transform=ax.transData,
                                          facecolor="none", edgecolor="#b63655", lw=1.5)
        ax.text(x=loc1[0]-5, y=loc1[1]+30, s="1", transform=ax.transData, fontweight="bold",
                fontsize=8, bbox=dict(boxstyle="circle", edgecolor="k", facecolor="white"))
        
        rec2 = mpl.patches.FancyBboxPatch((loc2[0], loc2[1]), width=loc2[2], height=loc2[3], transform=ax.transData,
                                          facecolor="none", edgecolor="#b63655", lw=1.5)
        ax.text(x=loc2[0]-5, y=loc2[1]+30, s="2", transform=ax.transData, fontweight="bold",
                fontsize=8, bbox=dict(boxstyle="circle", edgecolor="k", facecolor="white"))
        
        rec3 = mpl.patches.FancyBboxPatch((loc3[0], loc3[1]), width=loc3[2], height=loc3[3], transform=ax.transData,
                                          facecolor="none", edgecolor="#b63655", lw=1.5)
        ax.text(x=loc3[0]-5, y=loc3[1]+30, s="3", transform=ax.transData, fontweight="bold",
                fontsize=8, bbox=dict(boxstyle="circle", edgecolor="k", facecolor="white"))
        
        # 创建放置legend的bbox
        box_to_legend = mpl.transforms.Bbox.from_extents(ax.get_position().xmin,
                                                         ax.get_position().ymin-0.07,
                                                         ax.get_position().xmax,
                                                         ax.get_position().ymin-0.02)
        
        # 计算方框内的residual的mean和std
        rec1_data = self._cal_rec(loc1)
        rec2_data = self._cal_rec(loc2)
        rec3_data = self._cal_rec(loc3)
        legend_label = [f"{rec1_data[0]:.3f}$\pm${rec1_data[1]:.3f}",
                        f"{rec2_data[0]:.3f}$\pm${rec2_data[1]:.3f}",
                        f"{rec3_data[0]:.3f}$\pm${rec3_data[1]:.3f}"]
        
        # 注意将 frameon=False 可以避免框架和其他元素叠加
        rec_legend = self.fig.legend([rec1, rec2, rec3], legend_label, 
                                     bbox_to_anchor=box_to_legend, ncols=3,
                                     loc=6, edgecolor="none", columnspacing=4.0,
                                     handleheight=4,handlelength=7, fontsize=12,
                                     frameon=False)
        ax.add_patch(rec1)
        ax.add_patch(rec2)
        ax.add_patch(rec3)
        
        # 添加相应的数字标识不同的legend
        ax.text(x=-155, y=-110, s="1", transform=ax.transData, fontweight="bold",
                fontsize=10, bbox=dict(boxstyle="circle", edgecolor="k", facecolor="white"))
        ax.text(x=-67+25, y=-110, s="2", transform=ax.transData, fontweight="bold",
                fontsize=10, bbox=dict(boxstyle="circle", edgecolor="k", facecolor="white"))
        ax.text(x=45+25, y=-110, s="3", transform=ax.transData, fontweight="bold",
                fontsize=10, bbox=dict(boxstyle="circle", edgecolor="k", facecolor="white"))
        
        
        pass
    
    def _cal_rec(self, loc):
        # 计算方框内的值
        # loc=(x, y, width, height)
        
        CC = (self.grid_C>=loc[0])&(self.grid_C<=(loc[0]+loc[2]))
        RR = (self.grid_R>=loc[1])&(self.grid_R<=(loc[1]+loc[3]))
        s1 = self.s4max_density[(RR&CC)]
        s2 = self.test_s4max_density[(RR&CC)]
        return ((s1-s2).mean(), (s1-s2).std())
    
    pass
    

class OccurrenceDistribution(_PlotBase):
    """
    S4max-alt S4max-lct的分布统计
    发生率的分布统计比较
    """
    def __init__(self, backend: str,
                 file_path: str):
        super().__init__(backend, file_path)
        self._private_color = [(158/255, 202/255, 225/255),
                               (8/255, 48/255, 107/255),
                               ]
        pass
    
    def _load_data(self, file_path):
        file_path = Path(file_path)
        df = pd.read_hdf(file_path)
        
        alt_series = df.loc[:, "alt"]
        select_index = (alt_series >= 80) & (alt_series <= 130)
        select_df = df.loc[select_index, :]
        self.alt = select_df.loc[:, "alt"].to_numpy()
        self.lct = select_df.loc[:, "lct"].to_numpy()
        self.s4max = select_df.loc[:, "s4max"].to_numpy()
        self.test_s4max = select_df.loc[:, "test_s4max"].to_numpy()
        
        lat = select_df.loc[:, "lat"].to_numpy()
        lon = select_df.loc[:, "lon"].to_numpy()
        self.dip_lat = self._glat2diplat(lat, lon, self.alt)
        
        # a = self.test_s4max[self.test_s4max==0.13136229]
        # print(f"{a.shape}")
        pass
    
    @staticmethod
    @njit(parallel=True)
    def _glat2diplat(glat, glon, alt):
        """
        给定经纬度,高度转化为dip纬度
        """
        assert len(glat.shape)==1
        assert len(glon.shape)==1
        assert len(alt.shape)==1
        assert glat.shape==alt.shape
        
        # 把地理经纬度转化为dip纬度
        year = 2014
        with objmode(g="float64[:,:]", h="float64[:,:]"):
            g, h = igrf.ReadCoeffs().get_coeffs(year)
            pass
        dip_lat = np.zeros_like(glat)
        for i in prange(len(glat)):
            (dip, 
             gd, gi, 
             bh, bx, by, bz, bf) = igrf.igrf_value(g, h, 
                                                   glat[i], glon[i], alt[i], year)
            # dip纬度
            dip_lat[i] = dip
            pass
        return dip_lat
    
    @staticmethod
    @njit(parallel=True)
    def _compute_occurrence(coords, value, tag="s4max-alt"):
        """
        统计给定网格间隔内的均值
        划分coords的值 统计每个间隔内的均值/数目
        不同的具体参数可能有不同的间隔密度
        
        统计发生率
        在coords的给定间隔内统计Es发生
        occurrence = (value>=0.2的数目)/(总数目)
        
        return:
            (网格间隔数组, 每个间隔内的value总和, 每个间隔内的总数目,
             每个间隔内Es数目, 每个间隔内非Es数目)
        """
        threshold = 0.2
        assert len(coords.shape)==1
        assert len(value.shape)==1
        if tag == "s4max-s4max":
            grid_coords = np.arange(0.025, 2, 0.05)
            pass
        elif tag == "s4max-alt":
            # alt 80~130 km
            # grid_coords = np.arange(80.5, 130, 1)
            grid_coords = np.arange(81., 130, 2)
            pass
        elif tag == "s4max-lct":
            grid_coords = np.arange(0.5, 24, 1)
            pass
        elif tag == "s4max-dip_lat":
            grid_coords = np.arange(-88.,90,4)
            # grid_glat = np.arange(-89.5,90,1)
            # grid_glat = np.arange(-90+(1.5/2),90,1.5)
            # grid_glat = np.arange(-90+1., 90., 2)
            pass
        else:
            pass
        
        delta = grid_coords[1] - grid_coords[0]
        
        # 每个间隔内Es的数目
        N_es = np.zeros_like(grid_coords)
        # 非Es的数目
        N_nes = np.zeros_like(grid_coords)
        # 总数目
        N_all = np.zeros_like(grid_coords)
        Z = np.zeros_like(grid_coords)
        
        for i in prange(Z.shape[0]):
            for di in prange(len(value)):
                coordsi = coords[di]
                valuei = value[di]
                grid_coordsi = grid_coords[i]
                
                if (coordsi>=grid_coordsi-(delta/2))&(coordsi<grid_coordsi+(delta/2)):
                    N_all[i] += 1
                    Z[i] += valuei
                    if valuei >= threshold:
                        N_es[i] +=1
                        pass
                    else:
                        N_nes[i] += 1
                        pass
                    pass
                pass
            pass
        return (grid_coords, Z, N_all, N_es, N_nes)
    
    def plot(self,):
        figsize = (12, 6.5)
        dpi = 200
        nrows = 1
        ncols = 3
        axes = []
        
        self.fig = plt.figure(figsize=figsize, dpi=dpi, 
                              facecolor="white")
        gs = self.fig.add_gridspec(nrows=nrows, ncols=ncols, 
                                   left=0.15, right=0.95, bottom=0.09, top=0.8,
                                   wspace=0.4, hspace=0.2)
        
        for j in range(ncols):
            axes.append(self.fig.add_subplot(gs[0, j]))
            pass
        
        self._draw_axes_0(axes[0])
        self._draw_axes_1(axes[1])
        self._draw_axes_2(axes[2])
        
        # 调整轴刻度细节
        for ax in axes:
            ax.tick_params(axis="both", which="both", direction="in")
            ax.yaxis.get_ticklabels()[0].set_visible(False)
        self._save_fig()
        return
    
    def _set_alt_tick(self, ax):
        """
        设置高度的刻度细节
        """
        alt = np.linspace(80, 130, num=6, dtype=np.int32)
        altstr = []
        for i, j in enumerate(alt):
            altstr.append(f"{j}")
            pass
        
        lct = mpl.ticker.FixedLocator(alt)
        fmt = mpl.ticker.FixedFormatter(altstr)
        
        ax.xaxis.set_view_interval(80, 130, ignore=True)
        ax.xaxis.set_major_locator(lct)
        ax.xaxis.set_major_formatter(fmt)
        ax.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(n=2))
        ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
        
        pass
    
    def _set_lct_tick(self, ax):
        """
        设定当地时刻度
        """
        lct = np.linspace(0, 24, num=5, dtype=np.int32)
        lctstr = []
        for i, j in enumerate(lct):
            lctstr.append(f"{j}")
            pass
        lct = mpl.ticker.FixedLocator(lct)
        fmt = mpl.ticker.FixedFormatter(lctstr)
        
        ax.xaxis.set_view_interval(0, 24, ignore=True)
        ax.xaxis.set_major_locator(lct)
        ax.xaxis.set_major_formatter(fmt)
        ax.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(n=2))
        ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())

        pass
    
    def _set_lat_tick(self, ax):
        """
        设置纬度刻度
        """
        lat = np.linspace(-90, 90, num=13, dtype=np.int32)
        latstr = []
        for i, j in enumerate(lat):
            if i%2==0:
                jj = str(j) + "$^{\circ}$"
                latstr.append(f"{jj}")
                pass
            else:
                latstr.append("")
                pass
            pass
        lct = mpl.ticker.FixedLocator(lat)
        fmt = mpl.ticker.FixedFormatter(latstr)

        ax.xaxis.set_view_interval(-90, 90, ignore=True)
        ax.xaxis.set_major_locator(lct)
        ax.xaxis.set_major_formatter(fmt)
        ax.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(n=2))
        ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
        pass
    
    def _set_s4max_tick(self, ax):
        """
        设定S4max强度刻度
        """
        value = np.arange(0, 2.1, 0.5)
        vstr = [f"{v}" for v in value]
        
        lct = mpl.ticker.FixedLocator(value)
        fmt = mpl.ticker.FixedFormatter(vstr)
        
        ax.yaxis.set_view_interval(0, 2, ignore=True)
        ax.yaxis.set_major_locator(lct)
        ax.yaxis.set_major_formatter(fmt)
        ax.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(n=4))
        ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
        
        pass
    
    def _set_occurrence_tick(self, ax):
        """
        设定发生率刻度
        """
        value = np.arange(0, 1.1, 0.2)
        vstr = [f"{v:.1f}" for v in value]
        
        lct = mpl.ticker.FixedLocator(value)
        fmt = mpl.ticker.FixedFormatter(vstr)
        
        ax.yaxis.set_view_interval(0, 1, ignore=True)
        ax.yaxis.set_major_locator(lct)
        ax.yaxis.set_major_formatter(fmt)
        ax.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(n=4))
        ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
        pass
    
    def _set_ratio1_tick(self, ax):
        """
        设定s4max数目统计刻度
        """
        value = [0.05, 0.1, 0.15]
        vstr = [f"{v:.0%}" for v in value]
        
        lct = mpl.ticker.FixedLocator(value)
        fmt = mpl.ticker.FixedFormatter(vstr)
        
        ax.xaxis.set_view_interval(0, 0.188, ignore=True)
        ax.xaxis.set_major_locator(lct)
        ax.xaxis.set_major_formatter(fmt)
        ax.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(n=2))
        ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
        pass
    
    def _set_ratio2_tick(self, ax):
        """
        设定高度数目统计刻度
        """
        value = [0.05, 0.1]
        vstr = [f"{v:.0%}" for v in value]
        
        lct = mpl.ticker.FixedLocator(value)
        fmt = mpl.ticker.FixedFormatter(vstr)
        
        ax.yaxis.set_view_interval(0, 0.1, ignore=True)
        ax.yaxis.set_major_locator(lct)
        ax.yaxis.set_major_formatter(fmt)
        ax.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(n=2))
        ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
        pass
    
    def _set_ratio3_tick(self, ax):
        """
        设定local time 数目刻度
        """
        value = [0.02, 0.05]
        vstr = [f"{v:.0%}" for v in value]
        
        lct = mpl.ticker.FixedLocator(value)
        fmt = mpl.ticker.FixedFormatter(vstr)
        
        ax.yaxis.set_view_interval(0, 0.055, ignore=True)
        ax.yaxis.set_major_locator(lct)
        ax.yaxis.set_major_formatter(fmt)
        ax.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(n=2))
        ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
        pass
    
    def _set_ratio4_tick(self, ax):
        """
        设定dip latitude数目统计刻度
        """
        value = [0.02, 0.05]
        vstr = [f"{v:.0%}" for v in value]
        
        lct = mpl.ticker.FixedLocator(value)
        fmt = mpl.ticker.FixedFormatter(vstr)
        
        ax.yaxis.set_view_interval(0, 0.07, ignore=True)
        ax.yaxis.set_major_locator(lct)
        ax.yaxis.set_major_formatter(fmt)
        ax.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(n=2))
        ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
        
        pass
    
    def _draw_axes_0(self, ax):
        """
        绘制第一个axes
        该axes有左边和上边有两个小axes示意统计结果
        """
        # 创建附着坐标图
        source_bbox = ax.get_position()
        axes_bbox1 = mpl.transforms.Bbox.from_extents(source_bbox.xmin-0.14, source_bbox.ymin,
                                                      source_bbox.xmin-0.05, source_bbox.ymax)
        axes_bbox2 = mpl.transforms.Bbox.from_extents(source_bbox.xmin, source_bbox.ymax+0.045,
                                                      source_bbox.xmax, source_bbox.ymax+0.165)
        twine_ax1 = self.fig.add_axes((0,0,0,0), position=axes_bbox1)
        twine_ax2 = self.fig.add_axes((0,0,0,0), position=axes_bbox2)
        
        # 左边的axes绘制真实s4max的数目统计分布
        (s4max_grid_coords, _, N_s4max, _, _) = self._compute_occurrence(self.s4max, self.s4max,
                                                                         tag="s4max-s4max")
        twine_ax1.barh(s4max_grid_coords, width=N_s4max/np.sum(N_s4max),
                       height=0.049, left=0, align="center",
                       color=self._private_color[0], edgecolor=self._private_color[1],
                       linewidth=0.7)
        self._set_s4max_tick(twine_ax1)
        self._set_ratio1_tick(twine_ax1)
        twine_ax1.invert_xaxis()
        twine_ax1.tick_params(axis="y", which="both", left=False, labelleft=False)
        twine_ax1.tick_params(axis="x", which="both", labelsize=9)
        twine_ax1.text(0, 1.03, "(d)",
                       transform=twine_ax1.transAxes, ha="left", va="center",
                       rotation="horizontal", fontsize=14, fontweight="bold")
        twine_ax1.set_xlabel("Ratio", fontsize=12)
        
        # 上边axes绘制沿着高度的统计分布
        # 同时也给出了发生率随高度变化
        (alt_grid_coords, _, N_alt, 
         _, _) = self._compute_occurrence(self.alt, self.s4max,
                                          tag="s4max-alt")
        twine_ax2.bar(alt_grid_coords, N_alt/np.sum(N_alt),
                      width=1.9, align="center",
                      color=self._private_color[0], edgecolor=self._private_color[1],
                      linewidth=0.7)
        self._set_alt_tick(twine_ax2)
        self._set_ratio2_tick(twine_ax2)
        twine_ax2.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
        twine_ax2.text(0, 1.15, "(a)",
                       transform=twine_ax2.transAxes, ha="left", va="center",
                       rotation="horizontal", fontsize=14, fontweight="bold")
        twine_ax2.set_ylabel("Ratio", fontsize=12, labelpad=2)
        
        # 当前的axes首先绘制散点图
        alt_coords = self.alt.copy()
        s4max_coords = self.s4max.copy()
        test_s4max_coords = self.test_s4max.copy()
        
        sindex = (test_s4max_coords==0.13136229)
        alt_coords = alt_coords[~sindex]
        test_s4max_coords = test_s4max_coords[~sindex]
        s4max_coords = s4max_coords[~sindex]
        
        ax.scatter(alt_coords, s4max_coords, s=.5, 
                   color="#70b9e0", alpha=0.7, marker="o",
                   edgecolors="none")
        # 
        ax.scatter(alt_coords, test_s4max_coords, s=.5, 
                   color="#9D9B9B", alpha=0.2, marker="o",
                   edgecolors="none")
        ax.text(0, 1.03, "(e)",
                transform=ax.transAxes, ha="left", va="center",
                rotation="horizontal", fontsize=14, fontweight="bold")
        ax.set_xlabel("Altitude", fontsize=12, labelpad=7)
        ax.set_ylabel("S4max Intensity", fontsize=12, loc="bottom")
        
        # 当前axes创建一个孪生轴绘制s4max随高度均值变化/发生率随高度变化
        ax_twinx = ax.twinx()
        (alt_grid_coords, Z_1, N_all_1, 
         N_es_1, N_nes_1) = self._compute_occurrence(self.alt, self.s4max,
                                                     tag="s4max-alt")
        (alt_grid_coords, Z_2, N_all_2, 
         N_es_2, N_nes_2) = self._compute_occurrence(self.alt, self.test_s4max,
                                                     tag="s4max-alt")
        line1, = ax_twinx.plot(alt_grid_coords, N_es_1/N_all_1, c="#1580BD", drawstyle="steps")
        line2, = ax_twinx.plot(alt_grid_coords, N_es_2/N_all_2, c="#808080", drawstyle="steps")
        ax_twinx.set_ylabel("Occurrence Rate", fontsize=12, loc="top", color="red")
        blue_circle = mpl.lines.Line2D([],[], marker="o", ls="", color="#1580BD")
        gray_circle = mpl.lines.Line2D([],[], marker="o", ls="", color="#808080")
        point_legend = ax_twinx.legend([blue_circle, gray_circle], ["Data", "Model"],
                                       loc="upper right", edgecolor="black",
                                       markerscale=0.5, fontsize=10)
        ax_twinx.add_artist(point_legend)
        ax_twinx.legend(handles=[line1, line2], 
                        labels=["Data", "Model"], loc="upper left",
                        edgecolor="black", fontsize=10)
        ax_twinx.yaxis.get_ticklabels()[0].set_visible(False)
        self._set_occurrence_tick(ax_twinx)
        self._set_alt_tick(ax_twinx)
        ax_twinx.tick_params(axis="both", which="both", direction="in",
                             colors="red")
        ax_twinx.spines["right"].set_color("red")
        self._set_s4max_tick(ax)
        self._set_alt_tick(ax)
        pass
        
    def _draw_axes_1(self, ax):
        """
        绘制第二个axes
        该axes的上边有一个小axes示意统计结果
        """
        # 创建附着的子图
        source_bbox = ax.get_position()
        axes_bbox = mpl.transforms.Bbox.from_extents(source_bbox.xmin, source_bbox.ymax+0.045,
                                                     source_bbox.xmax, source_bbox.ymax+0.165)
        twine_ax = self.fig.add_axes((0,0,0,0), position=axes_bbox)
        
        # 上边axes绘制沿着当地时的统计分布
        # 同时也给出了发生率随当地时变化
        (lct_grid_coords, _, N_lct, 
         _, _) = self._compute_occurrence(self.lct, self.s4max,
                                          tag="s4max-lct")
        twine_ax.bar(lct_grid_coords, N_lct/np.sum(N_lct),
                      width=0.95, align="center",
                      color=self._private_color[0], edgecolor=self._private_color[1],
                      linewidth=0.7)
        self._set_lct_tick(twine_ax)
        self._set_ratio3_tick(twine_ax)
        twine_ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
        twine_ax.text(0, 1.15, "(b)",
                      transform=twine_ax.transAxes, ha="left", va="center",
                      rotation="horizontal", fontsize=14, fontweight="bold")
        twine_ax.set_ylabel("Ratio", fontsize=12)
        
        # 当前axes绘制散点图
        lct_coords = self.lct.copy()
        s4max_coords = self.s4max.copy()
        test_s4max_coords = self.test_s4max.copy()
        ax.scatter(lct_coords, s4max_coords, s=.5, 
                   color="#70b9e0", alpha=0.7, marker="o",
                   edgecolors="none")
        ax.scatter(lct_coords, test_s4max_coords, s=.5, 
                   color="#9D9B9B", alpha=0.2, marker="o",
                   edgecolors="none")
        ax.text(0, 1.03, "(f)",
                transform=ax.transAxes, ha="left", va="center",
                rotation="horizontal", fontsize=14, fontweight="bold")
        ax.set_xlabel("Local Time", fontsize=12, labelpad=7)
        ax.set_ylabel("S4max Intensity", fontsize=12, loc="bottom")
        
        # 当前axes创建孪生轴
        # 绘制s4max随当地时均值变化/发生率随当地时变化
        ax_twinx = ax.twinx()
        (lct_grid_coords, Z_1, N_all_1, 
         N_es_1, N_nes_1) = self._compute_occurrence(self.lct, self.s4max,
                                                     tag="s4max-lct")
        (lct_grid_coords, Z_2, N_all_2, 
         N_es_2, N_nes_2) = self._compute_occurrence(self.lct, self.test_s4max,
                                                     tag="s4max-lct")
        line1, = ax_twinx.plot(lct_grid_coords, N_es_1/N_all_1, c="#1580BD", drawstyle="steps")
        line2, = ax_twinx.plot(lct_grid_coords, N_es_2/N_all_2, c="#808080", drawstyle="steps")
        ax_twinx.set_ylabel("Occurrence Rate", fontsize=12, loc="top", color="red")
        blue_circle = mpl.lines.Line2D([],[], marker="o", ls="", color="#1580BD")
        gray_circle = mpl.lines.Line2D([],[], marker="o", ls="", color="#808080")
        point_legend = ax_twinx.legend([blue_circle, gray_circle], ["Data", "Model"],
                                       loc="upper right", edgecolor="black",
                                       markerscale=0.5, fontsize=10)
        ax_twinx.add_artist(point_legend)
        ax_twinx.legend(handles=[line1, line2], 
                        labels=["Data", "Model"], loc="upper left",
                        edgecolor="black", fontsize=10)
        ax_twinx.yaxis.get_ticklabels()[0].set_visible(False)
        self._set_occurrence_tick(ax_twinx)
        self._set_lct_tick(ax_twinx)
        ax_twinx.tick_params(axis="both", which="both", direction="in",
                             colors="red")
        ax_twinx.spines["right"].set_color("red")
        self._set_lct_tick(ax)
        self._set_s4max_tick(ax)
        pass
        
    def _draw_axes_2(self, ax):
        """
        绘制第三个axes
        该axes的上边有一个小axes示意统计结果
        """
        source_bbox = ax.get_position()
        axes_bbox = mpl.transforms.Bbox.from_extents(source_bbox.xmin, source_bbox.ymax+0.045,
                                                     source_bbox.xmax, source_bbox.ymax+0.165)
        twine_ax = self.fig.add_axes((0,0,0,0), position=axes_bbox)
        
        # 上边axes绘制沿着磁纬度的统计分布
        # 同时也给出了发生率随磁纬变化
        (dip_lat_grid_coords, _, N_dip_lat, 
         _, _) = self._compute_occurrence(self.dip_lat, self.s4max,
                                          tag="s4max-dip_lat")
        twine_ax.bar(dip_lat_grid_coords, N_dip_lat/np.sum(N_dip_lat),
                      width=4, align="center",
                      color=self._private_color[0], edgecolor=self._private_color[1],
                      linewidth=0.7)
        self._set_lat_tick(twine_ax)
        self._set_ratio4_tick(twine_ax)
        twine_ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
        twine_ax.text(0, 1.15, "(c)",
                      transform=twine_ax.transAxes, ha="left", va="center",
                      rotation="horizontal", fontsize=14, fontweight="bold")
        twine_ax.set_ylabel("Ratio", fontsize=12)
        
        # 当前axes绘制散点图
        dip_lat_coords = self.dip_lat.copy()
        s4max_coords = self.s4max.copy()
        test_s4max_coords = self.test_s4max.copy()
        # 真实值 蓝色
        PC1 = ax.scatter(dip_lat_coords, s4max_coords, s=.5, 
                   color="#70b9e0", alpha=0.7, marker="o",
                   edgecolors="none")
        PC2 = ax.scatter(dip_lat_coords, test_s4max_coords, s=.5, 
                   color="#9D9B9B", alpha=0.2, marker="o",
                   edgecolors="none")
        ax.text(0, 1.03, "(g)",
                transform=ax.transAxes, ha="left", va="center",
                rotation="horizontal", fontsize=14, fontweight="bold")
        ax.set_xlabel("Geomagnetic Latitude", fontsize=12, labelpad=7)
        ax.set_ylabel("S4max Intensity", fontsize=12, loc="bottom")
        
        # 当前axes创建孪生轴绘制s4max随当地时均值变化/发生率随当地时变化
        ax_twinx = ax.twinx()
        (dip_lat_grid_coords, Z_1, N_all_1, 
         N_es_1, N_nes_1) = self._compute_occurrence(self.dip_lat, self.s4max,
                                                     tag="s4max-dip_lat")
        (dip_lat_grid_coords, Z_2, N_all_2, 
         N_es_2, N_nes_2) = self._compute_occurrence(self.dip_lat, self.test_s4max,
                                                     tag="s4max-dip_lat")
        
        line1, = ax_twinx.plot(dip_lat_grid_coords, N_es_1/N_all_1, c="#1580BD", drawstyle="steps")
        line2, = ax_twinx.plot(dip_lat_grid_coords, N_es_2/N_all_2, c="#808080", drawstyle="steps")
        ax_twinx.set_ylabel("Occurrence Rate", fontsize=12, loc="top", color="red")
        blue_circle = mpl.lines.Line2D([],[], marker="o", ls="", color="#1580BD")
        gray_circle = mpl.lines.Line2D([],[], marker="o", ls="", color="#808080")
        point_legend = ax_twinx.legend([blue_circle, gray_circle], ["Data", "Model"],
                                       loc="upper right", edgecolor="black",
                                       markerscale=0.5, fontsize=10)
        ax_twinx.add_artist(point_legend)
        ax_twinx.legend(handles=[line1, line2], 
                        labels=["Data", "Model"], loc="upper left",
                        edgecolor="black", fontsize=10)
        ax_twinx.yaxis.get_ticklabels()[0].set_visible(False)
        self._set_occurrence_tick(ax_twinx)
        self._set_lat_tick(ax_twinx)
        ax_twinx.tick_params(axis="both", which="both", direction="in",
                             colors="red")
        ax_twinx.spines["right"].set_color("red")
        self._set_lat_tick(ax)
        self._set_s4max_tick(ax)
    pass


class EvaluationMetric(_PlotBase):
    """
    图一绘制观测和模型输出的RMSE/MAE/MSE的柱状图
    图二绘制Bland-Altman图
    图三绘制观测和模型输出的相关系数图 contour图
    图四绘制模型输出和某个台站的foEs的相关系数图 点图
    """
    def __init__(self, backend: str,
                 file_path: str,
                 foEs_infer_file):
        super().__init__(backend, file_path)
        self.foEs_df = pd.read_hdf(foEs_infer_file)
        pass
    
    def _load_data(self, file_path):
        file_path = Path(file_path)
        df = pd.read_hdf(file_path)
        
        alt_series = df.loc[:, "alt"]
        select_index = (alt_series >= 80) & (alt_series <= 130)
        select_df = df.loc[select_index, :]
        self.alt = select_df.loc[:, "alt"].to_numpy()
        # self.lct = select_df.loc[:, "lct"].to_numpy()
        self.s4max = select_df.loc[:, "s4max"].to_numpy()
        # self.s4max = select_df.loc[:, "target_s4max"].to_numpy()
        self.test_s4max = select_df.loc[:, "test_s4max"].to_numpy()
        self.lat = select_df.loc[:, "lat"].to_numpy()
        self.lon = select_df.loc[:, "lon"].to_numpy()
        
        pass
    
    @staticmethod
    @njit(parallel=True)
    def _count_num(value):
        """
        统计残差序列的柱状分布
        给定间隔统计其数目
        return:
            (网格间隔数组, 每个间隔内的总数目)
        """
        assert len(value.shape)==1
        grid_coords = np.arange(-2, 2.1, 0.2)
        delta = grid_coords[1] - grid_coords[0]
        
        # 每个间隔内的数目
        N = np.zeros_like(grid_coords)
        
        for i in prange(N.shape[0]):
            for di in prange(len(value)):
                valuei = value[di]
                grid_coordsi = grid_coords[i]
                
                if (valuei>=grid_coordsi-(delta/2))&(valuei<grid_coordsi+(delta/2)):
                    N[i] += 1
                    pass
                pass
            pass
        return (grid_coords, N)
    
    @staticmethod
    @njit(parallel=True, cache=False)
    def _compute_count(row_coords, col_coords):
        """
        与class StatisticalDistribution._compute_count一致
        统计网格内的数值
        网格方向按照直觉性排列
        左上为(row_coords[0,0], col_coords[0,0])
        左下为(row_coords[imax,0], col_coords[imax,0])
        """
        assert len(row_coords.shape) == 1
        assert len(col_coords.shape) == 1
        with objmode(row_grid="float64[:,:]", col_grid="float64[:,:]", 
                     delta_row="float64", delta_col="float64"):
            row_grid, col_grid = np.meshgrid(np.linspace(0,1,100),
                                             np.linspace(0,1,100),
                                             indexing="ij")
            delta_row = np.diff(row_grid, axis=0)[0,0]
            delta_col = np.diff(col_grid, axis=1)[0,0]
            pass
        N = np.zeros_like(row_grid)
        for i in prange(N.shape[0]):
            for j in prange(N.shape[1]):
                for data_index in prange(len(row_coords)):
                    rr = row_coords[data_index]
                    cc = col_coords[data_index]
                    
                    rp = row_grid[i, j]
                    cp = col_grid[i, j]
                    if ((rr>=rp-(delta_row/2))&(rr<rp+(delta_row/2))&
                        (cc>=cp-(delta_col/2))&(cc<cp+(delta_col/2))):
                        N[i, j] += 1
                        pass
                    pass
                pass
            pass
        return (row_grid, col_grid, N)
    
    def plot(self,):
        figsize = (12, 12)
        dpi = 200
        nrows = 2
        ncols = 2
        axes = []
        
        self.fig = plt.figure(figsize=figsize, dpi=dpi, 
                              facecolor="white")
        gs = self.fig.add_gridspec(nrows=nrows, ncols=ncols, 
                                   left=0.08, right=0.96, bottom=0.05, top=0.96,
                                   wspace=0.25, hspace=0.25)
        for i in range(nrows):
            for j in range(ncols):
                axes.append(self.fig.add_subplot(gs[i, j]))
                pass
            pass
        
        for (i, n) in zip(range(nrows * ncols), list("abcd")):
            axes[i].set_box_aspect(1)
            axes[i].text(0, 1.05, f"({n})",
                         transform=axes[i].transAxes, ha="left", va="center",
                         rotation="horizontal", fontsize=18, fontweight="bold")
            pass
        
        self._draw_axes_1(axes[0])
        self._draw_axes_2(axes[1])
        self._draw_axes_3(axes[2])
        self._draw_axes_4(axes[3])
        
        
        self._save_fig()
        return
    
    def _draw_axes_1(self, ax):
        """
        绘制观测值和模型值的残差柱状图
        """
        residual = self.s4max - self.test_s4max
        (grid_coords, N) = self._count_num(residual)
        
        ratio = np.sum(N[9:12]/len(residual))
        rec = mpl.patches.Rectangle(xy=(-0.3, 0), width=0.6, height=0.5,
                                    facecolor="none", edgecolor="red", 
                                    zorder=10, lw=2, transform=ax.transData)
        ax.text(0.4, 0.45, rf"$\approx${ratio:.0%}", color="red", fontsize=20,
                transform=ax.transData)
        ax.bar(grid_coords, N/len(residual),
               width=0.2, align="center",
               color="#6dadd3", edgecolor="black",
               linewidth=0.7)
        
        me = np.mean(self.test_s4max - self.s4max)
        mae = np.mean(np.abs(self.test_s4max - self.s4max))
        rmse = np.sqrt(np.mean((self.test_s4max - self.s4max)**2))
        
        ax.text(0.04, 0.88, f"ME={me:.2f}\nMAE={mae:.2f}\nRMSE={rmse:.2f}", color="black",
                fontsize=16, transform=ax.transAxes, ha="left", va="center",
                fontfamily="sans", bbox=dict(boxstyle="round", facecolor="none", edgecolor="black"))
        
        ax.add_patch(rec)
        self._set_axes1_tick(ax)
        
        ax.set_xlabel("Residual", fontsize=21, labelpad=12)
        ax.set_ylabel("Frequency", fontsize=21, labelpad=10)
        ax.tick_params(axis="both", which="both", direction="in", 
                       labelsize=15)
        ax.xaxis.get_ticklabels()[0].set_visible(False)
        ax.grid(ls="--", alpha=0.7)
        pass
    
    def _draw_axes_2(self, ax):
        """
        绘制Altman-Bland图 s4max against test_s4max
        具体方法参考文章
        Confidence in Altman–Bland plots: A critical review of 
        the method of differences
        """
        # 计算两个连续变量的均值和差值
        average = (self.s4max + self.test_s4max) / 2
        difference = self.s4max - self.test_s4max
        
        # 对均值和差值序列线性拟合
        # 在把均值代入拟合结果得到预测均值
        fit_curve1 = LinearRegression().fit(average.reshape(-1, 1), difference)
        predicted_difference = fit_curve1.predict(average.reshape(-1, 1)).squeeze()
        
        # 计算均值和预测均值的差值的绝对值
        # 对差值绝对值和均值进行线性拟合
        absolute_residual = np.abs(difference - predicted_difference)
        fit_curve2 = LinearRegression().fit(average.reshape(-1, 1), absolute_residual)
        
        # 给出均值的平均线 均值和差值拟合线
        # 95%置信度的上下界线
        xx = np.arange(0.05, 1.5, 0.1)
        tt = np.arange(0.1, 1.2, 0.1)
        mean_difference = np.mean(difference)
        curve1_xx = fit_curve1.predict(xx.reshape(-1, 1)).squeeze()
        curve1_tt = fit_curve1.predict(tt.reshape(-1, 1)).squeeze()
        curve2_tt_upper = curve1_tt + 1.96 * 1.2533 * fit_curve2.predict(tt.reshape(-1, 1)).squeeze()
        curve2_tt_lower = curve1_tt - 1.96 * 1.2533 * fit_curve2.predict(tt.reshape(-1, 1)).squeeze()
        ax.scatter(average, difference, s=.5, 
                   color="#fab051", alpha=0.1, marker="o",
                   edgecolors="none")
        # gray
        ax.plot(xx, curve1_xx, c="black", lw=2.2)
        ax.plot(tt, curve2_tt_upper, c="red", ls="--", lw=2)
        ax.plot(tt, curve2_tt_lower, c="red", ls="--", lw=2)
        
        self._set_axes2_tick(ax)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlabel("Average", fontsize=21, labelpad=12)
        ax.set_ylabel("Difference", fontsize=21, labelpad=5)
        ax.tick_params(axis="both", which="both", direction="in", 
                       labelsize=15)
        ax.xaxis.get_ticklabels()[0].set_visible(False)
        
        ax.text(0.85, 1.5, "UCL", transform=ax.transData,
                va="center", ha="left", fontsize=17, color="red",
                fontfamily="sans")
        ax.text(1.15, -1.28, "LCL", transform=ax.transData,
                va="center", ha="left", fontsize=17, color="red",
                fontfamily="sans")
        pass
    
    def _draw_axes_3(self, ax):
        """
        绘制观测值和模型的相关系数图
        """
        # true-infer
        grid_true, grid_infer, grid_count = self._compute_count(self.s4max, self.test_s4max)
        
        # 计算相关系数
        # R_value, P_value = scipy.stats.pearsonr(self.s4max, self.test_s4max)
        R_value, P_value = scipy.stats.spearmanr(self.s4max, self.test_s4max)
        
        # 对计数进行归一化
        N = grid_count
        shape = N.shape
        N = N.reshape(-1,1)
        N = MinMaxScaler().fit_transform(N)
        N = N.reshape(shape)
        # N = np.ma.masked_where((N==0), N)
        
        ####################################
        # 设定色图，0设置为白色
        # 自定义色图 截取某个色图的一部分以便加强视觉效果
        cmap = mpl.cm.get_cmap("RdYlBu_r").copy()
        carr = cmap(np.linspace(0,1,256))[100:]
        carr[:4] = np.array([249/255,249/255,249/255,.1])
        colormap = mpl.colors.LinearSegmentedColormap.from_list("a", carr)
        colormap.set_under("white")
        
        contourset = ax.pcolormesh(grid_true, grid_infer, N, cmap=colormap, 
                                   norm=mpl.colors.Normalize(0, 1),
                                   alpha=0.95)
        # contourset = ax.contourf(grid_true, grid_infer, N, 
        #                          levels=np.linspace(0,1,11), cmap=colormap, 
        #                          norm=mpl.colors.Normalize(0, 1), extend="both",
        #                          alpha=0.95)
        # 绘制colorbar
        box = mpl.transforms.Bbox.from_extents(ax.get_position().xmax+0.005,
                                               ax.get_position().ymin,
                                               ax.get_position().xmax+0.023,
                                               ax.get_position().ymax)
        cax = self.fig.add_axes([.01,.01,.01,.01], position=box)
        cbar = self.fig.colorbar(contourset, cax=cax, extend="both",
                                 format="%.1f")
        cbar.ax.minorticks_on()
        cbar.ax.tick_params(axis="both", which="both", direction="in",labeltop=False)
        for l in cbar.ax.yaxis.get_ticklabels():
            l.set_fontsize(10)
            pass
        cax.text(1.5, 1.09, "Normalized Density", fontsize=15, 
                 fontweight="normal", transform=cax.transAxes,
                 ha="right", va="center")
        
        ax.plot([0, .6], [0, .6], c="black", alpha=0.7, ls="--", lw=2.5)
        ax.grid(ls="--", alpha=0.6)
        self._set_axes3_tick(ax)
        
        ax.set_xlabel("Data", fontsize=21, labelpad=8)
        ax.set_ylabel("Prediction", fontsize=21, labelpad=12)
        ax.tick_params(axis="both", which="both", direction="in", 
                       labelsize=15)
        ax.xaxis.get_ticklabels()[0].set_visible(False)
        ax.text(0.05, 0.9, f"r: {R_value:.3f}\np$\;\lll\;$0.05", 
                transform=ax.transAxes,
                fontsize=20, fontweight="normal",
                fontfamily="serif", ha="left", va="center",
                bbox=dict(boxstyle="round", fc="white", ec="black"))
        return
    
    def _draw_axes_4(self, ax):
        """
        绘制s4max和foEs的相关系数图
        """
        foEs = self.foEs_df.loc[:, "foEs"].to_numpy()
        foEs_s4max = self.foEs_df.loc[:, "test_s4max"].to_numpy()
        # foEs_s4max = self.foEs_df.loc[:, "s4max"].to_numpy()
        # 去掉无效值
        # sindex = (foEs_s4max<=2)
        # sindex = (foEs<=10)&(foEs>=1)
        # sindex = (foEs_s4max<=2)&(foEs<=10)
        sindex = (foEs <=999)
        foEs = foEs[sindex]
        foEs_s4max = foEs_s4max[sindex]
        
        # 计算相关系数
        # R_value, P_value = scipy.stats.pearsonr(self.s4max, self.test_s4max)
        R_value, P_value = scipy.stats.spearmanr(foEs, foEs_s4max)
        
        # 绘制散点图
        sindex = (foEs_s4max<=2)&(foEs<=10)
        ax.scatter(foEs_s4max[sindex], foEs[sindex], s=70, 
                   facecolor="#c7dbf3", alpha=0.98, marker="o",
                   edgecolors="black")
        
        # 对foEs_s4max和foEs序列线性拟合
        # 绘制拟合的直线
        fit_curve = LinearRegression().fit(foEs_s4max.reshape(-1, 1), foEs)
        print(f"{fit_curve.intercept_}+{fit_curve.coef_}*x")
        print(f"CC: {R_value}")
        xx = np.arange(0.05, 1.6, 0.1)
        yy = fit_curve.predict(xx.reshape(-1, 1)).squeeze()
        ax.plot(xx, yy, color="red", lw=2.2)
        
        self._set_axes4_tick(ax)
        
        ax.spines[["top", "right"]].set_visible(False)
        
        ax.set_xlabel("S4max Intensity, Model ", fontsize=21, labelpad=8)
        ax.set_ylabel("foEs (MHz)", fontsize=21, labelpad=5)
        
        ax.tick_params(axis="both", which="both", direction="in", 
                       labelsize=15)
        ax.xaxis.get_ticklabels()[0].set_visible(False)
        
        ax.text(0.7, 0.9, f"r: {R_value:.3f}\np$\;\lll\;$0.05", 
                transform=ax.transAxes, color="black",
                fontsize=20, fontweight="normal",
                fontfamily="serif", ha="left", va="center",
                bbox=dict(boxstyle="round", fc="white", ec="black"))
        
        pass
    
    def _set_axes1_tick(self, ax):
        """
        绘制axes1的横纵坐标刻度
        """
        xticks = np.array([-2, -1.6, -1.2, -0.8, -0.4, 0, 
                           0.4, 0.8, 1.2, 1.6])
        xticklabels = [f"{tt}" for i, tt in enumerate(xticks)]
        xlct = mpl.ticker.FixedLocator(xticks)
        xfmt = mpl.ticker.FixedFormatter(xticklabels)
        ax.xaxis.set_major_locator(xlct)
        ax.xaxis.set_major_formatter(xfmt)
        ax.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(n=2))
        ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
        ax.xaxis.set_view_interval(-2, 2, ignore=True)
        
        yticks = np.arange(0, 0.51, 0.1)
        yticklabels = [f"{tt:.0%}" for tt in yticks]
        ylct = mpl.ticker.FixedLocator(yticks)
        yfmt = mpl.ticker.FixedFormatter(yticklabels)
        ax.yaxis.set_major_locator(ylct)
        ax.yaxis.set_major_formatter(yfmt)
        ax.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(n=2))
        ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
        ax.yaxis.set_view_interval(0, 0.5, ignore=True)
        
        pass
    
    def _set_axes2_tick(self, ax):
        """
        绘制axes2的横纵坐标刻度
        """
        xticks = np.arange(0, 1.41, 0.2)
        xticklabels = [f"{tt:.1f}" for tt in xticks]
        xlct = mpl.ticker.FixedLocator(xticks)
        xfmt = mpl.ticker.FixedFormatter(xticklabels)
        ax.xaxis.set_major_locator(xlct)
        ax.xaxis.set_major_formatter(xfmt)
        ax.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(n=2))
        ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
        ax.xaxis.set_view_interval(0, 1.4, ignore=True)
        
        yticks = np.arange(-1.5, 1.51, 0.5)
        yticklabels = [f"{tt:.1f}" for tt in yticks]
        ylct = mpl.ticker.FixedLocator(yticks)
        yfmt = mpl.ticker.FixedFormatter(yticklabels)
        ax.yaxis.set_major_locator(ylct)
        ax.yaxis.set_major_formatter(yfmt)
        ax.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(n=2))
        ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
        ax.yaxis.set_view_interval(-1.5, 1.5, ignore=True)
        
        pass
    
    def _set_axes3_tick(self, ax):
        """
        绘制axes 3的横纵坐标刻度
        """
        ticks = np.arange(0, 0.41, 0.1)
        ticklabels = [f"{t:.1f}" for i, t in enumerate(ticks)]
        lct = mpl.ticker.FixedLocator(ticks)
        fmt = mpl.ticker.FixedFormatter(ticklabels)
        
        ax.xaxis.set_major_locator(lct)
        ax.xaxis.set_major_formatter(fmt)
        ax.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(n=2))
        ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
        ax.xaxis.set_view_interval(0, .4, ignore=True)
        
        ax.yaxis.set_major_locator(lct)
        ax.yaxis.set_major_formatter(fmt)
        ax.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(n=2))
        ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
        ax.yaxis.set_view_interval(0, .4, ignore=True)
        return
    
    def _set_axes4_tick(self, ax):
        """
        绘制axes 4的横纵坐标刻度
        """
        xticks = np.arange(0, 2.1, 0.4)
        xticklabels = [f"{tt:.1f}" for tt in xticks]
        xlct = mpl.ticker.FixedLocator(xticks)
        xfmt = mpl.ticker.FixedFormatter(xticklabels)
        ax.xaxis.set_major_locator(xlct)
        ax.xaxis.set_major_formatter(xfmt)
        ax.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(n=2))
        ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
        ax.xaxis.set_view_interval(0, 2, ignore=True)
        
        yticks = np.arange(1, 10, 2)
        yticklabels = [f"{tt:.0f}" for tt in yticks]
        ylct = mpl.ticker.FixedLocator(yticks)
        yfmt = mpl.ticker.FixedFormatter(yticklabels)
        ax.yaxis.set_major_locator(ylct)
        ax.yaxis.set_major_formatter(yfmt)
        ax.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(n=2))
        ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
        ax.yaxis.set_view_interval(1, 10, ignore=True)
        
    
    pass


class ApplicationLegend(_PlotBase):
    """
    应用的图例
    右边是网页截图
    左边是观测和应用对比(s4max-lct)
    """
    def __init__(self, backend: str,
                 file_path: str):
        super().__init__(backend, file_path)
        self._private_color = [(158/255, 202/255, 225/255),
                               (8/255, 48/255, 107/255),
                               ]
        pass
    
    def _load_data(self, file_path):
        file_path = Path(file_path)
        df = pd.read_hdf(file_path)
        
        alt_series = df.loc[:, "alt"]
        select_index = (alt_series >= 80) & (alt_series <= 130)
        select_df = df.loc[select_index, :]
        select_df.reset_index(drop=True, inplace=True)
        
        # use_month_series = select_df.loc[:, "time"].dt.month
        # use_time_index = (use_month_series>=6)&(use_month_series<=6)
        # # use_time_index = (use_month_series>=6)&(use_month_series<=8)
        # select_df = select_df.loc[use_time_index, :]
        
        self.alt = select_df.loc[:, "alt"].to_numpy()
        self.lct = select_df.loc[:, "lct"].to_numpy()
        self.s4max = select_df.loc[:, "s4max"].to_numpy()
        self.test_s4max = select_df.loc[:, "test_s4max"].to_numpy()
        
        lat = select_df.loc[:, "lat"].to_numpy()
        lon = select_df.loc[:, "lon"].to_numpy()
        self.dip_lat = self._glat2diplat(lat, lon, self.alt)
        pass
    
    @staticmethod
    @njit(parallel=True)
    def _glat2diplat(glat, glon, alt):
        """
        给定经纬度,高度转化为dip纬度
        """
        assert len(glat.shape)==1
        assert len(glon.shape)==1
        assert len(alt.shape)==1
        assert glat.shape==alt.shape
        
        # 把地理经纬度转化为dip纬度
        year = 2014
        with objmode(g="float64[:,:]", h="float64[:,:]"):
            g, h = igrf.ReadCoeffs().get_coeffs(year)
            pass
        dip_lat = np.zeros_like(glat)
        for i in prange(len(glat)):
            (dip, 
             gd, gi, 
             bh, bx, by, bz, bf) = igrf.igrf_value(g, h, 
                                                   glat[i], glon[i], alt[i], year)
            # dip纬度
            dip_lat[i] = dip
            pass
        return dip_lat
    
    @staticmethod
    @njit(parallel=True)
    def _compute_occurrence(coords, value, tag="s4max-lct"):
        """
        统计给定网格间隔内的均值
        划分coords的值 统计每个间隔内的均值/数目
        不同的具体参数可能有不同的间隔密度
        
        统计lct 具体算法和class OccurrenceDistribution一致
        """
        assert len(coords.shape)==1
        assert len(value.shape)==1
        if tag == "s4max-lct":
            grid_coords = np.arange(0.5, 24, 1)
            pass
        elif tag == "s4max-dip_lat":
            grid_coords = np.arange(-88.,90,4)
            pass
        else:
            pass
        
        delta = grid_coords[1] - grid_coords[0]
        
        # 总数目
        N = np.zeros_like(grid_coords)
        Z = np.zeros_like(grid_coords)
        
        for i in prange(Z.shape[0]):
            for di in prange(len(value)):
                coordsi = coords[di]
                valuei = value[di]
                grid_coordsi = grid_coords[i]
                
                if (coordsi>=grid_coordsi-(delta/2))&(coordsi<grid_coordsi+(delta/2)):
                    N[i] += 1
                    Z[i] += valuei
                    pass
                pass
            pass
        return (grid_coords, Z, N)
    
    def plot(self,):
        figsize = (9, 6)
        dpi = 200
        nrows = 2
        ncols = 2
        axes = []
        
        self.fig = plt.figure(figsize=figsize, dpi=dpi, 
                              facecolor="white")
        gs = self.fig.add_gridspec(nrows=nrows, ncols=ncols, 
                                   left=0.01, right=0.98, bottom=0.09, top=0.95,
                                   wspace=0.26, hspace=0.2)
        
        axes.append(self.fig.add_subplot(gs[:, 0]))
        for i in range(nrows):
            axes.append(self.fig.add_subplot(gs[i, 1]))
            pass
        
        
        self._draw_axes_0(axes[0])
        self._draw_axes_1(axes[1])
        self._draw_axes_2(axes[2])
        
        self._save_fig()
        return
    
    def _tune_axes(self, ax, rx, ry, tw, th):
        """
        调整axes的大小和位置
        rx: 相对于原xmin的位移
        ry: ...ymin
        tw, th: 宽和高的增量
        """
        bbox1 = ax.get_position(original=True)
        x0 = bbox1.xmin
        y0 = bbox1.ymin
        w0 = bbox1.width
        h0 = bbox1.height
        
        x1 = x0 + rx
        y1 = y0 + ry
        w1 = w0 + tw
        h1 = h0 + th
        bbox2 = mpl.transforms.Bbox.from_bounds(x1, y1, w1, h1)
        ax.set_position(bbox2)
        return
    
    def _set_lat_tick(self, ax):
        """
        设置纬度刻度
        """
        lat = np.linspace(-90, 90, num=13, dtype=np.int32)
        latstr = []
        for i, j in enumerate(lat):
            if i%2==0:
                jj = str(j) + "$^{\circ}$"
                latstr.append(f"{jj}")
                pass
            else:
                latstr.append("")
                pass
            pass
        lct = mpl.ticker.FixedLocator(lat)
        fmt = mpl.ticker.FixedFormatter(latstr)

        ax.xaxis.set_view_interval(-90, 90, ignore=True)
        ax.xaxis.set_major_locator(lct)
        ax.xaxis.set_major_formatter(fmt)
        ax.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(n=2))
        ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
        pass
    
    def _set_lct_tick(self, ax):
        """
        设定当地时刻度
        """
        lct = np.linspace(0, 24, num=5, dtype=np.int32)
        lctstr = []
        for i, j in enumerate(lct):
            lctstr.append(f"{j}")
            pass
        lct = mpl.ticker.FixedLocator(lct)
        fmt = mpl.ticker.FixedFormatter(lctstr)
        
        ax.xaxis.set_view_interval(0, 24, ignore=True)
        ax.xaxis.set_major_locator(lct)
        ax.xaxis.set_major_formatter(fmt)
        ax.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(n=2))
        ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
        
        pass
    
    def _set_s4max_tick(self, ax):
        """
        设定S4max强度刻度
        """
        value = np.arange(0, 0.71, 0.1)
        vstr = [f"{v:.1f}" for v in value]
        
        lct = mpl.ticker.FixedLocator(value)
        fmt = mpl.ticker.FixedFormatter(vstr)
        
        ax.yaxis.set_view_interval(0, .7, ignore=True)
        ax.yaxis.set_major_locator(lct)
        ax.yaxis.set_major_formatter(fmt)
        ax.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(n=4))
        ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
        
        pass
    
    def _draw_axes_0(self, ax):
        """
        第一个axes
        绘制网页app截图
        """
        import PIL
        from PIL.Image import Resampling
        path = Path("./web_app.png")
        im = PIL.Image.open(path)
        
        ax.imshow(im, aspect="equal", interpolation="lanczos")
        
        ax.tick_params(axis="both", which="both", 
                       bottom=False, left=False,
                       labelbottom=False, labelleft=False)
        
        # tune the position of axes
        self._tune_axes(ax, -0.04, -0.05, 0.14, 0.07)
        ax.spines[:].set_visible(True)
        ax.spines[:].set_linewidth(1.25)
        ax.text(-0.065, 1, "(a)", transform=ax.transAxes,
                fontsize=15, fontweight="bold",
                va="center", ha="center")
        
        # bbox1 = ax.get_position(original=True)
        # rec_patch = mpl.patches.Rectangle(xy=(bbox1.xmin,bbox1.ymin),
        #                                   width=bbox1.width, height=bbox1.height,
        #                                   facecolor="none", lw=1.1,
        #                                   edgecolor="k", alpha=.7,
        #                                   transform=ax.transAxes)
        # ax.add_artist(rec_patch)
        pass
    
    def _draw_axes_1(self, ax):
        """
        第二个axes
        绘制观测和模型的s4max-lct均值变化
        """
        # 绘制s4max均值随当地时变化
        (lct_grid_coords, Z_1, N_1) = self._compute_occurrence(self.lct, 
                                                               self.s4max, 
                                                               tag="s4max-lct")
        (lct_grid_coords, Z_2, N_2) = self._compute_occurrence(self.lct, 
                                                               self.test_s4max, 
                                                               tag="s4max-lct")
        
        # line1为data f16c23
        # line2为model 2b6a99
        line1, = ax.plot(lct_grid_coords, Z_1/N_1, c="#c1c1c1", 
                         drawstyle="default", 
                         linestyle="-",
                         marker="o", markersize=9)
        line2, = ax.plot(lct_grid_coords, Z_2/N_2, c="#d32327", 
                         drawstyle="default", 
                         ls="-",marker="d",
                         fillstyle="top", markerfacecoloralt="white", markersize=9)
        ax.set_ylabel("S4max Intensity", fontsize=15, loc="center",)
        
        # 设定legend
        circle1 = mpl.lines.Line2D([],[], marker="o", ls="-", color="#c1c1c1")
        circle2 = mpl.lines.Line2D([],[], marker="d", ls="-", color="#d32327",
                                   fillstyle="top", markerfacecoloralt="white")
        point_legend = ax.legend([circle1, circle2], ["Data", "Prediction"],
                                 loc="upper left", edgecolor="black",
                                 markerscale=1.5, fontsize=13)
        ax.add_artist(point_legend)
        
        # 设定tick
        ax.tick_params(axis="both", which="both", direction="in", labelsize=12)
        ax.tick_params(axis="both", which="major", length=3, width=1.3)
        ax.tick_params(axis="both", which="minor", length=2, width=1.)
        ax.spines[:].set_linewidth(1.3)
        ax.set_xlabel("Local Time", fontsize=15)
        ax.text(-0.135, 1, "(b)", transform=ax.transAxes,
                fontsize=15, fontweight="bold",
                va="center", ha="center")
        ax.grid(ls="--", which="both", axis="x", lw=1.5)
        ax.grid(ls="--", which="major", axis="y", lw=1.5)
        # 调整axes位置
        self._tune_axes(ax, 0, 0.02, 0, 0)
        self._set_s4max_tick(ax)
        self._set_lct_tick(ax)
        pass
    
    def _draw_axes_2(self, ax):
        """
        第三个axes
        绘制观测和模型的s4max-dip_lat均值变化
        """
        # 绘制s4max均值随磁纬变化
        (dip_lat_grid_coords, Z_1, N_1) = self._compute_occurrence(self.dip_lat, 
                                                                   self.s4max, 
                                                                   tag="s4max-dip_lat")
        (dip_lat_grid_coords, Z_2, N_2) = self._compute_occurrence(self.dip_lat, 
                                                                   self.test_s4max, 
                                                                  tag="s4max-dip_lat")
        
        # line1为data
        # line2为model
        line1, = ax.plot(dip_lat_grid_coords, Z_1/N_1, c="#c1c1c1", 
                         drawstyle="default", marker="o", markersize=9)
        line2, = ax.plot(dip_lat_grid_coords, Z_2/N_2, c="#d32327", 
                         drawstyle="default", marker="d",
                         fillstyle="top", markerfacecoloralt="white", markersize=9)
        ax.set_ylabel("S4max Intensity", fontsize=15, loc="center",)
        circle1 = mpl.lines.Line2D([],[], marker="o", ls="-", color="#c1c1c1")
        circle2 = mpl.lines.Line2D([],[], marker="d", ls="-", color="#d32327",
                                   fillstyle="top", markerfacecoloralt="white")
        
        # legend
        point_legend = ax.legend([circle1, circle2], ["Data", "Prediction"],
                                 loc="upper left", edgecolor="black",
                                 markerscale=1.5, fontsize=13)
        ax.add_artist(point_legend)
        ax.tick_params(axis="both", which="both", direction="in", labelsize=12)
        ax.tick_params(axis="both", which="major", length=3, width=1.3)
        ax.tick_params(axis="both", which="minor", length=2, width=1.)
        ax.spines[:].set_linewidth(1.3)
        
        ax.set_xlabel("Geomagnetic Latitude", fontsize=15)
        ax.text(-0.135, 1, "(c)", transform=ax.transAxes,
                fontsize=15, fontweight="bold",
                va="center", ha="center")
        ax.grid(ls="--", which="major", axis="both", lw=1.5)
        self._set_s4max_tick(ax)
        self._set_lat_tick(ax)
        
        pass
    
    
    pass


class StationEvaluation(_PlotBase):
    """
    给出某个台站foEs图
    """
    def __init__(self, backend: str,
                 file_path: str,
                 foEs_infer_file):
        super().__init__(backend, file_path)
        self.foEs_df = pd.read_hdf(foEs_infer_file)
        pass
    
    def _load_data(self, file_path):
        
        # file_path = Path(file_path)
        # df = pd.read_hdf(file_path)
        # alt_series = df.loc[:, "alt"]
        # select_index = (alt_series >= 80) & (alt_series <= 130)
        # select_df = df.loc[select_index, :]
        # self.alt = select_df.loc[:, "alt"].to_numpy()
        # self.lct = select_df.loc[:, "lct"].to_numpy()
        # self.s4max = select_df.loc[:, "s4max"].to_numpy()
        # # self.s4max = select_df.loc[:, "target_s4max"].to_numpy()
        # self.test_s4max = select_df.loc[:, "test_s4max"].to_numpy()
        # self.lat = select_df.loc[:, "lat"].to_numpy()
        # self.lon = select_df.loc[:, "lon"].to_numpy()
        pass
    
    def plot(self):
        figsize = (12, 12)
        dpi = 200
        nrows = 1
        ncols = 1
        axes = []
        
        self.fig = plt.figure(figsize=figsize, dpi=dpi, 
                              facecolor="white")
        gs = self.fig.add_gridspec(nrows=nrows, ncols=ncols, 
                                   left=0.1, right=0.9, bottom=0.1, top=0.9,
                                   wspace=0.25, hspace=0.25)
        for i in range(nrows):
            for j in range(ncols):
                axes.append(self.fig.add_subplot(gs[i, j]))
                pass
            pass
        
        
        self._draw_axes_4(axes[0])
        
        self._save_fig()
        return
    
    
    def _draw_axes_4(self, ax):
        """
        绘制s4max和foEs的相关系数图
        """
        foEs = self.foEs_df.loc[:, "foEs"].to_numpy()
        foEs_s4max = self.foEs_df.loc[:, "test_s4max"].to_numpy()
        
        # 去掉无效值
        sindex = (foEs<9999)
        foEs = foEs[sindex]
        foEs_s4max = foEs_s4max[sindex]
        
        # 计算相关系数
        # R_value, P_value = scipy.stats.pearsonr(self.s4max, self.test_s4max)
        R_value, P_value = scipy.stats.spearmanr(foEs, foEs_s4max)
        
        # 绘制散点图
        ax.set_box_aspect(1)
        ax.scatter(foEs_s4max, foEs, s=120, 
                   facecolor="#c7dbf3", alpha=0.98, marker="o",
                   edgecolors="black")
        
        # 对foEs_s4max和foEs序列线性拟合
        # 绘制拟合的直线
        fit_curve = LinearRegression().fit(foEs_s4max.reshape(-1, 1), foEs)
        print(f"{fit_curve.intercept_}+{fit_curve.coef_}*x")
        xx = np.arange(0.05, 1.6, 0.1)
        yy = fit_curve.predict(xx.reshape(-1, 1)).squeeze()
        ax.plot(xx, yy, color="red", lw=2.2)
        
        self._set_axes4_tick(ax)
        
        ax.spines[["top", "right"]].set_visible(False)
        
        ax.set_xlabel("S4max Intensity, Model ", fontsize=21, labelpad=12)
        ax.set_ylabel("foEs (MHz)", fontsize=21, labelpad=12)
        
        ax.tick_params(axis="both", which="both", direction="in", 
                       labelsize=18)
        ax.xaxis.get_ticklabels()[0].set_visible(False)
        
        ax.text(0.7, 0.9, f"r: {R_value:.3f}\np$\;\lll\;$0.05", 
                transform=ax.transAxes, color="black",
                fontsize=28, fontweight="normal",
                fontfamily="serif", ha="left", va="center",
                bbox=dict(boxstyle="round", fc="white", ec="black"))
        
        pass
    
    def _set_axes4_tick(self, ax):
        """
        绘制axes 4的横纵坐标刻度
        """
        xticks = np.arange(0, 2.1, 0.4)
        xticklabels = [f"{tt:.1f}" for tt in xticks]
        xlct = mpl.ticker.FixedLocator(xticks)
        xfmt = mpl.ticker.FixedFormatter(xticklabels)
        ax.xaxis.set_major_locator(xlct)
        ax.xaxis.set_major_formatter(xfmt)
        ax.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(n=2))
        ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
        ax.xaxis.set_view_interval(0, 2, ignore=True)
        
        yticks = np.arange(1, 10, 2)
        yticklabels = [f"{tt:.0f}" for tt in yticks]
        ylct = mpl.ticker.FixedLocator(yticks)
        yfmt = mpl.ticker.FixedFormatter(yticklabels)
        ax.yaxis.set_major_locator(ylct)
        ax.yaxis.set_major_formatter(yfmt)
        ax.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(n=2))
        ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
        ax.yaxis.set_view_interval(1, 10, ignore=True)
        pass
    pass
    

class DifferentYearDistribution(_PlotBase):
    """
    绘制SELF-ANN 简单模型的不同时间范围的预测性质
    通过改变年份得到不同推断值 绘制推断值相应的参数
    """
    def __init__(self,
                 backend: str,
                 file_path: str):
        super().__init__(backend, file_path)
        pass
    
    def _load_data(self, file_path):
        df = pd.read_hdf("./test1.h5")
        indexx = df.index.tolist()
        y_list = []
        for yy in indexx:
            y_list.append(int(yy.split("_")[2]))
            pass
        df.loc[:, "start_year"] = y_list
        self.df = df
        pass
    
    def _set_cc_tick(self, ax):
        """
        相关系数的tick
        """
        cc = np.arange(0.2, 0.9, 0.1)
        ccstr = []
        for i, j in enumerate(cc):
            ccstr.append(f"{j:.1f}")
            pass
        lct = mpl.ticker.FixedLocator(cc)
        fmt = mpl.ticker.FixedFormatter(ccstr)
        
        ax.set_ylim(-0.05, 1)
        ax.yaxis.set_major_locator(lct)
        ax.yaxis.set_major_formatter(fmt)
        
        
        # 以下代码为参考 AutoMinorLocator source code 后编写
        # 固定locs在需要的位置
        # 在更改坐标轴位置时 tick不会出现预期之外的情况
        t0 = cc[0]
        minorstep = (cc[1] - cc[0]) / 3
        vmin = cc[0]
        vmax = cc[-1]
        tmin = ((vmin - t0) // minorstep + 1) * minorstep
        tmax = ((vmax - t0) // minorstep + 1) * minorstep
        minor_locs = np.arange(tmin, tmax, minorstep) + t0
        minor_locs = minor_locs[(minor_locs>=cc[0])&(minor_locs<=cc[-1])]
        
        ax.yaxis.set_minor_locator(mpl.ticker.FixedLocator(minor_locs))
        ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
        
        pass
    
    def _set_mae_tick(self, ax):
        """
        MAE的tick
        """
        cc = np.arange(0., 1.1, 0.2)
        ccstr = []
        for i, j in enumerate(cc):
            ccstr.append(f"{j:.1f}")
            pass
        lct = mpl.ticker.FixedLocator(cc)
        fmt = mpl.ticker.FixedFormatter(ccstr)
        
        ax.set_ylim(-1, 2)
        ax.yaxis.set_major_locator(lct)
        ax.yaxis.set_major_formatter(fmt)
        
        
        # 以下代码为参考 AutoMinorLocator source code 后编写
        # 固定locs在需要的位置
        # 在更改坐标轴位置时 tick不会出现预期之外的情况
        t0 = cc[0]
        minorstep = (cc[1] - cc[0]) / 3
        vmin = cc[0]
        vmax = cc[-1]
        tmin = ((vmin - t0) // minorstep + 1) * minorstep
        tmax = ((vmax - t0) // minorstep + 1) * minorstep
        minor_locs = np.arange(tmin, tmax, minorstep) + t0
        minor_locs = minor_locs[(minor_locs>=cc[0])&(minor_locs<=cc[-1])]
        
        ax.yaxis.set_minor_locator(mpl.ticker.FixedLocator(minor_locs))
        ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
        
        pass
    
    def _set_rmse_tick(self, ax):
        """
        相关系数的tick
        """
        cc = np.arange(0., 1.1, 0.2)
        ccstr = []
        for i, j in enumerate(cc):
            ccstr.append(f"{j:.1f}")
            pass
        lct = mpl.ticker.FixedLocator(cc)
        fmt = mpl.ticker.FixedFormatter(ccstr)
        
        ax.set_ylim(-0.5, 3)
        ax.yaxis.set_major_locator(lct)
        ax.yaxis.set_major_formatter(fmt)
        
        
        # 以下代码为参考 AutoMinorLocator source code 后编写
        # 固定locs在需要的位置
        # 在更改坐标轴位置时 tick不会出现预期之外的情况
        t0 = cc[0]
        minorstep = (cc[1] - cc[0]) / 3
        vmin = cc[0]
        vmax = cc[-1]
        tmin = ((vmin - t0) // minorstep + 1) * minorstep
        tmax = ((vmax - t0) // minorstep + 1) * minorstep
        minor_locs = np.arange(tmin, tmax, minorstep) + t0
        minor_locs = minor_locs[(minor_locs>=cc[0])&(minor_locs<=cc[-1])]
        
        ax.yaxis.set_minor_locator(mpl.ticker.FixedLocator(minor_locs))
        ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
        
        pass
    
    def _draw_axes(self, ax):
        
        year_index = self.df.loc[:, "start_year"].to_numpy()
        spearR = self.df.loc[:, "spearR"].to_numpy()
        pred_std = self.df.loc[:, "pred_std"].to_numpy()
        true_std = self.df.loc[:, "true_std"].to_numpy()
        
        # 绘制子图的序号
        ax.text(-0.09, 1, "(d)", transform=ax.transAxes, fontsize=15,
                fontweight="bold", va="center", ha="center")
        
        ax.grid(axis="x", linestyle="--", dashes=(5, 5))
        
        # 绘制spearR的线条
        ax.hlines(spearR, year_index, year_index+11, color="#2572a9", linewidth=0.8,
                  linestyle="-", alpha=.8)
        
        # 绘制阈值线条
        # 注意因为 axes的左边spine向外移动
        # 然而axhline只能在axes画布内部作用
        # 因此创建一个Line2D 实例 在figure中添加进去
        # ax.axhline(y=0.4, xmin=-0.5, xmax=0.91, linestyle=":", linewidth=1.8, color="#2572a9")
        line_ther = mpl.lines.Line2D(xdata=[1983.7, 2033], ydata=[0.4, 0.4], linewidth=1.8,
                                     color="#2572a9", linestyle=":", transform=ax.transData)
        self.fig.add_artist(line_ther)
        
        # 绘制真实值和预测值的std误差范围
        ax.fill_between(x=year_index+5.5, y1=spearR-true_std/10, y2=spearR+true_std/10,
                        alpha=.9, color="#f4f3dd")
        
        ax.fill_between(x=year_index+5.5, y1=spearR-pred_std/10, y2=spearR+pred_std/10,
                        alpha=.5, color="#ede0ef")
        
        
        
        ax.set_xlim(year_index[0], year_index[-1]+11)
        
        ax.set_ylabel("Correlation Coefficient", loc="center", color="#2572a9", fontweight="bold")
        ax.set_xlabel("Year", loc="center", labelpad=0.5, fontsize=12, fontweight="bold")
        
        # ax.set_yticks(np.arange(0.2, 0.9, 0.1))
        # ax.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(n=2))
        # ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
        ax.set_xticks(np.arange(year_index[0], year_index[-1]+12, 3))
        self._set_cc_tick(ax)
        
        # 设定整体tick参数和spines性质
        ax.tick_params(which="both", direction="in", labelleft=True, labeltop=True, top=True,
                       )
        ax.spines[[ "right"]].set_visible(False)
        
        # 设定Y轴参数
        ax.tick_params(which="major", axis="y", length=5, width=1.3, labelsize=12, 
                       color="#2572a9", labelcolor="#2572a9")
        ax.tick_params(which="minor", axis="y", length=2.5, width=1, labelsize=12, 
                       color="#2572a9", labelcolor="#2572a9")
        ax.spines["left"].set_bounds(0.2, 0.8)
        ax.spines.left.set_color("#2572a9")
        ax.spines.left.set_linewidth(1.3)
        ax.spines.left.set_position(("data", 1983.7))
        
        # 设定X轴参数
        ax.tick_params(which="major", axis="x", length=5, width=1.3, labelsize=12)
        ax.spines[["top", "bottom"]].set_linewidth(1.3)
        
        
        
        
        # 绘制有效时间范围的阴影
        # ax.axvspan(xmin=2007, xmax=2018, fc="#c2e9fe", ec="none", alpha=.5)
        ax.axvspan(xmin=2002, xmax=2025, fc="#daeaf5", ec="none", alpha=.4)
        
        # 绘制箭头指示时间范围
        arrow1 = mpl.patches.FancyArrowPatch(posA=(2007,0.07), posB=(2018,0.07), arrowstyle="<|-|>",
                                            shrinkA=0, shrinkB=0, color="#1f8fb6",
                                            transform=ax.transData, mutation_scale=10)
        arrow2 = mpl.patches.FancyArrowPatch(posA=(2002,0.001), posB=(2025,0.001), arrowstyle="<|-|>",
                                            shrinkA=0, shrinkB=0, color="#a3132f",
                                            transform=ax.transData, mutation_scale=13)
        arrow3 = mpl.patches.FancyArrowPatch(posA=(2002,0.95), posB=(2025,0.95), arrowstyle="<|-|>",
                                            shrinkA=0, shrinkB=0, color="#a3132f",
                                            transform=ax.transData, mutation_scale=13)
        # 设定训练时间和有效预测时间的文字
        ax.text(x=2011, y=0.08, s="2007~2018", fontweight="bold", fontsize=12, transform=ax.transData, color="#1f8fb6")
        ax.text(x=2011, y=0.01, s="2002~2025", fontweight="bold", fontsize=12, transform=ax.transData, color="#a3132f")
        ax.add_artist(arrow1)
        ax.add_artist(arrow2)
        ax.add_artist(arrow3)
        
        
        # 绘制legend
        artist1 = mpl.lines.Line2D([],[], color="#b2112e", marker="o", markerfacecolor="white")
        artist2 = mpl.lines.Line2D([],[], color="#7b7625", marker="o", markerfacecolor="white")
        artist3 = mpl.lines.Line2D([],[], color="#2572a9", linewidth=0.8, linestyle="-", alpha=.6)
        artist4 = mpl.patches.Patch(alpha=.9, facecolor="#f4f3dd", edgecolor="none")
        artist5 = mpl.patches.Patch(alpha=.9, facecolor="#ede0ef", edgecolor="none")
        
        # 自定义箭头handler 绘制legend
        # 具体见 https://matplotlib.org/stable/tutorials/intermediate/legend_guide.html
        class AnyObjectHandler1():
            def legend_artist(self, legend, orig_handle, fontsize, handlebox):
                x0, y0 = handlebox.xdescent, handlebox.ydescent
                width, height = handlebox.width, handlebox.height
                patch1 = mpl.patches.FancyArrowPatch(posA=(x0,y0+height/2), posB=(x0+width,y0+height/2), 
                                                    shrinkA=0, shrinkB=0, color="#1f8fb6",
                                                    arrowstyle="<|-|>",mutation_scale=8,
                                                    transform=handlebox.get_transform())
                handlebox.add_artist(patch1)
                return 
            pass
        
        class AnyObjectHandler2():
            def legend_artist(self, legend, orig_handle, fontsize, handlebox):
                x0, y0 = handlebox.xdescent, handlebox.ydescent
                width, height = handlebox.width, handlebox.height
                patch1 = mpl.patches.FancyArrowPatch(posA=(x0,y0+height/2), posB=(x0+width,y0+height/2), 
                                                    shrinkA=0, shrinkB=0, color="#a3132f",
                                                    arrowstyle="<|-|>",mutation_scale=8,
                                                    transform=handlebox.get_transform())
                handlebox.add_artist(patch1)
                return 
        # artist6 = mpl.patches.Arrow(x=0, y=0, dx=0.1, dy=0, facecolor="#ede0ef", edgecolor="none")
        # artist7 = arrow2
        point_legend = ax.legend([artist1, artist2, artist3, artist4, artist5, arrow1, arrow2], 
                                 ["Mean absolute error", "Root mean square error", "Correlation coefficient",
                                  "Observation standard variation", "SELF-ANN standard variation",
                                  "Scope of training data", "Scope of valid prediction"],
                                 loc="center", bbox_to_anchor=(0.08, 0.7, 0.01, 0.01),
                                 edgecolor="black", fontsize=10,
                                 handler_map={artist1: mpl.legend_handler.HandlerLine2D(numpoints=2),
                                              arrow1: AnyObjectHandler1(), arrow2: AnyObjectHandler2()})
                                 # markerscale=, 2)
        ax.add_artist(point_legend)
        
        return
    
    def _draw_axes1(self, axposition):
        year_index = self.df.loc[:, "start_year"].to_numpy()
        # 创建MAE的坐标轴并绘制
        rec = (axposition.xmin, axposition.ymin+0.28,
               axposition.width, axposition.height)
        
        ax = self.fig.add_axes(rec, facecolor="none")
        ax.set_xlim(year_index[0], year_index[-1]+11)
        
        self._set_mae_tick(ax)
        ax.tick_params(which="both", direction="in", labelright=True, bottom=False, labelbottom=False,
                        right=True, left=False, labelleft=False)
        # 设定Y轴参数
        ax.tick_params(which="major", axis="y", length=5, width=1.3, labelsize=12, 
                       color="#b2112e", labelcolor="#b2112e")
        ax.tick_params(which="minor", axis="y", length=2.5, width=1, labelsize=12, 
                       color="#b2112e", labelcolor="#b2112e")
        ax.spines[["top", "left", "bottom"]].set_visible(False)
        ax.spines["right"].set_bounds(0, 1.)
        ax.spines.right.set_color("#b2112e")
        ax.spines.right.set_linewidth(1.3)
        
        # 设定Y轴标签
        ax.text(0.98, 0.29, "Mean Abosult Error", color="#b2112e", fontweight="bold", fontsize=10,
                transform=ax.transAxes, ha="center")
        
        start_year = self.df.loc[:, "start_year"].to_numpy()
        mae = self.df.loc[:, "mae"].to_numpy()
        sindex = (mae>=0)&(mae<=1)
        start_year = start_year[sindex]
        mae = mae[sindex]
        # line1, = ax.plot(start_year+5.5, mae, color="#b2112e", marker="d",
                         # fillstyle="top", markerfacecoloralt="white")
        line1, = ax.plot(start_year+5.5, mae, color="#b2112e", marker="o",
                          markerfacecolor="white")
        
        # 阈值线
        line_ther = mpl.lines.Line2D(xdata=[1992, 2038], ydata=[0.8, 0.8], linewidth=1.8,
                                     color="#b2112e", linestyle=":", transform=ax.transData)
        ax.add_artist(line_ther)
        # ax.axhline(y=0.8, xmin=0.05, xmax=1, linestyle=":", linewidth=1.8,
        #            color="#b2112e")
        # line1.set_xdata(np.arange(2007,2019))
        
        
        return
    
    def _draw_axes2(self, axposition):
        year_index = self.df.loc[:, "start_year"].to_numpy()
        # 创建RMSE的坐标轴并绘制
        rec = (axposition.xmin, axposition.ymin,
               axposition.width, axposition.height)
        ax = self.fig.add_axes(rec, facecolor="none")
        ax.set_xlim(year_index[0], year_index[-1]+11)
        self._set_rmse_tick(ax)
        ax.tick_params(which="both", direction="in", labelright=True, bottom=False, labelbottom=False,
                        right=True, left=False, labelleft=False)
        
        # Y轴
        ax.tick_params(which="major", axis="y", length=5, width=1.3, labelsize=12, 
                       color="#7b7625", labelcolor="#7b7625")
        ax.tick_params(which="minor", axis="y", length=2.5, width=1, labelsize=12, 
                       color="#7b7625", labelcolor="#7b7625")
        ax.spines[["top", "left", "bottom"]].set_visible(False)
        ax.spines["right"].set_bounds(0, 1.)
        ax.spines.right.set_color("#7b7625")
        ax.spines.right.set_linewidth(1.3)
        
        # Y轴标签
        ax.text(2036, 1.1, "Root Mean Square Error", color="#7b7625", fontweight="bold", fontsize=10,
                transform=ax.transData, ha="center")
        
        start_year = self.df.loc[:, "start_year"].to_numpy()
        rmse = self.df.loc[:, "rmse"].to_numpy()
        sindex = (rmse>=0)&(rmse<=3)
        start_year = start_year[sindex]
        rmse = rmse[sindex]
        # ax.plot(start_year+5.5, rmse, color="#7b7625", marker="d",
        #         fillstyle="top", markerfacecoloralt="white")
        line1, = ax.plot(start_year+5.5, rmse, color="#7b7625", marker="o",
                          markerfacecolor="white")
        
        # 阈值线
        line_ther = mpl.lines.Line2D(xdata=[1992, 2038], ydata=[0.8, 0.8], linewidth=1.8,
                                     color="#7b7625", linestyle=":", transform=ax.transData)
        ax.add_artist(line_ther)
    
    def plot(self,):
        figsize = (12, 6)
        dpi = 200
        nrows = 1
        ncols = 1
        
        self.fig = plt.figure(figsize=figsize, dpi=dpi, 
                              facecolor="white")
        gs = self.fig.add_gridspec(nrows=nrows, ncols=ncols, 
                                   left=0.1, right=0.95, bottom=0.06, top=0.96)
        
        axes = self.fig.add_subplot(gs[0, 0])
        
        # 绘制相关系数图
        self._draw_axes(axes)
        self._draw_axes1(axes.get_position())
        self._draw_axes2(axes.get_position())
        # self._draw_axes_0(axes[0])
        # self._draw_axes_1(axes[1])
        # self._draw_axes_2(axes[2])
        
        # 调整轴刻度细节
        # for ax in axes:
        #     ax.tick_params(axis="both", which="both", direction="in")
        #     ax.yaxis.get_ticklabels()[0].set_visible(False)
        self._save_fig()
        return
    
    pass


class ConcatTwoFigure(_PlotBase):
    """
    合并多个子图
    合并ApplicationLegend和DifferentYearDistribution
    
    定义两个类继承相应的父类
    重写plot方法 不再内部创建fig而是由外界传入
    在外界定义一个大fig 分割成两个subfig分别传入即可
    
    note: 分割过程中可以用gridspec分成上百份 可以为子图留下空间
    因为fig分割subfig过程中不会留下空间 而是填满的
    只有实例化axes才会留有空间
    """
    
    class FIG1(ApplicationLegend):
    
        def plot(self, fig):
            nrows = 2
            ncols = 2
            axes = []
            
            self.fig = fig
            gs = self.fig.add_gridspec(nrows=nrows, ncols=ncols, 
                                       left=0.01, right=0.98, bottom=0.09, top=0.95,
                                       wspace=0.26, hspace=0.2)

            axes.append(self.fig.add_subplot(gs[:, 0]))
            for i in range(nrows):
                axes.append(self.fig.add_subplot(gs[i, 1]))
                pass


            self._draw_axes_0(axes[0])
            self._draw_axes_1(axes[1])
            self._draw_axes_2(axes[2])

            # self._save_fig()
            pass
        pass

    class FIG2(DifferentYearDistribution):

        def plot(self, fig):
            nrows = 1
            ncols = 1

            self.fig = fig
            gs = self.fig.add_gridspec(nrows=nrows, ncols=ncols, 
                                       left=0.1, right=0.95, bottom=0.05, top=0.95)

            axes = self.fig.add_subplot(gs[0, 0])

            # 绘制相关系数图
            self._draw_axes(axes)
            self._draw_axes1(axes.get_position())
            self._draw_axes2(axes.get_position())
            return
        pass
    
    def __init__(self,
                 backend: str,
                 file_path1: str,
                 file_path2: str):
        super().__init__(backend, file_path1)
        # 实例化两个对象
        self.obj1 = ConcatTwoFigure.FIG1(backend=backend, file_path=file_path1)
        self.obj2 = ConcatTwoFigure.FIG2(backend=backend, file_path=file_path2)
        pass
    
    def _load_data(self, file_path):
        pass
    
    def plot(self):
        figsize = (11.9, 14)
        dpi = 200
        nrows = 141
        ncols = 1

        self.fig = plt.figure(figsize=figsize, dpi=dpi, facecolor="white")

        # 这里将fig分成141个 前80个为应用的图 中间60个为不同年份的预测性能图 最后一个是为了美观留白
        gs = self.fig.add_gridspec(nrows=nrows, ncols=ncols)
        subfig1 = self.fig.add_subfigure(gs[:80, 0], facecolor="white")
        subfig2 = self.fig.add_subfigure(gs[80:140, 0], facecolor="white")
        
        # 此时fig分成subfig1 subfig2
        # 不同obj在不同subfig上自行绘制
        # 注意subfig.transFigure实际上是fig.transFigure
        # 因此减少transFigure的使用
        self.obj1.plot(subfig1)
        self.obj2.plot(subfig2)
        
        self._save_fig()
    
    pass


class DensityDistribution5(_PlotBase):
    """
    绘制lat-lon
    各个季节的全球统计分布
    """
    def __init__(self,
                 backend: str,
                 file_path: str):
        super().__init__(backend, file_path)
        pass
    
    def _load_data(self, file_path):
        # 自定义时间范围
        file_path = Path(file_path)
        df = pd.read_hdf(file_path)
        
        s4max_str = "test_s4max" if "inference" in file_path.stem else "s4max"
        
        time_series = df.loc[:, "time"]
        time_index = (time_series >= pd.Timestamp("2013-6-1")) & (time_series < pd.Timestamp("2014-6-1"))
        use_df = df.loc[time_index, ["time", "alt", "lat", "lon", "lct", s4max_str]]
        use_df.reset_index(drop=True, inplace=True)
        
        alt_series = df.loc[:, "alt"]
        alt_index = (alt_series >= 80) & (alt_series <= 130)
        use_df = df.loc[alt_index, :]
        use_df.reset_index(drop=True, inplace=True)
        
        # 给出不同的季节
        # spring: MAM(3,4,5)
        # summer: JJA(6,7,8)
        # autumn: SON(9,10,11)
        # winter: DJF(12,1,2)
        use_month_series = use_df.loc[:, "time"].dt.month
        # use_time_index = (use_month_series>=3)&(use_month_series<=5) # spring
        # use_time_index = (use_month_series>=6)&(use_month_series<=8) # summer
        # use_time_index = (use_month_series>=9)&(use_month_series<=11) # autumn
        use_time_index = ((use_month_series>=1)&(use_month_series<=2))|(use_month_series==12) # winter
        
        try:
            data = use_df.loc[use_time_index, 
                              ["alt", "lat", "lon", "lct", s4max_str]].values.astype(np.float64)
            pass
        except:
            data = use_df.loc[:, 
                              ["alt", "lat", "lon", "lct", s4max_str]].values.astype(np.float64)
            
        self.alt = data[:, 0]
        self.lat = data[:, 1]
        self.lon = data[:, 2]
        self.lct = data[:, 3]
        self.s4max = data[:, 4]
        
        self.doy = use_df.loc[:, "time"].dt.day_of_year.values
    
    def _draw_axes_0(self, ax):
        # lat-lon
        grid_lon, grid_lat, grid_count, grid_value = compute_count(self.lon, self.lat, 
                                                                   self.s4max,
                                                                   tag="lat-lon")
        grid_count[grid_count==0] = 1
        ax.add_feature(cartopy.feature.COASTLINE)
        ax.set_global()
        # contourset = ax.pcolormesh(grid_doy, grid_lat, grid_value/grid_count,
        #                            cmap="rainbow", norm=mpl.colors.Normalize(0.2,0.6))
        contourset = ax.contourf(grid_lon, grid_lat, grid_value/grid_count, 
                                 levels=50, cmap="rainbow", 
                                 norm=mpl.colors.Normalize(0.,0.9),
                                 transform=cartopy.crs.PlateCarree())
        
        # self.fig.colorbar(contourset, )
        # ax.set_xlim(-180,180)
        # ax.set_ylim(-90,90)
        # ax.set_xticks([])
        # ax.set_ylabel("Latitude", loc="center")
        # ax.set_xlabel("Longitude", loc="center")
        
        return
    
    def plot(self):
        figsize = (10,8)
        dpi = 200
        nrows = 1
        ncols = 1
        axes = []
        
        self.fig = plt.figure(figsize=figsize, dpi=dpi, 
                              facecolor="white", )
        gs = self.fig.add_gridspec(nrows=nrows, ncols=ncols, 
                                   left=0.05, right=0.95, bottom=0.05, top=0.95,
                                   wspace=0.1, hspace=0.1)
        for i in range(nrows):
            for j in range(ncols):
                axes.append(self.fig.add_subplot(gs[i, j], projection=cartopy.crs.Robinson()))
        
        self._draw_axes_0(axes[0])
        
        self._save_fig()
        return
    
    pass



class Correlation1(_PlotBase):
    """
    绘制观测和模型的相关性
    """
    def __init__(self,
                 backend: str,
                 file_path: str):
        super().__init__(backend, file_path)
        pass
    
    def _load_data(self, file_path):
        # 自定义时间范围
        file_path = Path(file_path)
        use_df = pd.read_hdf(file_path)
#         assert "inference" in file_path.stem
        
#         time_series = df.loc[:, "time"]
#         time_index = (time_series >= pd.Timestamp("2013-6-1")) & (time_series < pd.Timestamp("2014-6-1"))
#         use_df = df.loc[time_index, ["time", "alt", "lat", "lon", "lct", "s4max", "test_s4max"]]
#         use_df.reset_index(drop=True, inplace=True)
        
        alt_series = use_df.loc[:, "alt"]
        alt_index = (alt_series >= 80) & (alt_series <= 130)
        use_df = use_df.loc[alt_index, :]
        
        # 给出不同的季节
        # spring: MAM(3,4,5)
        # summer: JJA(6,7,8)
        # autumn: SON(9,10,11)
        # winter: DJF(12,1,2)
        use_month_series = use_df.loc[:, "time"].dt.month
        # use_time_index = (use_month_series>=3)&(use_month_series<=5) # spring
        # use_time_index = (use_month_series>=6)&(use_month_series<=8) # summer
        # use_time_index = (use_month_series>=9)&(use_month_series<=11) # autumn
        # use_time_index = ((use_month_series>=1)&(use_month_series<=2))|(use_month_series==12) # winter
        
        try:
            data = use_df.loc[use_time_index, ["s4max", "test_s4max"]].values.astype(np.float64)
            pass
        except:
            data = use_df.loc[:, ["s4max", "test_s4max"]].values.astype(np.float64)
            pass
        self.s4max_true = data[:, 0]
        self.s4max_infer = data[:, 1]
    
    def _set_r2_locator_formatter(self, ax):
        ticks = np.linspace(0, 1, 11)
        ticklabels = [f"{t:.1f}" if i%2==0 else f"" for i, t in enumerate(ticks)]
        lct = mpl.ticker.FixedLocator(ticks)
        fmt = mpl.ticker.FixedFormatter(ticklabels)
        
        ax.xaxis.set_major_locator(lct)
        ax.xaxis.set_major_formatter(fmt)
        ax.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(n=2))
        ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
        
        ax.yaxis.set_major_locator(lct)
        ax.yaxis.set_major_formatter(fmt)
        ax.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(n=2))
        ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
        return
    
    def _draw_axes_0(self, ax):
        import scipy
        # true-infer
        grid_true, grid_infer, grid_count, _ = compute_count(self.s4max_true, self.s4max_infer, 
                                                             self.s4max_true,
                                                             tag="true-infer")
        
        # 计算相关系数
        R_value = scipy.stats.pearsonr(self.s4max_true, self.s4max_infer)[0]
        
        
        # 对计数进行归一化
        N = grid_count
        shape = N.shape
        N = N.reshape(-1,1)
        N = MinMaxScaler().fit_transform(N)
        N = N.reshape(shape)
        N = np.ma.masked_where((N==0), N)
        
        # 设定为正方形
        ax.set_box_aspect(1)
        
        # 设定色图，0设置为白色
        colormap = mpl.cm.get_cmap("Greys").copy()
        # colormap.set_under("white")
        
        contourset = ax.contourf(grid_true, grid_infer, N, 
                                 levels=np.linspace(0,1,11), cmap=colormap, 
                                 norm=mpl.colors.Normalize(0,1), extend="both",
                                 alpha=0.95)
        # 绘制colorbar
        box = mpl.transforms.Bbox.from_extents(ax.get_position().xmax+0.01,
                                               ax.get_position().ymin,
                                               ax.get_position().xmax+0.045,
                                               ax.get_position().ymax)
        cax = self.fig.add_axes([.01,.01,.01,.01], position=box)
        cbar = self.fig.colorbar(contourset, cax=cax, extend="both",
                                 format="%.1f")
        cbar.ax.minorticks_on()
        cbar.ax.tick_params(axis="both", which="both", direction="in",labeltop=False)
        for l in cbar.ax.yaxis.get_ticklabels():
            l.set_fontsize(13)
            pass
        cbar.set_label("Normalized number density", loc="center",
                       fontsize=20, fontweight="bold")
        
        ax.set_title(f"r: {R_value:.3f}", fontsize=25, fontweight="bold")
        ax.plot([0, 1], [0, 1], c="black", alpha=0.7, ls="--", lw=3.772)
        ax.set_xlim(0,.5)
        ax.set_ylim(0,.5)
        ax.grid(ls="--", alpha=0.6)
        self._set_r2_locator_formatter(ax)
        
        for l in ax.xaxis.get_ticklabels():
            l.set_fontsize(15)
            pass
        for l in ax.yaxis.get_ticklabels():
            l.set_fontsize(15)
            pass
        
        ax.set_ylabel("Inference", loc="center", fontsize=22)
        ax.set_xlabel("Observation", loc="center", fontsize=22)
        
        return
    
    def plot(self):
        figsize = (10,10)
        dpi = 200
        nrows = 1
        ncols = 1
        axes = []
        
        self.fig = plt.figure(figsize=figsize, dpi=dpi, 
                              facecolor="white", )
        gs = self.fig.add_gridspec(nrows=nrows, ncols=ncols, 
                                   left=0.1, right=0.85, bottom=0.05, top=0.95,
                                   wspace=0.1, hspace=0.1)
        for i in range(nrows):
            for j in range(ncols):
                axes.append(self.fig.add_subplot(gs[i, j]))
        
        self._draw_axes_0(axes[0])
        
        self._save_fig()
        return
    
    pass


class Plot3D1(_PlotBase):
    """
    绘制3D图
    """
    def __init__(self,
                 backend: str,
                 file_path: str):
        super().__init__(backend, file_path)
        pass
    
    def _load_data(self, file_path):
        # 自定义时间范围
        file_path = Path(file_path)
        df = pd.read_hdf(file_path)
        
        s4max_str = "test_s4max" if "_inference" in file_path.stem else "s4max"
        # s4max_str = "s4max"
        
        time_series = df.loc[:, "time"]
        time_index = (time_series >= pd.Timestamp("2012-1-1")) & (time_series < pd.Timestamp("2013-1-1"))
        use_df = df.loc[time_index, ["time", "alt", "lat", "lon", "lct", s4max_str]]
        use_df.reset_index(drop=True, inplace=True)
        
        alt_series = use_df.loc[:, "alt"]
        alt_index = (alt_series >= 80) & (alt_series <= 130)
        use_df = use_df.loc[alt_index, :]
        
        # 给出不同的季节
        # spring: MAM(3,4,5)
        # summer: JJA(6,7,8)
        # autumn: SON(9,10,11)
        # winter: DJF(12,1,2)
        use_month_series = use_df.loc[:, "time"].dt.month
        # use_time_index = (use_month_series>=3)&(use_month_series<=5) # spring
        # use_time_index = (use_month_series>=6)&(use_month_series<=8) # summer
        # use_time_index = (use_month_series>=9)&(use_month_series<=11) # autumn
        # use_time_index = ((use_month_series>=1)&(use_month_series<=2))|(use_month_series==12) # winter
        
        try:
            data = use_df.loc[use_time_index, 
                              ["alt", "lat", "lon", "lct", s4max_str]].values.astype(np.float64)
            pass
        except:
            data = use_df.loc[:, 
                              ["alt", "lat", "lon", "lct", s4max_str]].values.astype(np.float64)
            pass
        self.alt = data[:, 0]
        self.lat = data[:, 1]
        self.lon = data[:, 2]
        self.lct = data[:, 3]
        self.s4max = data[:, 4]
        
        self.doy = use_df.loc[:, "time"].dt.day_of_year.values
    
    def _draw_axes_0(self, ax):
        # lat-lon
        grid_lon, grid_lat, grid_count, grid_value = compute_count(self.lon, self.lat, 
                                                                   self.s4max,
                                                                   tag="3d-lat-lon")
        grid_count[grid_count==0] = 1
        
        
        # 设定色图，0设置为白色
        cdata = mpl.cm.get_cmap("rainbow").copy()(np.linspace(0,1,256))
        colormap = mpl.colors.LinearSegmentedColormap.from_list("ss", cdata[20:])
        # colormap.set_under("white")
        
        
        ax.xaxis._axinfo["axisline"]["color"] = "red"
        
        
        ax.set_box_aspect((4,4,2))
        ax.view_init(elev=30, azim=-50, roll=0)
        contourset = ax.plot_surface(grid_lon, grid_lat, grid_value/grid_count, 
                                     cmap=colormap, 
                                     norm=mpl.colors.Normalize(0.,0.7),
                                     # vmin=0, vmax=1,
                                     alpha=0.9)
        
        ax.contour3D(grid_lon, grid_lat, grid_value/grid_count,
                   zdir="x", offset=-180, cmap=colormap, 
                   norm=mpl.colors.Normalize(0.2,0.6),
                     alpha=0.9)
        
        ax.contour(grid_lon, grid_lat, grid_value/grid_count,
                   zdir="y", offset=90, cmap=colormap, 
                   norm=mpl.colors.Normalize(0.2,0.6),
                   alpha=0.9)
        
        ax.contour3D(grid_lon, grid_lat, grid_value/grid_count,
                   zdir="z", offset=0, cmap=colormap, 
                   norm=mpl.colors.Normalize(0.2,0.6),
                     alpha=0.9)
        # self.fig.colorbar(contourset)
        ax.set_zlim(0,1)
        
        
        
        # 绘制colorbar
        box = mpl.transforms.Bbox.from_extents(ax.get_position().xmin-0.08,
                                               ax.get_position().ymin-0.025,
                                               ax.get_position().xmin-0.04,
                                               ax.get_position().ymax-0.025)
        cax = self.fig.add_axes([.01,.01,.01,.01], position=box)
        cbar = self.fig.colorbar(contourset, cax=cax, extend="both",
                                 format="%.1f")
        cbar.ax.minorticks_on()
        cbar.ax.tick_params(axis="both", which="both", direction="in",labeltop=False)
        for l in cbar.ax.yaxis.get_ticklabels():
            l.set_fontsize(13)
            pass
        cbar.set_label("Normalized number density", loc="center",
                       fontsize=20, fontweight="bold")
        
        
        ax.set_xlim(-180,180)
        ax.set_ylim(-90,90)
        # ax.set_xticks([])
        ax.set_ylabel("Latitude", loc="center", labelpad=10.0, fontsize=20, fontweight="bold")
        ax.set_xlabel("Longitude", loc="center", labelpad=10.0, fontsize=20, fontweight="bold")
        ax.set_zlabel("S4max", labelpad=10.0, fontsize=20, fontweight="bold")
        
        # 由于mpl内部原因，对字典变量设置来改变一些属性
        ax.xaxis.line.set_linewidth(2.)
        ax.yaxis.line.set_linewidth(2.)
        ax.zaxis.line.set_linewidth(2.)
        
        
        for l in ax.get_xticklabels():
            l.set_color("k")
            l.set_fontsize(14)
            l.set_fontweight("bold")
        for l in ax.get_yticklabels():
            l.set_color("k")
            l.set_fontsize(14)
            l.set_fontweight("bold")
        for l in ax.get_zticklabels():
            l.set_color("k")
            l.set_fontsize(14)
            l.set_fontweight("bold")
            
        return
    
    def plot(self):
        figsize = (12,10)
        dpi = 200
        nrows = 1
        ncols = 1
        axes = []
        
        self.fig = plt.figure(figsize=figsize, dpi=dpi, 
                              facecolor="white", )
        gs = self.fig.add_gridspec(nrows=nrows, ncols=ncols, 
                                   left=0.1, right=0.95, bottom=0.05, top=0.95,
                                   wspace=0.1, hspace=0.1)
        for i in range(nrows):
            for j in range(ncols):
                axes.append(self.fig.add_subplot(gs[i, j], projection="3d"))
        self._draw_axes_0(axes[0])
        
        self._save_fig()
        return
    
    pass


@njit(parallel=True)
def compute_count(data_x, data_y, data_z, tag):
    assert len(data_x.shape) == 1
    assert len(data_y.shape) == 1
    with objmode(X="float64[:,:]", Y="float64[:,:]", deltax="float64", deltay="float64"):
        if tag=="alt-lct":
            X, Y = np.mgrid[0: 24: 100*1j, 80: 130: 100*1j]
            pass
        elif tag=="alt-doy":
            X, Y = np.mgrid[0: 365: 100*1j, 80: 130: 100*1j]
            pass
        elif tag=="alt-lat":
            X, Y = np.mgrid[-90: 90: 100*1j, 80: 130: 100*1j]
            pass
        elif tag=="alt-lon":
            X, Y = np.mgrid[-180: 180: 100*1j, 80: 130: 100*1j]
            pass
        elif tag=="lct-lon":
            X, Y = np.mgrid[-180: 180: 80*1j, 0: 24: 80*1j]
            pass
        elif tag=="lat-doy":
            X, Y = np.mgrid[0: 365: 150j, -90: 90: 150j]
            pass
        elif tag=="lat-lon":
            X, Y = np.mgrid[-180: 180: 100j, -90: 90: 100j]
            pass
        elif tag=="true-infer":
            X, Y = np.mgrid[0: 1: 150j, 0: 1: 150j]
            pass
        elif tag=="3d-lat-lon":
            X, Y = np.mgrid[-179: 179: 50j, -89: 89: 50j]
            pass
        else:
            pass

        deltax = np.diff(X, axis=0)[0,0]
        deltay = np.diff(Y, axis=1)[0,0]
        pass
    Z = np.zeros_like(X)
    N = np.zeros_like(X)
    for i in prange(X.shape[0]):
        for j in prange(X.shape[1]):
            for di in prange(len(data_x)):
                dx = data_x[di]
                dy = data_y[di]
                dz = data_z[di]

                grid_x = X[i, j]
                grid_y = Y[i, j]
                if ((dx>=grid_x-(deltax/2))&
                    (dx<grid_x+(deltax/2))&
                    (dy>=grid_y-(deltay/2))&
                    (dy<grid_y+(deltay/2))):
                    N[i, j] += 1
                    Z[i, j] += dz
                    pass
                pass
            pass
        pass
    # with objmode(N="float64[:,:]"):
    #     shape = N.shape
    #     N = N.reshape(-1,1)
    #     N = MinMaxScaler().fit_transform(N)
    #     N = N.reshape(shape)
    #     N = np.ma.masked_where((N==0), N)
    
    return (X, Y, N, Z)


def main():
    
    ### local
    # data_file = Path("./config_01/Infer_resnet_bottleneck_50_L1Loss_NAdam_CosineAnnealingLR_3406_2E-04_099.h5")
    
    # data_file = Path("./config_05/Infer_resnet_bottleneck_50_L1Loss_NAdam_None_3406_2E-04_099.h5")
    # data_file = Path("./config_07/Infer_resnet_bottleneck_50_L1Loss_NAdam_None_3406_1E-04_199.h5")
    # data_file = Path("./config_08/Infer_resnet_bottleneck_50_L1Loss_NAdam_None_3406_2E-04_199.h5")
    
    ### hanhai20
    # data_file = Path("./hanhai20/config_01/Infer_resnet_bottleneck_50_L1Loss_NAdam_None_3406_2E-04_159.h5")
    # data_file = Path("./hanhai20/config_02/Infer_resnet_bottleneck_50_L1Loss_NAdam_StepLR_3406_2E-04_158.h5")
    
    ########################################
    ### 简化版
    data_file = Path("./hanhai20/simple_config_01/Infer_resnet_bottleneck_50_L1Loss_NAdam_StepLR_3406_1E-04_049.h5")
    ## 所有数据
    # data_file = Path("./hanhai22/config_02/Infer_2008_2014_resnet_bottleneck_50_L1Loss_NAdam_StepLR_3406_2E-04_099.h5")
    #####
    # data_file = Path("./hanhai22/config_02/Infer_resnet_bottleneck_50_L1Loss_NAdam_StepLR_3406_2E-04_099.h5")
    #######################################
    
    ### hanhai22 测试集
    # data_file = Path("./hanhai22/config_01/Infer_resnet_bottleneck_50_L1Loss_NAdam_None_3406_2E-04_098.h5")
    
    
    
    # data_file = Path("./hanhai22/config_03/Infer_resnet_bottleneck_50_L1Loss_NAdam_CosineAnnealingLR_3406_2E-04_099.h5")
    # data_file = Path("./hanhai22/config_04/Infer_resnet_bottleneck_50_L1Loss_NAdam_ReduceLROnPlateau_3406_2E-04_099.h5")
    
    # data_file = Path("./hanhai22/config_08/Infer_resnet_bottleneck_50_L1Loss_NAdam_StepLR_3406_2E-04_299.h5") # 2localGPU
    # data_file = Path("./hanhai22/config_06/Infer_1gpu_resnet_bottleneck_50_L1Loss_NAdam_None_3406_1E-04_299.h5") # 1localGPU
    # data_file = Path("./hanhai22/hanhai22_Infer_resnet_bottleneck_50_L1Loss_NAdam_None_3406_1E-04_299.h5") # 4hhGPU
    # data_file = Path("./hanhai22/hanhai22_Infer01_resnet_bottleneck_50_L1Loss_NAdam_None_3406_1E-04_299.h5") # 1hhGPU
    # data_file = Path("./hanhai22/hanhai22_Infer02_resnet_bottleneck_50_L1Loss_NAdam_None_3406_1E-04_299.h5") # 2hhGPU
    # data_file = Path("./hanhai22/config_06/Infer01_resnet_bottleneck_50_L1Loss_NAdam_None_3406_1E-04_299.h5")
    
    # data_file = Path("./hanhai22/config_07/Infer_resnet_bottleneck_50_L1Loss_NAdam_None_3406_2E-04_299.h5")
    # data_file = Path("./hanhai22/config_07/Infer01_resnet_bottleneck_50_L1Loss_NAdam_None_3406_2E-04_299.h5")
    
    # data_file = Path("./hanhai22/config_08/Infer_resnet_bottleneck_50_L1Loss_NAdam_StepLR_3406_2E-04_299.h5")
    # data_file = Path("./hanhai22/config_08/Infer_resnet_bottleneck_50_L1Loss_NAdam_StepLR_3406_2E-04_280.h5")
    
    foEs_infer_file = Path("./hanhai22/config_02/foEs_V1_BP440_2008_2014_resnet_bottleneck_50_L1Loss_NAdam_StepLR_3406_2E-04_099.h5")
    # foEs_infer_file = Path("./hanhai22/config_02/foEs_V1_MH453_2008_2014_resnet_bottleneck_50_L1Loss_NAdam_StepLR_3406_2E-04_099.h5")
    # foEs_infer_file = Path("./hanhai22/config_02/foEs_V1_SA418_2008_2014_resnet_bottleneck_50_L1Loss_NAdam_StepLR_3406_2E-04_099.h5")
    # foEs_infer_file = Path("./hanhai22/config_02/foEs_V1_SH427_2008_2014_resnet_bottleneck_50_L1Loss_NAdam_StepLR_3406_2E-04_099.h5")
    # foEs_infer_file = Path("./hanhai22/config_02/foEs_V1_WU430_2008_2014_resnet_bottleneck_50_L1Loss_NAdam_StepLR_3406_2E-04_099.h5")
    
    
    # data_file = Path("./data/s4max_rf_part_01_downsample.h5")
    
    # foEs_infer_file = Path("./hanhai22/config_02/foEs_V1_BP440_2008_2014_resnet_bottleneck_50_L1Loss_NAdam_StepLR_3406_2E-04_099.h5")
    
    # foEs_infer_file = Path("./hanhai20/simple_config_01/foEs_BP440_resnet_bottleneck_50_L1Loss_NAdam_StepLR_3406_1E-04_049.h5")
    
    
    # NEW CODE
    # backend = "svg"
    backend = "agg"
    # backend = "pdf"
    # backend = "ps"
    
    # obj = FeatureImportance(backend=backend, file_path=data_file, index_name="mdi", is_pressure="level")
    # obj = StatisticalDistribution(backend=backend, file_path=data_file)
    # obj = SeasonDistribution(backend=backend, file_path=data_file)
    # obj = GlobalDistribution(backend=backend, file_path=data_file)
    # obj = OccurrenceDistribution(backend=backend, file_path=data_file)
    # obj = EvaluationMetric(backend=backend, file_path=data_file, foEs_infer_file=foEs_infer_file)
    # obj = ApplicationLegend(backend=backend, file_path=data_file)
    # obj = StationEvaluation(backend=backend, file_path=data_file, foEs_infer_file=foEs_infer_file)
    
    ## 审稿回复的图
    # obj = DifferentYearDistribution(backend=backend, file_path="./test1.h5")
    obj = ConcatTwoFigure(backend=backend, file_path1=data_file, file_path2="./test1.h5")
    # OLD CODE
    # obj = DensityDistribution(backend="agg", file_path=data_file)
    # obj = DensityDistribution2(backend="agg", file_path=data_file)
    # obj = DensityDistribution3(backend="agg", file_path=data_file)
    # obj = DensityDistribution4(backend="agg", file_path=data_file)
    # obj = DensityDistribution5(backend="agg", file_path=data_file)
    
    # obj = Correlation1(backend="agg", file_path=data_file)
    
    # obj = Plot3D1(backend="agg", file_path=data_file)
    
    
    obj.plot()

if __name__ == "__main__":
    main()
    print("Done")