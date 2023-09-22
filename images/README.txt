目前数据处理方法采用 3 in dataset.py
随机种子 3406
SELF-ANN使用./hanhai22/config_02/099
简化版数据处理方法用 9 in dataset.py
简化版的使用./hanhai20/simple_config_01/049

随机森林算法
./configs/random_forest_cfig_01.ini
./data/s4max_rf_part_01_downsample.h5
随机种子 1016
每天1/3降采样

./configs/random_forest_cfig_01.ini
./data/s4max_rf_part_01_all.h5
随机种子1016
不采用降采样 使用全部数据

verbose
hanhai22 上的steplr参数为(50, 0.5)
./hanhai20/config_01 对于2013-2014年作为测试集 其损失函数变化曲线较好 可以使用其变化
./hanhai20/config_02 方法3处理 可能已经过拟合了 效果过于优秀
./hanhai22/config_01 方法3处理 可能是一次失败的结果


