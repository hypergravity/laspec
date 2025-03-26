from astropy.time import Time
from astropy.utils.iers import IERS_Auto

# 获取 IERS 数据表（显式触发更新）
iers_table = IERS_Auto.open()

# 计算某时间的极移参数（触发下载）
t = Time("2024-01-01", scale="utc")
pm_x, pm_y = iers_table.pm_xy(t)


from astropy.time import Time

# 创建 UTC 时间对象
t_utc = Time("2024-01-01 00:00:00", scale="utc")

# 转换为 UT1 时间（触发下载）
t_ut1 = t_utc.ut1


from astropy.coordinates import EarthLocation

# 获取所有预定义站点名称
sites = EarthLocation.get_site_names()
