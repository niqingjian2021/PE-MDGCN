import math
import time
from datetime import datetime, timedelta


def date_to_timestamp(date: str, climate_data=False):
    """
    将 2022-03-01 00:00:18 类似的时间格式转换为当天开始的时间戳（秒级别）
    """
    if climate_data:
        date_format = "%Y-%m-%dT%H:%M:%S"
    else:
        date_format = "%Y-%m-%d"
    try:
        if not climate_data:
            date = date.split(' ')[0]
        time_array = time.strptime(date, date_format)
        return int(time.mktime(time_array))
    except Exception as ex:
        print('date_to_timestamp... exception: ' + str(ex))
        return 0


def date_to_nearest_multiple_of_timestamp(date: str, times: int, date_divider='-'):
    """
    将 2022-03-01 00:00:18 类似的时间格式转换为间隔 times 分钟“取整”的时间戳（秒级别）
    当 times = 5 时，即对分钟取最接近的 5 的倍数，不考虑秒……
    例如：
    2022-03-01 00:13:18 -> 2022-03-01 00:15:00 -> timestamp
    2022-03-01 10:22:11 -> 2022-03-01 10:20:00 -> timestamp
    2021-07-01 23:59:53 -> 2021-07-02 00:00:00 -> timestamp
    """
    try:
        t = datetime.strptime(date, "%Y{}%m{}%d %H:%M:%S".format(date_divider, date_divider))
        t = t.replace(second=0)
        trans_min = math.ceil(t.minute / times) * times  # 最近的 times 的倍数
        # 找差值
        minute_offset = trans_min - t.minute
        t += timedelta(minutes=minute_offset)
        return int(datetime.timestamp(t))
    except Exception as ex:
        print('date_to_nearest_multiple_of_timestamp... exception: ' + str(ex))
        return 0


def date_to_multiple_of_timestamp(date: str, times: int, date_divider='-'):
    """
    取整区间，比如 times=30 时，0-30 的数据归为一体，31-60 的数据归为一体
    times 必须是 60 的约数
    """
    if 60 % times != 0:
        print('date_to_multiple_of_timestamp... 60 % times != 0')
        return 0
    try:
        t = datetime.strptime(date, "%Y{}%m{}%d %H:%M:%S".format(date_divider, date_divider))
        t = t.replace(second=0)
        idx = t.minute / times
        res = int(idx) * times
        t = t.replace(minute=int(res))
        return int(datetime.timestamp(t))
    except Exception as ex:
        print('date_to_multiple_of_timestamp... exception: ' + str(ex))
        return 0


def timestamp_to_dayofweek(timestamp):
    date = time.strftime("%Y-%m-%d", time.localtime(int(timestamp)))
    return datetime.strptime(date, "%Y-%m-%d").weekday()


def timestamp_to_hour_min_sec(timestamp):
    d = datetime.fromtimestamp(timestamp)
    return d.hour, d.minute, d.second


def get_last_week_timestamp(cur_timestamp):
    """
    获取 cur_timestamp（秒级别） 的上一周同一时候的时间戳
    比如输入这周三早上十点十分的时间戳，可以得到上周三早上十点十分的时间戳
    用于 HA
    """
    return cur_timestamp - 1 * 60 * 60 * 24 * 7


def get_yesterday_timestamp(cur_timestamp):
    return cur_timestamp - 1 * 60 * 60 * 24


def timestamp_2_time(timestamp):
    time_local = time.localtime(timestamp)
    # 转换成新的时间格式(2016-05-05 20:28:54)
    dt = time.strftime("%Y-%m-%d", time_local)
    return dt

def timestamp_2_detailed_time(timestamp):
    time_local = time.localtime(timestamp)
    dt = time.strftime("%Y-%m-%d %H:%M:%S", time_local)
    return dt