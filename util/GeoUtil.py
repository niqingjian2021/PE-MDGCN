from math import sin, asin, cos, radians, fabs, sqrt, pi

EARTH_RADIUS = 6370.856  # 地球平均半径大约6371km
MATH_2_PI = pi * 2
pis_per_degree = MATH_2_PI / 360  # 角度一度所对应的弧度数，360对应2*pi
ONE_KM_PER_DEGREE_FOR_LAT = 1 / float(111.7)


def hav(theta):
    s = sin(theta / 2)
    return s * s


def get_distance(lat0, lon0, lat1, lon2):
    # 用haversine公式计算球面两点间的距离
    # 经纬度转换成弧度
    lat0 = radians(lat0)
    lat1 = radians(lat1)
    lon0 = radians(lon0)
    lon2 = radians(lon2)
    h = hav(fabs(lat0 - lat1)) + cos(lat0) * cos(lat1) * hav(fabs(lon0 - lon2))
    distance = 2 * EARTH_RADIUS * asin(sqrt(h))  # km
    return distance


def get_bbox_for_one_km(lon: float, lat: float):
    """
    纬度：1度= 110.574公里
    经度：1度= 111,320 * cos（纬度）公里
    纬度每相差一度是 111 km
    获取以某经纬度为中心，一公里区域内的 bbox (l_lon. l_lat, r_lon, r_lat)
    """
    ONE_KM_PER_DEGREE_FOR_LON = 1 / (111.7 * cos(radians(fabs(lat))))
    l_lat = lat - ONE_KM_PER_DEGREE_FOR_LAT / 2
    r_lat = lat + ONE_KM_PER_DEGREE_FOR_LAT / 2
    l_lon = lon - ONE_KM_PER_DEGREE_FOR_LON / 2
    r_lon = lon + ONE_KM_PER_DEGREE_FOR_LON / 2
    return l_lon, l_lat, r_lon, r_lat


def get_bounds(lon_arr: list, lat_arr: list):
    """
    获取一堆地点的左上角和右下角坐标
    Args:
        lon_arr: 经度 arr
        lat_arr: 纬度 arr

    Returns:
        l_lon, l_lat, r_lon, r_lat
    """
    sorted_lon_arr = sorted(lon_arr)
    sorted_lat_arr = sorted(lat_arr)
    return sorted_lon_arr[0], sorted_lat_arr[0], sorted_lon_arr[-1], sorted_lat_arr[-1]


def lat_degree2km(dif_degree=.0001, radius=EARTH_RADIUS):
    """
    通过圆环求法，纯纬度上，度数差转距离(km)，与中间点所处的地球上的位置关系不大
    :param dif_degree: 度数差, 经验值0.0001对应11.1米的距离
    :param radius: 圆环求法的等效半径，纬度距离的等效圆环是经线环，所以默认都是earth_radius
    :return: 这个度数差dif_degree对应的距离，单位km
    """
    return radius * dif_degree * pis_per_degree


def lat_km2degree(dis_km, radius=EARTH_RADIUS):
    """
    通过圆环求法，纯纬度上，距离值转度数(diff)，与中间点所处的地球上的位置关系不大
    :param dis_km: 输入的距离，单位km，经验值111km相差约(接近)1度
    :param radius: 圆环求法的等效半径，纬度距离的等效圆环是经线环，所以默认都是earth_radius
    :return: 这个距离dis_km对应在纯纬度上差多少度
    """
    return dis_km / radius / pis_per_degree


def lng_degree2km(dif_degree, center_lat):
    """
    通过圆环求法，纯经度上，度数差转距离(km)，纬度的高低会影响距离对应的经度角度差，具体表达式为：
    :param dif_degree: 度数差
    :param center_lat: 中心点的纬度，默认22为深圳附近的纬度值；为0时表示赤道，赤道的纬线环半径使得经度计算和上面的纬度计算基本一致
    :return: 这个度数差dif_degree对应的距离，单位km
    """
    # 修正后，中心点所在纬度的地表圆环半径
    real_radius = EARTH_RADIUS * cos(center_lat * pis_per_degree)
    return lat_degree2km(dif_degree, real_radius)


def lng_km2degree(dis_km, center_lat):
    """
    纯经度上，距离值转角度差(diff)，单位度数。
    :param dis_km: 输入的距离，单位km
    :param center_lat: 中心点的纬度，默认22为深圳附近的纬度值；为0时表示赤道。
         赤道、中国深圳、中国北京、对应的修正系数分别约为： 1  0.927  0.766
    :return: 这个距离dis_km对应在纯经度上差多少度
    """
    # 修正后，中心点所在纬度的地表圆环半径
    real_radius = EARTH_RADIUS * cos(center_lat * pis_per_degree)
    return lat_km2degree(dis_km, real_radius)


def ab_distance(a_lat, a_lng, b_lat, b_lng):
    """
    计算经纬度表示的ab两点的距离，这是种近似计算，当两点纬度差距不大时更为准确，产生近似的原因也是来主要自于center_lat
    :param a_lat: a点纬度
    :param a_lng: a点经度
    :param b_lat: b点纬度
    :param b_lng: b点纬度
    :return:
    """
    center_lat = .5 * a_lat + .5 * b_lat
    lat_dis = lat_degree2km(abs(a_lat - b_lat))
    lng_dis = lng_degree2km(abs(a_lng - b_lng), center_lat)
    return sqrt(lat_dis ** 2 + lng_dis ** 2)



if __name__ == '__main__':
    print(get_bbox_for_one_km(-73.9600616, 41.0957272))