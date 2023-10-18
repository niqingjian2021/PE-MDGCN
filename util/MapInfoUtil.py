"""
辅助类，主要负责处理地点额外信息
"""

import pandas as pd

# code ref: http://download.geofabrik.de/osm-data-in-gis-formats-free.pdf
from util import GeoUtil
from util.GeoUtil import ab_distance

include_public_pois = ['2001', '2005', '2007', '2009', '2010', '2011', '2014', '2015', '2016']

include_education_poi_f_layer = ['208']

include_poi_f_layers = ['21', '22', '23', '24', '25', '26', '27']

include_pois = include_public_pois + include_education_poi_f_layer + include_poi_f_layers

include_pois_pos_map = {}

include_traffic_roads = ['5111']

poi_a_list = []
poi_list = []
traffic_list = []


def is_considered_poi(poi: str):
    try:
        if poi in include_public_pois:
            return True, 0
        elif poi[0:3] in include_education_poi_f_layer:
            return True, 1
        elif poi[0:2] in include_poi_f_layers:
            return True, 2
    except Exception as ex:
        print('is_considered_poi... err for input [{}], err: {}'.format(poi, str(ex)))
    return False, -1


def is_considered_road(road: str):
    if road in include_traffic_roads:
        return True
    return False


def trans_poi(poi: str):
    res, idx = is_considered_poi(poi)
    if res is True:
        if idx == 0:
            return poi
        elif idx == 1:
            return '208'
        elif idx == 2:
            return poi[0:2]
    return ''


class POI:

    def __init__(self, lon, lat, osm_id, name, code):
        self.lon = lon
        self.lat = lat
        self.osm_id = osm_id
        self.name = name
        self.code = code


    @staticmethod
    def get_near_pois_ret(lon, lat):
        l_lon, l_lat, r_lon, r_lat = GeoUtil.get_bbox_for_one_km(lon, lat)
        _ret = [0] * len(include_pois)
        for poi in poi_list:
            if r_lon > poi.lon > l_lon and r_lat > poi.lat > l_lat:
                _ret[include_pois_pos_map[poi.code]] += 1
        return _ret


    @staticmethod
    def get_near_pois_round(lon, lat, radius=1, taxi=False):
        _ret = [0] * len(include_pois)
        for poi in poi_list:
            if taxi:
                lng = -poi.lon
            else:
                lng = poi.lon
            if ab_distance(lat, lon, poi.lat, lng) <= radius:
                _ret[include_pois_pos_map[poi.code]] += 1
        return _ret


class POI_A:

    def __init__(self, l_lon, l_lat, r_lon, r_lat, osm_id, name, code):
        self.l_lon = l_lon
        self.l_lat = l_lat
        self.r_lon = r_lon
        self.r_lat = r_lat
        self.osm_id = osm_id
        self.name = name
        self.code = code


    @staticmethod
    def get_near_a_pois(lon, lat):
        l_lon, l_lat, r_lon, r_lat = GeoUtil.get_bbox_for_one_km(lon, lat)
        # 判断两矩形是否相交，笑死
        # https://leetcode-cn.com/problems/rectangle-overlap/comments/
        _ret = [0] * len(include_pois)
        for poi in poi_a_list:
            if not (r_lon < poi.l_lon or poi.r_lon < l_lon or r_lat < poi.l_lat or poi.r_lat < l_lat):
                _ret[include_pois_pos_map[poi.code]] += 1
        return _ret


class Traffic:

    def __init__(self, l_lon, l_lat, r_lon, r_lat, osm_id, name, code, ref):
        self.l_lon = l_lon
        self.l_lat = l_lat
        self.r_lon = r_lon
        self.r_lat = r_lat
        self.osm_id = osm_id
        self.name = name
        self.code = code
        self.ref = ref


    def __eq__(self, other):
        if self.osm_id == other.osm_id or self.ref == other.ref:
            return True
        return False


    def __str__(self):
        return 'osm_id: {}, code: {}, name: {}, ref: {}'.format(self.osm_id, self.code, self.name, self.ref)


    @staticmethod
    def get_near_traffic(lon, lat, taxi=False):
        """
        获取某地附近的道路列表
        """
        l_lon, l_lat, r_lon, r_lat = GeoUtil.get_bbox_for_one_km(lon, lat)
        _ret = []
        for road in traffic_list:
            # 我也不懂为什么里面的 lon 全都有负号了
            if taxi:
                llon = -road.l_lon
                rlon = -road.r_lon
            else:
                llon = road.l_lon
                rlon = road.r_lon
            if not (r_lon < llon or rlon < l_lon or r_lat < road.l_lat or road.r_lat < l_lat):
                # print(road)
                _ret.append(road)
        return _ret


    @staticmethod
    def cal_connectivity(arr1: list, arr2: list):
        """
        Aci,j = max(0, conn(vi, vj) - Ani,j)
        计算两地的交通连接性，本方法为 conn(vi, vj) 的部分
        根据论文，conn 仅是一个指示函数
        todo: 发掘其他的连通性函数？
        """
        for road1 in arr1:
            for road2 in arr2:
                if road1 == road2:
                    # 已经重载了 Traffic 的 == 运算符，如果 osm_id 一致或路名一致就说明连通。
                    return 1
        return 0


def load_data():
    include_pois_pos_map.clear()
    poi_list.clear()
    traffic_list.clear()
    for _ in range(len(include_pois)):
        include_pois_pos_map[include_pois[_]] = _

    poi_a_data = pd.read_csv("../OSM/openstreetmap_poi.csv")
    for data in poi_a_data.values:
        _code = trans_poi(str(data[6]))
        if _code != '':
            poi_a_list.append(POI_A(data[0], data[1], data[2], data[3], data[4], data[5], _code))

    poi_data = pd.read_csv("../OSM/openstreetmap_poi_point.csv")
    traffic_data = pd.read_csv("../OSM/openstreetmap_traffic.csv", sep=',', index_col=False)

    for data in poi_data.values:
        _code = trans_poi(str(data[4]))
        if _code != '':
            poi_list.append(POI(data[0], data[1], data[2], data[3], _code))


    for data in traffic_data.values:
        if is_considered_road(str(data[5])):
            traffic_list.append(Traffic(data[0], data[1], data[2], data[3], data[4], str(data[6]), str(data[5]), data[7]))


if __name__ == '__main__':
    load_data()
    locations = pd.read_csv('../location position.csv')
    ret = []
    for location in locations.values:
        print('>>>>{}'.format(location[0]))
        Traffic.get_near_traffic(location[4], location[3])

        # ret.append(get_near_pois(location[4], location[3]))
    # print(include_pois)
