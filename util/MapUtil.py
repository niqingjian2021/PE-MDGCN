import folium
from folium.plugins import HeatMap

def draw_heatmap(data, name):
    # data: [[lat, lon, num], ...]
    map_osm = folium.Map(location=[40.75930, -73.98131], zoom_start=5,tiles='Stamen Toner',)
    HeatMap(data).add_to(map_osm)
    file_path = './heatmap_{}.html'.format(name)
    map_osm.save(file_path)


def draw_points(data=[], name=''):
    import folium
    san_map = folium.Map(location=[40.75930, -73.98131], zoom_start=12)
    incidents = folium.map.FeatureGroup()
    # Loop through the 200 crimes and add each to the incidents feature group
    for e in data:
        incidents.add_child(
            folium.CircleMarker(
                [e[0], e[1]],
                radius=7,  # define how big you want the circle markers to be
                color='yellow',
                fill=True,
                fill_color='red',
                fill_opacity=0.4
            )
        )
    san_map.add_child(incidents)
    file_path = './heatmap_{}.html'.format(name)
    san_map.save(file_path)


if __name__ == '__main__':
    draw_points()
