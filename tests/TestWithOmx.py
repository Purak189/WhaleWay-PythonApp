import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt
import geopandas as gpd
import json
import pandas as pd
from geopy.distance import great_circle
from geopy.distance import geodesic
from shapely.geometry import Point
from sklearn.neighbors import BallTree
import numpy as np

place_name = "Independencia, Lima, Perú"
graph = ox.graph_from_place(place_name, network_type="drive")

nodes_gdf, edges_gdf = ox.graph_to_gdfs(graph)

geojson_file = 'filtered_nodes.geojson'
with open(geojson_file) as f:
    geojson_data = json.load(f)

gdf = gpd.GeoDataFrame.from_features(geojson_data['features'])

coords = np.array([(y, x) for x, y in nodes_gdf[['x', 'y']].values])
tree = BallTree(np.radians(coords), metric='haversine')

def calculate_distance_geodesic(y1, x1, y2, x2):
    distance = geodesic((y1, x1), (y2, x2)).meters
    return int(round(distance))

for idx, row in gdf.iterrows():
    node_id = row['@id']
    y, x = row.geometry.y, row.geometry.x
    shop_type = row['name'] if pd.notna(row['name']) else 'N/A'
    graph.add_node(node_id, y=y, x=x, shop=shop_type, pos=(x, y))
    
    point = np.array([np.radians(y), np.radians(x)])
    dist, ind = tree.query([point], k=5)  
    for i in ind[0]:
        neighbor_id = nodes_gdf.iloc[i].name  
        distance = int(round(great_circle((y, x), coords[i]).meters))
        graph.add_edge(node_id, neighbor_id, weight=distance)
        graph.add_edge(neighbor_id, node_id, weight=distance)  

for u, v, data in graph.edges(data=True):
    if 'weight' not in data:
        y1, x1 = graph.nodes[u]['y'], graph.nodes[u]['x']
        y2, x2 = graph.nodes[v]['y'], graph.nodes[v]['x']
        distance = calculate_distance_geodesic(y1, x1, y2, x2)
        data['weight'] = distance

fig, ax = plt.subplots(figsize=(12, 12))
ox.plot_graph(graph, ax=ax, node_color='blue', node_size=10, edge_color='gray', show=False, close=False)

almacen_ElHoyo_id = 6394939470
highlight_nodes = gdf['@id'].tolist()

nc = ['green' if node == almacen_ElHoyo_id else ('red' if node in highlight_nodes else 'blue') for node in graph.nodes()]
node_pos = {node: (data['x'], data['y']) for node, data in graph.nodes(data=True)}
nx.draw(graph, pos=node_pos, node_color=nc, node_size=20, ax=ax)


edge_weights = nx.get_edge_attributes(graph, 'weight')
nx.draw_networkx_edge_labels(graph, pos=node_pos, edge_labels=edge_weights, ax=ax, font_size=5, font_color='purple')


plt.show()