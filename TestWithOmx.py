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

# Load the road network graph for a specific area
place_name = "Independencia, Lima, Perú"
graph = ox.graph_from_place(place_name, network_type="drive")

# Convert the graph to GeoDataFrame
nodes_gdf, edges_gdf = ox.graph_to_gdfs(graph)

# Load additional nodes from the GeoJSON file
geojson_file = 'filtered_nodes.geojson'
with open(geojson_file) as f:
    geojson_data = json.load(f)

# Convert GeoJSON features to GeoDataFrame
gdf = gpd.GeoDataFrame.from_features(geojson_data['features'])

# Prepare BallTree for efficient distance computation
coords = np.array([(y, x) for x, y in nodes_gdf[['x', 'y']].values])
tree = BallTree(np.radians(coords), metric='haversine')

# Función para calcular la distancia entre dos puntos usando geodesic
def calculate_distance_geodesic(y1, x1, y2, x2):
    distance = geodesic((y1, x1), (y2, x2)).meters
    return int(round(distance))

# Adding nodes and edges with weights to the graph
for idx, row in gdf.iterrows():
    node_id = row['@id']
    y, x = row.geometry.y, row.geometry.x
    shop_type = row['name'] if pd.notna(row['name']) else 'N/A'
    graph.add_node(node_id, y=y, x=x, shop=shop_type, pos=(x, y))
    
    # Find nearest neighbors and add edges with weights
    point = np.array([np.radians(y), np.radians(x)])
    dist, ind = tree.query([point], k=5)  # Adjust k as needed
    for i in ind[0]:
        neighbor_id = nodes_gdf.iloc[i].name  # Si el índice del DataFrame es el identificador del nodo.
        distance = int(round(great_circle((y, x), coords[i]).meters))
        graph.add_edge(node_id, neighbor_id, weight=distance)
        graph.add_edge(neighbor_id, node_id, weight=distance)  # If the graph is undirected

for u, v, data in graph.edges(data=True):
    if 'weight' not in data:
        # Calcula la distancia geodésica si no se ha asignado un peso
        y1, x1 = graph.nodes[u]['y'], graph.nodes[u]['x']
        y2, x2 = graph.nodes[v]['y'], graph.nodes[v]['x']
        distance = calculate_distance_geodesic(y1, x1, y2, x2)
        data['weight'] = distance

# Plot the graph
fig, ax = plt.subplots(figsize=(12, 12))
ox.plot_graph(graph, ax=ax, node_color='blue', node_size=10, edge_color='gray', show=False, close=False)
highlight_nodes = gdf['@id'].tolist()
nc = ['red' if node in highlight_nodes else 'blue' for node in graph.nodes()]
node_pos = {node: (data['x'], data['y']) for node, data in graph.nodes(data=True)}
nx.draw(graph, pos=node_pos, node_color=nc, node_size=20, ax=ax)

# Draw edge labels to show weights
edge_weights = nx.get_edge_attributes(graph, 'weight')
nx.draw_networkx_edge_labels(graph, pos=node_pos, edge_labels=edge_weights, ax=ax, font_size=5, font_color='purple')

plt.show()