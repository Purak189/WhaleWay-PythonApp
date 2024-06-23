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
import heapq
import math

# Load the road network graph for a specific area
place_name = "Independencia, Lima, Perú"
graph = ox.graph_from_place(place_name, network_type="drive")

# Convert the graph to GeoDataFrame
nodes_gdf, edges_gdf = ox.graph_to_gdfs(graph)

# Load additional nodes from the GeoJSON file
geojson_file = 'WhaleWay-PythonApp/filtered_nodes.geojson'
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

# Especifica el ID del nodo 'almacen_ElHoyo'
almacen_ElHoyo_id = 6394939470  # Asegúrate de que este ID esté presente en tu grafo

# Plot the graph
fig, ax = plt.subplots(figsize=(12, 12))
ox.plot_graph(graph, ax=ax, node_color='blue', node_size=10, edge_color='gray', show=False, close=False)

highlight_nodes = gdf['@id'].tolist()

nc = ['green' if node == almacen_ElHoyo_id else ('red' if node in highlight_nodes else 'blue') for node in graph.nodes()]
node_pos = {node: (data['x'], data['y']) for node, data in graph.nodes(data=True)}
nx.draw(graph, pos=node_pos, node_color=nc, node_size=20, ax=ax)

# Draw edge labels to show weights
edge_weights = nx.get_edge_attributes(graph, 'weight')
nx.draw_networkx_edge_labels(graph, pos=node_pos, edge_labels=edge_weights, ax=ax, font_size=5, font_color='purple')

# Dijkstra algorithm
def dijkstra(G, s, t):
    q = []
    heapq.heappush(q, (0, s))
    distances = {node: float('inf') for node in G.nodes}
    distances[s] = 0
    previous_nodes = {node: None for node in G.nodes}

    while q:
        current_distance, current_node = heapq.heappop(q)

        if current_node == t:
            path = []
            while previous_nodes[current_node] is not None:
                path.insert(0, current_node)
                current_node = previous_nodes[current_node]
            path.insert(0, s)
            return path, distances[t]

        if current_distance > distances[current_node]:
            continue

        for neighbor in G.neighbors(current_node):
            if G.has_edge(current_node, neighbor):
                weight = G[current_node][neighbor].get('weight', float('inf'))
                distance = current_distance + weight

                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    previous_nodes[neighbor] = current_node
                    heapq.heappush(q, (distance, neighbor))

    return [], float('inf')

# Define start and end nodes
start_node = list(graph.nodes())[0]  # Replace with actual start node id if known
end_node = 4457301898  # Actualiza el nodo de destino

# Find the shortest path using Dijkstra
path, total_distance = dijkstra(graph, start_node, end_node)

# Highlight the path
path_edges = list(zip(path, path[1:]))
ec = ['red' if (u, v) in path_edges or (v, u) in path_edges else 'gray' for u, v in graph.edges()]

# Plot the graph with the path highlighted
fig, ax = plt.subplots(figsize=(12, 12))
ox.plot_graph(graph, ax=ax, node_color='blue', node_size=10, edge_color=ec, show=False, close=False)

nc = ['green' if node == almacen_ElHoyo_id else ('red' if node in highlight_nodes else ('yellow' if node in path else 'blue')) for node in graph.nodes()]
nx.draw(graph, pos=node_pos, node_color=nc, node_size=20, edge_color=ec, ax=ax)

# Draw edge labels to show weights
nx.draw_networkx_edge_labels(graph, pos=node_pos, edge_labels=edge_weights, ax=ax, font_size=5, font_color='purple')

plt.show()

# Plot the path separately
fig, ax = plt.subplots(figsize=(12, 12))
ox.plot_graph(graph, ax=ax, node_color='blue', node_size=10, edge_color='gray', show=False, close=False)

# Resaltar los nodos y bordes del camino
nc = ['green' if node == almacen_ElHoyo_id else ('red' if node in highlight_nodes else ('yellow' if node in path else 'blue')) for node in graph.nodes()]
ec = ['red' if (u, v) in path_edges or (v, u) in path_edges else 'gray' for u, v in graph.edges()]

nx.draw(graph, pos=node_pos, node_color=nc, node_size=20, edge_color=ec, ax=ax, width=[3 if (u, v) in path_edges or (v, u) in path_edges else 1 for u, v in graph.edges()])

# Draw edge labels to show weights
nx.draw_networkx_edge_labels(graph, pos=node_pos, edge_labels=edge_weights, ax=ax, font_size=5, font_color='purple')

plt.show()

print(f"Shortest path: {path}")
print(f"Total distance: {total_distance} meters")
