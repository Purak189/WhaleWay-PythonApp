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
import random


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

for idx, row in gdf.iterrows():
    node_id = row['@id']
    y, x = row.geometry.y, row.geometry.x
    shop_type = row['name'] if pd.notna(row['name']) else 'N/A'
    graph.add_node(node_id, y=y, x=x, shop=shop_type, pos=(x, y))
    
    # Find nearest neighbors and add edges with weights
    point = np.array([np.radians(y), np.radians(x)])
    dist, ind = tree.query([point], k=5)  # Ajusta k según sea necesario
    for i in ind[0]:
        neighbor_id = nodes_gdf.iloc[i].name  # Asegúrate de que esto obtenga el ID correcto
        distance = int(round(great_circle((y, x), coords[i]).meters))
        if distance > 0:  # Solo agrega la arista si la distancia es mayor que 0
            graph.add_edge(node_id, neighbor_id, weight=distance)
            graph.add_edge(neighbor_id, node_id, weight=distance)  # Si el grafo es no dirigido

# Asegurarse de que todas las aristas tengan un peso asignado
for u, v, data in graph.edges(data=True):
    if 'weight' not in data or data['weight'] == 0:
        y1, x1 = graph.nodes[u]['y'], graph.nodes[u]['x']
        y2, x2 = graph.nodes[v]['y'], graph.nodes[v]['x']
        distance = calculate_distance_geodesic(y1, x1, y2, x2)
        data['weight'] = distance

# Leer los archivos externos Nodos_Tiendas_ids y Tienda_Cantidad_Productos
with open('WhaleWay-PythonApp/Nodos_Tiendas_ids.txt') as f:
    nodos_tiendas_ids = [int(id.strip()) for line in f for id in line.split(',')]

with open('WhaleWay-PythonApp/Tienda_Cantidad_Productos.txt') as f:
    tienda_cantidad_productos = [int(num) for line in f for num in line.strip().split(',')]

# Asignar la cantidad de productos a cada tienda
tiendas_productos = dict(zip(nodos_tiendas_ids, tienda_cantidad_productos))

# Modificar la función dijkstra para incluir la repartición de productos
def dijkstra_con_reparto(G, s, t, tiendas_productos):
    q = []
    heapq.heappush(q, (0, s))
    distances = {node: float('inf') for node in G.nodes}
    distances[s] = 0
    previous_nodes = {node: None for node in G.nodes}
    productos_repartidos = {tienda: 0 for tienda in tiendas_productos}
    total_productos_entregados = 0

    while q:
        current_distance, current_node = heapq.heappop(q)

        if current_node == t:
            path = []
            while previous_nodes[current_node] is not None:
                path.insert(0, current_node)
                current_node = previous_nodes[current_node]
            path.insert(0, s)
            return path, distances[t], productos_repartidos, total_productos_entregados

        if current_distance > distances[current_node]:
            continue

        for neighbor in G.neighbors(current_node):
            if G.has_edge(current_node, neighbor):
                edge_data = G.get_edge_data(current_node, neighbor)
                if isinstance(edge_data, dict):
                    if 0 in edge_data:
                        weight = edge_data[0].get('weight', float('inf'))
                    else:
                        weight = float('inf')
                else:
                    weight = float('inf')
                
                distance = current_distance + weight

                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    previous_nodes[neighbor] = current_node
                    heapq.heappush(q, (distance, neighbor))

                    # Restar 20 productos a la capacidad de la tienda
                    if neighbor in tiendas_productos:
                        productos_restantes = tiendas_productos[neighbor]
                        if productos_restantes >= 10000:
                            tiendas_productos[neighbor] -= 20
                            productos_repartidos[neighbor] += 20
                            total_productos_entregados += 20
                        else:
                            productos_repartidos[neighbor] += productos_restantes
                            total_productos_entregados += productos_restantes
                            tiendas_productos[neighbor] = 0

    return [], float('inf'), productos_repartidos, total_productos_entregados

# Define start and end nodes
start_node = 6394939470  # Reemplazar con el ID de inicio real si se conoce
end_node = random.choice(nodos_tiendas_ids)  # Reemplazar con el nodo de destino real

# Ejecutar Dijkstra modificado
path, total_distance, productos_repartidos, total_productos_entregados = dijkstra_con_reparto(graph, start_node, end_node, tiendas_productos)

# Mostrar resultados
print(f"Recorrido más corto: {path}")
print(f"Distancia total: {total_distance} metros")
print("Productos repartidos por cada tienda:")
for tienda, cantidad in productos_repartidos.items():
    print(f"Tienda {tienda}: {cantidad} productos")

print(f"Total de productos entregados: {total_productos_entregados}")


# Plotear el recorrido en el grafo
fig, ax = plt.subplots(figsize=(12, 12))
ox.plot_graph(graph, ax=ax, node_color='blue', node_size=10, edge_color='gray', show=False, close=False)


node_pos = {node: (data['x'], data['y']) for node, data in graph.nodes(data=True)}
edge_weights = nx.get_edge_attributes(graph, 'weight')



# Highlight the path
path_edges = list(zip(path, path[1:]))
ec = ['red' if (u, v) in path_edges or (v, u) in path_edges else 'gray' for u, v in graph.edges()]

nc = ['green' if node == start_node else ('red' if node == end_node else ('yellow' if node in path else 'blue')) for node in graph.nodes()]

nx.draw(graph, pos=node_pos, node_color=nc, node_size=20, edge_color=ec, ax=ax)
nx.draw_networkx_edge_labels(graph, pos=node_pos, edge_labels=edge_weights, ax=ax, font_size=5, font_color='purple')

plt.show()