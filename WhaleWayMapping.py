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
import sys

# Definir la clase Tienda
class Tienda:
    def __init__(self, id, nombre, cantidad_productos):
        self.id = id
        self.nombre = nombre
        self.cantidad_productos = cantidad_productos

place_name = "Independencia, Lima, Perú"
graph = ox.graph_from_place(place_name, network_type="drive")

nodes_gdf, edges_gdf = ox.graph_to_gdfs(graph)

geojson_file = 'filtered_nodes.geojson'
with open(geojson_file) as f:
    geojson_data = json.load(f)

gdf = gpd.GeoDataFrame.from_features(geojson_data['features'])

coords = np.array([(y, x) for x, y in nodes_gdf[['x', 'y']].values])
tree = BallTree(np.radians(coords), metric='haversine')

# Función para calcular la distancia entre dos puntos
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

# Modificar la función dijkstra para incluir la repartición de productos
def dijkstra_con_reparto(G, s, t, tiendas_productos):
    q = []
    heapq.heappush(q, (0, s))
    distances = {node: float('inf') for node in G.nodes}
    distances[s] = 0
    previous_nodes = {node: None for node in G.nodes}
    productos_repartidos = {tienda: 0 for tienda in tiendas_productos.values()}
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
                        tienda = tiendas_productos[neighbor]
                        if tienda.cantidad_productos >= 10000:
                            tienda.cantidad_productos -= 20
                            productos_repartidos[tienda] += 20
                            total_productos_entregados += 20
                        else:
                            productos_repartidos[tienda] += tienda.cantidad_productos
                            total_productos_entregados += tienda.cantidad_productos
                            tienda.cantidad_productos = 0

    return [], float('inf'), productos_repartidos, total_productos_entregados

def main(args):
    if len(args) < 3 or len(args) % 2 != 1:
        print("Uso: python WhaleWayMapping.py <ID_tienda_1> <cantidad_1> <ID_tienda_2> <cantidad_2> ...")
        return

    nodos = [int(args[i]) for i in range(1, len(args), 2)]
    cantidades_productos = [int(args[i+1]) for i in range(1, len(args), 2)]

   
    tiendas = []
    tiendas_productos = {}  
    for id, cantidad in zip(nodos, cantidades_productos):
        nombre = f"Tienda {id}"
        tienda = Tienda(id, nombre, cantidad)
        tiendas.append(tienda)
        tiendas_productos[id] = tienda
    
    # Lista para almacenar todos los recorridos
    all_paths = []
    all_total_distances = []
    all_productos_repartidos = []
    all_total_productos_entregados = []
    almacen = 6394939470  # Nodo de inicio (almacén)

    nodos = [almacen] + nodos + [almacen]  # Nodo de inicio al final

    start_node = nodos[0]  # Nodo de inicio
    end_node = nodos[-1]  # Nodo de destino
    total_distance = 0
    productos_repartidos = {tienda: 0 for tienda in tiendas_productos.values()}
    total_productos_entregados = 0
    paths = []

    print(f"Nodos de inicio y fin: {start_node} -> {end_node}\n")

    for j in range(len(nodos)-1):
        path, distance, productos, entregados = dijkstra_con_reparto(graph, nodos[j], nodos[j+1], tiendas_productos)
        paths.extend(path)  # Extender el camino actual al final de la lista paths
        total_distance += distance
        for tienda in productos:
            productos_repartidos[tienda] += productos[tienda]
        total_productos_entregados += entregados

    # Agregar el nodo de inicio al inicio de paths (almacén)
    paths.insert(0, start_node)

    # Almacenar los resultados
    all_paths.append(paths)
    all_total_distances.append(total_distance)
    all_productos_repartidos.append(productos_repartidos)
    all_total_productos_entregados.append(total_productos_entregados)

    print(f"Recorrido finalizado. Distancia total recorrida: {total_distance} metros")
    print("Productos repartidos por cada tienda:")
    for tienda, cantidad in productos_repartidos.items():
        print(f"{tienda.nombre}: {cantidad} productos")
    print(f"Total de productos entregados en este recorrido: {total_productos_entregados}\n\n")

    # Mostrar todos los recorridos en el grafo
    fig, ax = plt.subplots(figsize=(12, 12))
    ox.plot_graph(graph, ax=ax, node_color='blue', node_size=10, edge_color='gray', show=False, close=False)
    node_pos = {node: (data['x'], data['y']) for node, data in graph.nodes(data=True)}
    edge_weights = nx.get_edge_attributes(graph, 'weight')

    # Colorear los nodos según el tipo
    node_colors = []
    for node in graph.nodes():
        if node == 6394939470:  # Almacén
            node_colors.append('green')
        elif node in nodos:  # Tiendas
            node_colors.append('red')
        else:  # Otros nodos
            node_colors.append('blue')

    # Dibujar los nodos y las aristas del grafo
    nx.draw(graph, pos=node_pos, node_color=node_colors, node_size=20, edge_color='gray', ax=ax)
    nx.draw_networkx_edge_labels(graph, pos=node_pos, edge_labels=edge_weights, ax=ax, font_size=5, font_color='purple')

    # Dibujar la ruta completa de cada recorrido aleatorio en amarillo
    for idx, paths in enumerate(all_paths):
        path_edges = [(paths[i], paths[i+1]) for i in range(len(paths) - 1)]
        nx.draw_networkx_edges(graph, pos=node_pos, edgelist=path_edges, edge_color='brown', width=2.0, ax=ax)

    plt.title("Todos los recorridos generados")
    plt.show()

if __name__ == "__main__":
    # Leer los argumentos de la línea de comandos
    arguments = sys.argv
    main(arguments)