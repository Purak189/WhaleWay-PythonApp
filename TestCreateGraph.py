import networkx as nx
import matplotlib.pyplot as plt

# Función para leer nodos y sus pesos desde el archivo
def read_nodes_with_weights(file_path):
    edges = []
    with open(file_path, 'r') as file:
        next(file)  # Omitir la primera línea (cabecera)
        lines = file.readlines()
        for i in range(len(lines) - 1):  # Iterar hasta el penúltimo elemento
            parts1 = lines[i].strip().split(',')
            parts2 = lines[i + 1].strip().split(',')
            y1, x1 = float(parts1[0]), float(parts1[1])
            y2, x2 = float(parts2[0]), float(parts2[1])
            weight = float(parts1[-1])  # El peso hacia el siguiente nodo
            edges.append(((y1, x1), (y2, x2), weight))
    return edges

# Función para agregar aristas con peso al grafo
def add_weighted_edges_to_graph(G, edges):
    for edge in edges:
        node1, node2, weight = edge
        G.add_edge(node1, node2, weight=weight)

def read_nodes(file_path): 
    nodes = [] 
    with open(file_path, 'r') as file: 
        lines = file.readlines() 
        for line in lines[1:]: 
        # Saltar la primera línea que contiene los encabezados 
            parts = line.strip().split(',') 
            y = float(parts[0]) 
            x = float(parts[1]) 
            street_count = int(parts[2]) 
            highway = parts[3] if parts[3] != 'N/A' else None 
            ref = parts[4] if parts[4] != 'N/A' else None 
            geometry = parts[5] 
            nodes.append((y, x, street_count, highway, ref, geometry)) 

            return nodes

# Crear el grafo
def create_graph(nodes):
    G = nx.Graph()
    for node in nodes:
        y, x, street_count, highway, ref, geometry = node
        G.add_node((y, x), street_count=street_count, highway=highway, ref=ref, geometry=geometry)
    return G

# Leer los nodos y crear el grafo
nodes = read_nodes('nodes.txt')  # Asumiendo que aún necesitas leer los nodos de otro archivo
G = create_graph(nodes)

# Leer las aristas con peso del archivo y agregarlas al grafo
edges_with_weights = read_nodes_with_weights('nodes_with_weights.txt')
add_weighted_edges_to_graph(G, edges_with_weights)

# Dibujar el grafo
def draw_graph(G):
    pos = {node: (node[1], node[0]) for node in G.nodes()}  # NetworkX usa (x, y) para las posiciones
    nx.draw(G, pos, with_labels=False, node_size=10 , node_color='blue', font_size=8, font_color='red')
    plt.show()

# Dibujar el grafo
draw_graph(G)

# Opcional: agregar aristas si tienes información sobre las conexiones entre los nodos
# Ejemplo: G.add_edge((y1, x1), (y2, x2))
