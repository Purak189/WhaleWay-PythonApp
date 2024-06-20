import json

# Ruta al archivo GeoJSON
archivo_geojson = 'export.geojson'

# Leer el archivo GeoJSON
with open(archivo_geojson, 'r', encoding='utf-8') as f:
    datos_geojson = json.load(f)

# Inicializar un contador para los IDs
id_nodo = 0

# Iterar sobre los nodos en el archivo GeoJSON
for feature in datos_geojson['features']:
    # Obtener las coordenadas del nodo
    coordenadas = feature['geometry']['coordinates']
    
    # Mostrar las coordenadas y el ID del nodo
    print(f'ID: {id_nodo}, Coordenadas: {coordenadas}')
    
    # Incrementar el contador de IDs para el próximo nodo
    id_nodo += 1
