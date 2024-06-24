import json

archivo_geojson = 'export.geojson'

# Leer el archivo GeoJSON
with open(archivo_geojson, 'r', encoding='utf-8') as f:
    datos_geojson = json.load(f)

id_nodo = 0

for feature in datos_geojson['features']:
    # Obtener las coordenadas del nodo
    coordenadas = feature['geometry']['coordinates']
    
    # Mostrar las coordenadas y el ID del nodo
    print(f'ID: {id_nodo}, Coordenadas: {coordenadas}')
    
    id_nodo += 1
