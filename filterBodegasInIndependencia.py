import json

# Definir el bounding box para Independencia, Lima (ajusta según sea necesario)
min_lat, max_lat = -12.05, -11.97
min_lon, max_lon = -77.07, -77.04

# Cargar los datos desde el archivo GeoJSON
with open('export.geojson', 'r') as infile:
    data = json.load(infile)

# Filtrar las características basadas en el bounding box
filtered_features = [
    feature for feature in data['features']
    if min_lat <= feature['geometry']['coordinates'][1] <= max_lat and min_lon <= feature['geometry']['coordinates'][0] <= max_lon
]

# Crear una nueva estructura GeoJSON con las características filtradas
filtered_data = {
    "type": "FeatureCollection",
    "features": filtered_features
}

# Guardar los datos filtrados en un nuevo archivo GeoJSON
with open('filtered_bodegas_nodes', 'w') as outfile:
    json.dump(filtered_data, outfile, indent=2)

# Output a confirmation message
print("Filtered nodes have been saved to filtered_nodes.geojson")
