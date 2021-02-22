from . import parameters_default
from . import parameters

if hasattr(parameters, "input_folder"):
    input_folder = parameters.input_folder
else:
    input_folder = parameters_default.input_folder


# Filepaths
import os
## Transport
filepaths = {
    "transport_parameters": os.path.join('input', input_folder, 'Transport', 'transport_parameters.yaml'),
    "transport_modes": os.path.join('input', input_folder, 'Transport', 'transport_modes.csv'),
    # "roads_nodes": os.path.join('input', input_folder, 'Transport', 'Roads', 'roads_nodes.shp'),
    # "roads_edges": os.path.join('input', input_folder, 'Transport', 'Roads', 'roads_edges.shp'),
    # "railways_nodes": os.path.join('input', input_folder, 'Transport', 'Railways', 'railways_nodes.shp'),
    # "railways_edges": os.path.join('input', input_folder, 'Transport', 'Railways', 'railways_edges.shp'),
    # "waterways_nodes": os.path.join('input', input_folder, 'Transport', 'Waterways', 'waterways_nodes.shp'),
    # "waterways_edges": os.path.join('input', input_folder, 'Transport', 'Waterways', 'waterways_edges.shp'),
    # "multimodal_edges": os.path.join('input', input_folder, 'Transport', 'Multimodal', 'multimodal_edges.shp'),
    # "maritime_nodes": os.path.join('input', input_folder, 'Transport', 'Maritime', 'maritime_nodes.shp'),
    # "maritime_edges": os.path.join('input', input_folder, 'Transport', 'Maritime', 'maritime_edges.shp'),
    # "extra_roads_edges": os.path.join('input', input_folder, 'Transport', 'Roads', "road_edges_extra.shp"),
    "roads_nodes": os.path.join('input', input_folder, 'Transport', 'roads_nodes.geojson'),
    "roads_edges": os.path.join('input', input_folder, 'Transport', 'roads_edges.geojson'),
    "railways_nodes": os.path.join('input', input_folder, 'Transport', 'railways_nodes.geojson'),
    "railways_edges": os.path.join('input', input_folder, 'Transport', 'railways_edges.geojson'),
    "waterways_nodes": os.path.join('input', input_folder, 'Transport', 'waterways_nodes.geojson'),
    "waterways_edges": os.path.join('input', input_folder, 'Transport', 'waterways_edges.geojson'),
    "multimodal_edges": os.path.join('input', input_folder, 'Transport', 'multimodal_edges.geojson'),
    "maritime_nodes": os.path.join('input', input_folder, 'Transport', 'maritime_nodes.geojson'),
    "maritime_edges": os.path.join('input', input_folder, 'Transport', 'maritime_edges.geojson'),
    "extra_roads_edges": os.path.join('input', input_folder, 'Transport', "road_edges_extra.geojson"),
    ## Supply
    "district_sector_importance": os.path.join('input', input_folder, 'Supply', 'district_sector_importance.csv'),
    "sector_table": os.path.join('input', input_folder, 'Supply', "sector_table.csv"),
    "tech_coef": os.path.join('input', input_folder, 'Supply', "tech_coef_matrix.csv"),
    "inventory_duration_targets": os.path.join('input', input_folder, 'Supply', "inventory_duration_targets.csv"),
    "sector_cutoffs": os.path.join('input', input_folder, 'Supply', "sector_firm_cutoffs.csv"),
    "adminunit_economic_data": os.path.join('input', input_folder, 'Supply', "commune_economic_data.geojson"),
    ## Trade
    # "entry_points": os.path.join('input', input_folder, 'Trade', "entry_points.csv"),
    "imports": os.path.join('input', input_folder, 'Trade', "import_table.csv"),
    "exports": os.path.join('input', input_folder, 'Trade', "export_table.csv"),
    "transit_matrix": os.path.join('input', input_folder, 'Trade', "transit_matrix.csv"),
    ## Demand
    "population": os.path.join('input', input_folder, 'Demand', "population.csv"),
    "final_demand": os.path.join('input', input_folder, 'Demand', "final_demand.csv"),
    "adminunit_demographic_data": os.path.join('input', input_folder, 'Demand', "commune_demographic_data.geojson")
}
