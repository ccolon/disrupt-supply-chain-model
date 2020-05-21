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
    "roads_nodes": os.path.join('input', input_folder, 'Transport', 'Roads', 'roads_nodes.shp'),
    "roads_edges": os.path.join('input', input_folder, 'Transport', 'Roads', 'roads_edges.shp'),
    "railways_nodes": os.path.join('input', input_folder, 'Transport', 'Railways', 'railways_nodes.shp'),
    "railways_edges": os.path.join('input', input_folder, 'Transport', 'Railways', 'railways_edges.shp'),
    "waterways_nodes": os.path.join('input', input_folder, 'Transport', 'Waterways', 'waterways_nodes.shp'),
    "waterways_edges": os.path.join('input', input_folder, 'Transport', 'Waterways', 'waterways_edges.shp'),
    "multimodal_edges": os.path.join('input', input_folder, 'Transport', 'Multimodal', 'multimodal_edges.shp'),
    "extra_roads_edges": os.path.join('input', input_folder, 'Transport', 'Roads', "road_edges_extra.shp"),
    "odpoints": os.path.join('input', input_folder, 'Transport', "odpoints.csv"),
    ## Supply
    "district_sector_importance": os.path.join('input', input_folder, 'Supply', 'district_sector_importance.csv'),
    "sector_table": os.path.join('input', input_folder, 'Supply', "sector_table.csv"),
    "tech_coef": os.path.join('input', input_folder, 'Supply', "tech_coef_matrix.csv"),
    "inventory_duration_targets": os.path.join('input', input_folder, 'Supply', "inventory_duration_targets.csv"),
    ## Trade
    "entry_points": os.path.join('input', input_folder, 'Trade', "entry_points.csv"),
    "imports": os.path.join('input', input_folder, 'Trade', "import_table.csv"),
    "exports": os.path.join('input', input_folder, 'Trade', "export_table.csv"),
    "transit_matrix": os.path.join('input', input_folder, 'Trade', "transit_matrix.csv"),
    ## Demand
    "population": os.path.join('input', input_folder, 'Demand', "population.csv"),
    "final_demand": os.path.join('input', input_folder, 'Demand', "final_demand.csv")
}
