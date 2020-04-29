from . import parameters_default
from . import parameters

if hasattr(parameters, "input_folder"):
    input_folder = parameters.input_folder
else:
    input_folder = parameters_default.input_folder


# Filepaths
import os
## Transport
filepath_transport_parameters = os.path.join('input', input_folder, 'Transport', 'transport_parameters.yaml')
filepath_road_nodes = os.path.join('input', input_folder, 'Transport', 'Roads', 'road_nodes.shp')
filepath_road_edges = os.path.join('input', input_folder, 'Transport', 'Roads', 'road_edges.shp')
filepath_extra_road_edges = os.path.join('input', input_folder, 'Transport', 'Roads', "road_edges_extra.shp")
filepath_odpoints = os.path.join('input', input_folder, 'Transport', "odpoints.csv")
## Supply
filepath_district_sector_importance = os.path.join('input', input_folder, 'Supply', 'district_sector_importance.csv')
filepath_sector_table = os.path.join('input', input_folder, 'Supply', "sector_table.csv")
filepath_tech_coef = os.path.join('input', input_folder, 'Supply', "tech_coef_matrix.csv")
filepath_inventory_duration_targets = os.path.join('input', input_folder, 'Supply', "inventory_duration_targets.csv")
## Trade
filepath_transit_points = os.path.join('input', input_folder, 'Trade', "country_transit_points.csv")
filepath_imports = os.path.join('input', input_folder, 'Trade', "imports.csv")
filepath_exports = os.path.join('input', input_folder, 'Trade', "exports.csv")
filepath_export_shares = os.path.join('input', input_folder, 'Trade', "export_shares.csv")
filepath_transit_matrix = os.path.join('input', input_folder, 'Trade', "transit_matrix.csv")
## Demand
filepath_population = os.path.join('input', input_folder, 'Demand', "population.csv")
filepath_final_demand = os.path.join('input', input_folder, 'Demand', "final_demand.csv")