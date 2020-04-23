from . import parameters_default
from . import parameters

if hasattr(parameters, "input_folder"):
    input_folder = parameters.input_folder
else:
    input_folder = parameters_default.input_folder


# Filepaths
import os
filepath_district_sector_importance = os.path.join('input', input_folder, 'district_sector_importance.csv')
filepath_transport_parameters = os.path.join('input', input_folder, 'transport_parameters.yaml')
filepath_road_nodes = os.path.join('input', input_folder, 'road_nodes.shp')
filepath_road_edges = os.path.join('input', input_folder, 'road_edges.shp')
filepath_extra_road_edges = os.path.join('input', input_folder, "road_edges_extra.shp")
filepath_special_sectors = os.path.join('input', input_folder, "special_sectors.yaml")
filepath_odpoints = os.path.join('input', input_folder, "odpoints.csv")
filepath_tech_coef = os.path.join('input', input_folder, "tech_coef_matrix.csv")
filepath_inventory_duration_targets = os.path.join('input', input_folder, "inventory_duration_targets.csv")
filepath_transit_points = os.path.join('input', input_folder, "country_transit_points.csv")
filepath_imports = os.path.join('input', input_folder, "imports.csv")
filepath_exports = os.path.join('input', input_folder, "exports.csv")
filepath_export_shares = os.path.join('input', input_folder, "export_shares.csv")
filepath_transit_matrix = os.path.join('input', input_folder, "transit_matrix.csv")
filepath_ton_usd_equivalence = os.path.join('input', input_folder, "ton_usd_equivalence.csv")
filepath_population = os.path.join('input', input_folder, "population.csv")
filepath_final_demand = os.path.join('input', input_folder, "final_demand.csv")