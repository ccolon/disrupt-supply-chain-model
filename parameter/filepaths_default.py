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