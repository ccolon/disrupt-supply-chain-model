from parameter.parameters_default import *
import logging
import os

input_folder = "Cambodia"
inventory_duration_target = "inputed"

countries_to_include = ['AFR', 'AME', 'ASI', 'EUR', 'OCE', 'THA', 'VNM']

# top_10_nodes = [2608, 2404, 2386, 2380, 2379, 2376, 2373, 2366, 2363, 2361]
top_10_nodes = [1473, 1619, 992, 1832, 1269, 428, 224]
floodable_road_battambang = 3170
tsubasa_bridge = 2001
disruption_analysis = {
    "disrupt_nodes_or_edges": "edges",
    # "nodeedge_tested": ["primary", "trunk"],
    # "nodeedge_tested": [1487, 1462, 1525, 1424],
    "nodeedge_tested": [tsubasa_bridge],
    # "nodeedge_tested": os.path.join('input', input_folder, 'top_hh_loss_nodes.csv'),
    # "nodeedge_tested": ["Sihanoukville international port"],
    # "identified_by": "name",
    "identified_by": "id",
    # "identified_by": "class",
    "duration": 1
}
disruption_analysis = None
congestion = True

# cutoffs
sectors_to_exclude = ['ADM']
district_sector_cutoff = 0.003
cutoff_sector_output = {
    'type': 'percentage',
    'value': 0.01
}
io_cutoff = 0.02
transport_modes = ['roads', 'railways', 'waterways', 'maritime']

route_optimization_weight = "agg_cost" #cost_per_ton time_cost agg_cost

export = {key: True for key in export.keys()}

cost_repercussion_mode = "type1"

# logging_level = logging.DEBUG
