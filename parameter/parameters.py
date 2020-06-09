from parameter.parameters_default import *

input_folder = "Cambodia"
inventory_duration_target = "inputed"
# extra_inventory_target = 1
import os
# nodeedge_tested = os.path.join('input', input_folder, "all_nodes_short_ranked.csv")
disruption_analysis = {
    "disrupt_nodes_or_edges": "edges",
    "nodeedge_tested": [2001],
    "duration": 1
}
disruption_analysis = None
congestion = True
# disruption_analysis = None
district_sector_cutoff = 0.005
cutoff_sector_output = {
    'type': 'percentage',
    'value': 0.01
}
io_cutoff = 0.02
transport_modes = ['roads', 'railways', 'waterways', 'maritime']
# countries_to_include = ["BDI", "EUR"]#"all"
# countries_to_include = ["BDI","COD","KEN","MWI","MOZ","RWA","UGA","ZMB","AME","ASI","EUR","MDE","OCE","OAF"]

export = {key: True for key in export.keys()}

# inputs_with_extra_inventories = ['TRD']
# buying_sectors_with_extra_inventories = ["SUG"]