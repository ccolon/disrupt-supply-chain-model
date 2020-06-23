from parameter.parameters_default import *

input_folder = "Cambodia"
inventory_duration_target = "inputed"

disruption_analysis = {
    "disrupt_nodes_or_edges": "nodes",
    "nodeedge_tested": ["Sihanoukville international port"],
    "identified_by": "name",
    "duration": 1
}
# disruption_analysis = None
congestion = True
district_sector_cutoff = 0.005
cutoff_sector_output = {
    'type': 'percentage',
    'value': 0.01
}
io_cutoff = 0.02
transport_modes = ['roads', 'railways', 'waterways', 'maritime']

route_optimization_weight = "cost_per_ton" 

export = {key: True for key in export.keys()}

