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
    "nodeedge_tested": "all",
    # "nodeedge_tested": ["primary", "trunk"],
    # "nodeedge_tested": [1487, 1462, 1525, 1424],
    #"nodeedge_tested": [tsubasa_bridge],
    # "nodeedge_tested": os.path.join('input', input_folder, 'top_hh_loss_nodes.csv'),
    # "nodeedge_tested": ["Sihanoukville international port"],
    # "identified_by": "name",
    #"identified_by": "id",
    # "identified_by": "class",
    "duration": 1
}
# disruption_analysis = None
congestion = True

# cutoffs
sectors_to_exclude = ['ADM']
district_sector_cutoff = 0.003
# cutoff_sector_output = {
#     'type': 'percentage',
#     'value': 0.02
# }
io_cutoff = 0.02
transport_modes = ['roads', 'railways', 'waterways', 'maritime']

route_optimization_weight = "agg_cost" #cost_per_ton time_cost agg_cost

export = {key: True for key in export.keys()}
# export['transport'] = True

cost_repercussion_mode = "type1"

#duration_dic[1] = 1
# logging_level = logging.DEBUG


export = {
    # Save a log file in the output folder, called "exp.log"
    "log": True,

    # Transport nodes and edges as geojson
    "transport": False,

    # Save the main result in a "criticality.csv" file in the output folder
    # Each line is a simulation, it saves what is disrupted and for how long, and aggregate observables
    "criticality": True,

    # Save the amount of good flowing on each transport segment
    # It saves a flows.json file in the output folder
    # The structure is a dic {"timestep: {"transport_link_id: {"sector_id: flow_quantity}}
    # Can be True or False
    "flows": False,

    # Export information on aggregate supply chain flow at initial conditions
    # Used only if "disruption_analysis: None"
    # See analyzeSupplyChainFlows function for details
    "sc_flow_analysis": False,

    # Whether or not to export data for each agent for each time steps
    # See exportAgentData function for details.
    "agent_data": False,

    # Save firm-level impact results
    # It creates an "extra_spending.csv" file and an "extra_consumption.csv" file in the output folder
    # Each line is a simulation, it saves what was disrupted and the corresponding impact for each firm
    "impact_per_firm": False,

    # Save aggregated time series
    # It creates an "aggregate_ts.csv" file in the output folder
    # Each columns is a time series
    # Exports:
    # - aggregate production
    # - total profit, 
    # - household consumption, 
    # - household expenditure, 
    # - total transport costs, 
    # - average inventories.
    "time_series": False,

    # Save the firm table
    # It creates a "firm_table.xlsx" file in the output folder
    # It gives the properties of each firm, along with production, sales to households, to other firms, exports
    "firm_table": False,


    # Save the OD point table
    # It creates a "odpoint_table.xlsx" file in the output folder
    # It gives the properties of each OD point, along with production, sales to households, to other firms, exports
    "odpoint_table": False,

    # Save the country table
    # It creates a "country_table.xlsx" file in the output folder
    # It gives the trade profile of each country
    "country_table": False,

    # Save the edgelist table
    # It creates a "edgelist_table.xlsx" file in the output folder
    # It gives, for each supplier-buyer link, the distance and amounts of good that flows
    "edgelist_table": False,

    # Save inventories per sector
    # It creates an "inventories.xlsx" file in the output folder
    "inventories": False,

    # Save the combination of district and sector that are over the cutoffs value
    # It creates an "filtered_district_sector.xlsx" file in the output folder
    "district_sector_table": False,

    # Whether or not to export a csv summarizing some topological caracteristics of the supply chain network
    "sc_network_summary": False
}