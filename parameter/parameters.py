input_folder = "Cambodia"
inventory_duration_target = "inputed"
# extra_inventory_target = 1
import os
# nodeedge_tested = os.path.join('input', input_folder, "all_nodes_short_ranked.csv")
disruption_analysis = {
    "disrupt_nodes_or_edges": "nodes",
    "nodeedge_tested": [456,4],
    "duration": 1
}
# disruption_analysis = None
district_sector_cutoff = 0.003*20

# countries_to_include = ["BDI", "EUR"]#"all"
# countries_to_include = ["BDI","COD","KEN","MWI","MOZ","RWA","UGA","ZMB","AME","ASI","EUR","MDE","OCE","OAF"]

export_flows = True
export_sc_flow_analysis = True
export_agent_data = True
export_log = True
export_criticality = True
export_flows = True
export_sc_flow_analysis = True
export_agent_data = True
export_impact_per_firm = True
export_time_series = True
export_firm_table = True
export_odpoint_table = True
export_country_table = True
export_edgelist_table = True
export_inventories = True
export_district_sector_table = True


# inputs_with_extra_inventories = ['TRD']
# buying_sectors_with_extra_inventories = ["SUG"]