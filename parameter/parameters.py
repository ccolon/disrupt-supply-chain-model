input_folder = "Tanzania2"
inventory_duration_target = "inputed"
# extra_inventory_target = 1
import os
# nodeedge_tested = os.path.join('input', input_folder, "all_nodes_short_ranked.csv")
disruption_analysis = {
    "disrupt_nodes_or_edges": "nodes",
    "nodeedge_tested": [456,4]
}
disruption_analysis = None
district_sector_cutoff = 0.1

export_flows = True
export_sc_flow_analysis = True
export_agent_data = True


# inputs_with_extra_inventories = ['TRD']
# buying_sectors_with_extra_inventories = ["SUG"]