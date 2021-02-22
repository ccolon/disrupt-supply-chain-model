# Check that the module is used correctly
import sys
if (len(sys.argv)<=2):
    raise ValueError('Syntax: python36 code/main.py (reuse_data 1 0)')

# Import modules
import os
import networkx as nx
import pandas as pd
import time
import random
import logging
import json
import yaml
from datetime import datetime
import importlib
import geopandas as gpd
import pickle

# Import functions and classes
from builder import *
from functions import *
from simulations import *
from export_functions import *
from class_firm import Firm
from class_observer import Observer
from class_transport_network import TransportNetwork

# Import parameters. It should be in this specific order.
project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1, project_path)
from parameter.parameters_default import *
from parameter.parameters import *
from parameter.filepaths_default import *
from parameter.filepaths import *

# Start run
t0 = time.time()
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

# If there is sth to export, then we create the output folder
if any(list(export.values())):
    exp_folder = os.path.join('output', input_folder, timestamp)
    if not os.path.isdir(os.path.join('output', input_folder)):
        os.mkdir(os.path.join('output', input_folder))
    os.mkdir(exp_folder)
    exportParameters(exp_folder)
else:
    exp_folder = None

# Set logging parameters
if export['log']:
    log_filename = os.path.join(exp_folder, 'exp.log')
    importlib.reload(logging)
    logging.basicConfig(
            filename=log_filename,
            level=logging_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    logging.getLogger().addHandler(logging.StreamHandler())
else:
    importlib.reload(logging)
    logging.basicConfig(
        level=logging_level, 
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

logging.info('Simulation '+timestamp+' starting using '+input_folder+' input data.')


# Create transport network
with open(filepaths['transport_parameters'], "r") as yamlfile:
    transport_params = yaml.load(yamlfile, Loader=yaml.FullLoader)

## Creating the transport network consumes time
## To accelerate, we enable storing the transport network as pickle for later reuse
## With new input data, run the model with first arg = 0, it generates the pickle
## Then use first arg = 1, to skip network building and use directly the pickle
pickle_suffix = '_'.join([mode[:3] for mode in transport_modes])+'_pickle'
extra_road_log = ""
if extra_roads:
    pickle_suffix = '_'.join([mode[:3] for mode in transport_modes])+'ext_pickle'
    extra_road_log = " with extra roads"
pickle_transNet_filename = "transNet_"+pickle_suffix
pickle_transEdg_filename = "transEdg_"+pickle_suffix
pickle_transNod_filename = "transNod_"+pickle_suffix
if sys.argv[1] == "0":
    logging.info('Creating transport network'+extra_road_log)
    T, transport_nodes, transport_edges = createTransportNetwork(transport_modes, filepaths, transport_params)
    logging.info('Transport network'+extra_road_log+' created.')
    pickle.dump(T, open(os.path.join('tmp', pickle_transNet_filename), 'wb'))
    pickle.dump(transport_edges, open(os.path.join('tmp', pickle_transEdg_filename), 'wb'))
    pickle.dump(transport_nodes, open(os.path.join('tmp', pickle_transNod_filename), 'wb'))
    logging.info('Transport network saved in tmp folder: '+pickle_transNet_filename)
else:
    T = pickle.load(open(os.path.join('tmp', pickle_transNet_filename), 'rb'))
    transport_edges = pickle.load(open(os.path.join('tmp', pickle_transEdg_filename), 'rb'))
    transport_nodes = pickle.load(open(os.path.join('tmp', pickle_transNod_filename), 'rb'))
    logging.info('Transport network'+extra_road_log+' generated from temp file.')
# Generate weight
logging.info('Generating shortest-path weights on transport network')
T.defineWeights(route_optimization_weight)
# Print data on km per modes
km_per_mode = pd.DataFrame({"km": nx.get_edge_attributes(T, "km"), "type": nx.get_edge_attributes(T, "type")})
km_per_mode = km_per_mode.groupby('type')['km'].sum().to_dict()
logging.info("Total length of transport network is: "+
    "{:.0f} km".format(sum(km_per_mode.values())))
for mode, km in km_per_mode.items():
    logging.info(mode+": {:.0f} km".format(km))
logging.info('Nb of nodes: '+str(len(T.nodes))+', Nb of edges: '+str(len(T.edges)))
# Export transport network
if export['transport']:
    transport_nodes.to_file(os.path.join(exp_folder, "transport_nodes.geojson"), driver='GeoJSON')
    transport_edges.to_file(os.path.join(exp_folder, "transport_edges.geojson"), driver='GeoJSON')


### Create firms, households, and countries
if sys.argv[2] == "0":
    tmp_data = {}

    ### Filter sectors
    logging.info('Filtering the sectors based on their output. '+
        "Cutoff type is "+cutoff_sector_output['type']+
        ", cutoff value is "+str(cutoff_sector_output['value']))
    sector_table = pd.read_csv(filepaths['sector_table'])
    filtered_sectors = filterSector(sector_table, 
        cutoff_sector_output=cutoff_sector_output,
        cutoff_sector_demand=cutoff_sector_demand,
        combine_sector_cutoff=combine_sector_cutoff,
        sectors_to_include=sectors_to_include,
        sectors_to_exclude=sectors_to_exclude)
    output_selected = sector_table.loc[sector_table['sector'].isin(filtered_sectors), 'output'].sum()
    final_demand_selected = sector_table.loc[sector_table['sector'].isin(filtered_sectors), 'final_demand'].sum()
    logging.info(
        str(len(filtered_sectors))+ ' sectors selected over '+str(sector_table.shape[0])+' representing '+
        "{:.0f}%".format(output_selected / sector_table['output'].sum() * 100)+' of total output and '+
        "{:.0f}%".format(final_demand_selected / sector_table['final_demand'].sum() * 100)+' of final demand'
    )
    logging.info('The filtered sectors are: '+str(filtered_sectors))

    logging.info('Generating the firms')
    firm_table = defineFirmsFromGranularEcoData(
        filepath_adminunit_economic_data=filepaths['adminunit_economic_data'], 
        filepath_sector_cutoffs=filepaths['sector_cutoffs'],
        sectors_to_include=filtered_sectors,
        transport_nodes=transport_nodes,
        filepath_sector_table=filepaths['sector_table']
    )
    nb_firms = 'all'
    logging.info('Creating firm_list. nb_firms: '+str(nb_firms)+
        ' reactivity_rate: '+str(reactivity_rate)+
        ' utilization_rate: '+str(utilization_rate))
    firm_list = createFirms(firm_table, nb_firms, reactivity_rate, utilization_rate)
    n = len(firm_list)
    present_sectors = list(set([firm.sector for firm in firm_list]))
    present_sectors.sort()
    flow_types_to_export = present_sectors+['domestic_B2B', 'transit', 'import', 'export', 'total']
    logging.info('Firm_list created, size is: '+str(n))
    logging.info('Sectors present are: '+str(present_sectors))

    ### Create households
    logging.info('Defining the number of housesholds to generate and their purchase plan')
    household_table, household_sector_consumption = defineHouseholds(sector_table, filepaths['adminunit_demographic_data'], 
        present_sectors, pop_density_cutoff, transport_nodes, time_resolution)
    household_list = createHouseholds(household_table, household_sector_consumption)
    logging.info('Households generated')


    # Loading the technical coefficients
    import_code = sector_table.loc[sector_table['type']=='imports', 'sector'].iloc[0]
    firm_list = loadTechnicalCoefficients(firm_list, filepaths['tech_coef'], io_cutoff, import_code)
    logging.info('Technical coefficient loaded. io_cutoff: '+str(io_cutoff))


    # Loading the inventories
    firm_list = loadInventories(firm_list, inventory_duration_target=inventory_duration_target, 
        filepath_inventory_duration_targets=filepaths['inventory_duration_targets'], 
        extra_inventory_target=extra_inventory_target, 
        inputs_with_extra_inventories=inputs_with_extra_inventories, 
        buying_sectors_with_extra_inventories=buying_sectors_with_extra_inventories,
        min_inventory=1)
    logging.info('Inventory duration targets loaded, inventory_duration_target: '+str(inventory_duration_target))
    if extra_inventory_target:
        logging.info("Extra inventory duration: "+str(extra_inventory_target)+\
            " for inputs "+str(inputs_with_extra_inventories)+\
            " for buying sectors "+str(buying_sectors_with_extra_inventories))


    # inventories = {sec:{} for sec in present_sectors}
    # for firm in firm_list:
    #     for input_id, inventory in firm.inventory_duration_target.items():
    #         if input_id not in inventories[firm.sector].keys():
    #             inventories[firm.sector][input_id] = inventory
    #         else:
    #             inventories[firm.sector][input_id] = inventory
    # with open(os.path.join("output", "Test", "inventories.json"), 'w') as f:
    #     json.dump(inventories, f)

    # Adding the firms and household onto the nodes of the transport network

    ### Create agents: Countries
    logging.info('Creating country_list. Countries included: '+str(countries_to_include))
    country_list = createCountries(filepaths['imports'], filepaths['exports'], 
        filepaths['transit_matrix'], transport_nodes, 
        present_sectors, countries_to_include=countries_to_include, 
        time_resolution=time_resolution,
        target_units=monetary_units_in_model, input_units=monetary_units_inputed)
    logging.info('Country_list created: '+str([country.pid for country in country_list]))


    ### Specify the weight of a unit worth of good, which may differ according to sector, or even to each firm/countries
    # Note that for imports, i.e. for the goods delivered by a country, and for transit flows, we do not disentangle sectors
    # In this case, we use an average.
    firm_list, country_list, sector_to_usdPerTon = loadTonUsdEquivalence(sector_table, firm_list, country_list)

    # Save to tmp folder
    tmp_data['sector_table'] = sector_table
    tmp_data['firm_table'] = firm_table
    tmp_data['present_sectors'] = present_sectors
    tmp_data['flow_types_to_export'] = flow_types_to_export
    tmp_data['firm_list'] = firm_list
    tmp_data['household_table'] = household_table
    tmp_data['household_list'] = household_list
    tmp_data['country_list'] = country_list
    pickle_filename = os.path.join('tmp', 'firms_households_countries_pickle')
    pickle.dump(tmp_data, open(pickle_filename, 'wb'))
    logging.info('Firms, households, and countries saved in tmp folder: '+pickle_filename)

else:
    pickle_filename = os.path.join('tmp', 'firms_households_countries_pickle')
    tmp_data = pickle.load(open(pickle_filename, 'rb'))
    sector_table = tmp_data['sector_table']
    present_sectors = tmp_data['present_sectors']
    flow_types_to_export = tmp_data['flow_types_to_export']
    firm_table = tmp_data['firm_table']
    household_table = tmp_data['household_table']
    firm_list = tmp_data['firm_list']
    household_list = tmp_data['household_list']
    country_list = tmp_data['country_list']
    logging.info('Firms, households, and countries generated from temp file.')


# Loacte firms and households on transport network
T.locate_firms_on_nodes(firm_list)
T.locate_households_on_nodes(firm_list)
logging.info('Firms and household located on the transport network')


'''logging.info('Defining the final demand to each firm. time_resolution: '+str(time_resolution))
firm_table = defineFinalDemand(firm_table, odpoint_table, 
    filepath_population=filepaths['population'], filepath_final_demand=filepaths['final_demand'],
    time_resolution=time_resolution, 
    target_units=monetary_units_in_model, input_units=monetary_units_inputed)
logging.info('Creating households and loaded their purchase plan')
households = createSingleHouseholds(firm_table)
logging.info('Households created')'''

### Create supply chain network
if sys.argv[3] == "0":
    logging.info('The supply chain graph is being created. nb_suppliers_per_input: '+str(nb_suppliers_per_input))
    G = nx.DiGraph()

    logging.info('Households are selecting their retailers (domestic B2C flows)')
    for household in household_list:
        household.select_suppliers(G, firm_list, firm_table, nb_suppliers_per_input, weight_localization)

    logging.info('Exporters are being selected by purchasing countries (export B2B flows)')
    logging.info('and trading countries are being connected (transit flows)')
    for country in country_list:
        country.select_suppliers(G, firm_list, country_list, sector_table, transport_nodes)

    logging.info('Firms are selecting their domestic and international suppliers (import B2B flows) (domestic B2B flows).'+
     ' Weight localisation is '+str(weight_localization))
    import_code = sector_table.loc[sector_table['type']=='imports', 'sector'].iloc[0]
    for firm in firm_list:
        firm.select_suppliers(G, firm_list, country_list, nb_suppliers_per_input, weight_localization, 
            import_code=import_code)
    logging.info('The nodes and edges of the supplier--buyer have been created')
    if export['sc_network_summary']:
        exportSupplyChainNetworkSummary(G, firm_list, exp_folder)

    # Save to tmp folder
    tmp_data['supply_chain_network'] = G
    tmp_data['firm_list'] = firm_list
    tmp_data['household_list'] = household_list
    tmp_data['country_list'] = country_list
    pickle_filename = os.path.join('tmp', 'supply_chain_pickle')
    pickle.dump(tmp_data, open(pickle_filename, 'wb'))
    logging.info('Supply chain saved in tmp folder: '+pickle_filename)

else:
    pickle_filename = os.path.join('tmp', 'supply_chain_pickle')
    tmp_data = pickle.load(open(pickle_filename, 'rb'))
    G = tmp_data['supply_chain_network']
    firm_list = tmp_data['firm_list']
    household_list = tmp_data['household_list']
    country_list = tmp_data['country_list']
    logging.info('Supply chain generated from temp file.')


logging.info('Compute the orders on each supplier--buyer link')
setInitialSCConditions(T, G, firm_list, 
    country_list, household_list, initialization_mode="equilibrium")

exit()

### Coupling transportation network T and production network G
logging.info('The supplier--buyer graph is being connected to the transport network')
logging.info('Each B2B and transit edge is being linked to a route of the transport network')
transport_modes = pd.read_csv(filepaths['transport_modes'])
logging.info('Routes for transit flows and import flows are being selected by trading countries finding routes to their clients')
for country in country_list:
    country.decide_initial_routes(G, T, transport_modes, account_capacity, monetary_units_in_model)
logging.info('Routes for export flows and B2B domestic flows are being selected by Tanzanian firms finding routes to their clients')
for firm in firm_list:
    if firm.sector_type not in ['services', 'utility', 'transport']:
        firm.decide_initial_routes(G, T, transport_modes, account_capacity, monetary_units_in_model)
logging.info('The supplier--buyer graph is now connected to the transport network')

logging.info("Initialization completed, "+str((time.time()-t0)/60)+" min")


if disruption_analysis is None:
    logging.info("No disruption. Simulation of the initial state")
    t0 = time.time()

    # comments: not sure if the other initialization mode is (i) working and (ii) useful
    logging.info("Calculating the equilibrium")
    setInitialSCConditions(transport_network=T, sc_network=G, firm_list=firm_list, 
        country_list=country_list, households=households, initialization_mode="equilibrium")

    obs = Observer(firm_list, 0)

    if export['district_sector_table']:
        exportDistrictSectorTable(filtered_district_sector_table, export_folder=exp_folder)

    if export['firm_table'] or export['odpoint_table']:
        exportFirmODPointTable(firm_list, firm_table, odpoint_table, filepaths['roads_nodes'],
    export_firm_table=export['firm_table'], export_odpoint_table=export['odpoint_table'], 
    export_folder=exp_folder)

    if export['country_table']:
        exportCountryTable(country_list, export_folder=exp_folder)

    if export['edgelist_table']:
        exportEdgelistTable(supply_chain_network=G, export_folder=exp_folder)

    if export['inventories']:
        exportInventories(firm_list, export_folder=exp_folder)

    ### Run the simulation at the initial state
    logging.info("Simulating the initial state")
    runOneTimeStep(transport_network=T, sc_network=G, firm_list=firm_list, 
        country_list=country_list, households=households,
        disruption=None,
        congestion=congestion,
        route_optimization_weight=route_optimization_weight,
        explicit_service_firm=explicit_service_firm,
        propagate_input_price_change=propagate_input_price_change,
        rationing_mode=rationing_mode,
        observer=obs,
        time_step=0,
        export_folder=exp_folder,
        export_flows=export['flows'], 
        flow_types_to_export = flow_types_to_export,
        transport_edges = transport_edges,
        export_sc_flow_analysis=export['sc_flow_analysis'],
        monetary_unit_transport_cost="USD", 
        monetary_unit_flow=monetary_units_in_model,
        cost_repercussion_mode=cost_repercussion_mode)

    if export['agent_data']:
        exportAgentData(obs, exp_folder)

    logging.info("Simulation completed, "+str((time.time()-t0)/60)+" min")


else:
    logging.info("Criticality analysis. Defining the list of disruptions")
    disruption_list = defineDisruptionList(disruption_analysis, transport_network=T, 
        nodes=transport_nodes, edges=transport_edges,
        nodeedge_tested_topn=nodeedge_tested_topn, nodeedge_tested_skipn=nodeedge_tested_skipn)
    logging.info(str(len(disruption_list))+" disruptions to simulates.")

    if export['criticality']:
        criticality_export_file = initializeCriticalityExportFile(export_folder=exp_folder)

    if export['impact_per_firm']:
        extra_spending_export_file, missing_consumption_export_file = \
            initializeResPerFirmExportFile(exp_folder, firm_list)

    ### Disruption Loop
    for disruption in disruption_list:
        logging.info("Simulating disruption "+str(disruption))
        t0 = time.time()

        ### Set initial conditions and create observer
        logging.info("Calculating the equilibrium")
        setInitialSCConditions(transport_network=T, sc_network=G, firm_list=firm_list, 
            country_list=country_list, households=households, initialization_mode="equilibrium")

        Tfinal = duration_dic[disruption['duration']]
        obs = Observer(firm_list, Tfinal)

        if disruption == disruption_list[0]:
            export_flows = export['flows']
        else:
            export_flows = False
            
        logging.info("Simulating the initial state")
        runOneTimeStep(transport_network=T, sc_network=G, firm_list=firm_list, 
            country_list=country_list, households=households,
            disruption=None,
            congestion=congestion,
            route_optimization_weight=route_optimization_weight,
            explicit_service_firm=explicit_service_firm,
            propagate_input_price_change=propagate_input_price_change,
            rationing_mode=rationing_mode,
            observer=obs,
            time_step=0,
            export_folder=exp_folder,
            export_flows=export_flows, 
            flow_types_to_export = flow_types_to_export,
            transport_edges = transport_edges,
            export_sc_flow_analysis=export['sc_flow_analysis'],
            monetary_unit_transport_cost="USD", 
            monetary_unit_flow=monetary_units_in_model,
            cost_repercussion_mode=cost_repercussion_mode)

        if disruption == disruption_list[0]:
            if export['district_sector_table']:
                exportDistrictSectorTable(filtered_district_sector_table, export_folder=exp_folder)

            if export['firm_table'] or export['odpoint_table']:
                exportFirmODPointTable(firm_list, firm_table, odpoint_table, filepaths['roads_nodes'],
            export_firm_table=export['firm_table'], export_odpoint_table=export['odpoint_table'], 
            export_folder=exp_folder)

            if export['country_table']:
                exportCountryTable(country_list, export_folder=exp_folder)

            if export['edgelist_table']:
                exportEdgelistTable(supply_chain_network=G, export_folder=exp_folder)

            if export['inventories']:
                exportInventories(firm_list, export_folder=exp_folder)

        obs.disruption_time = 1
        logging.info('Simulation will last '+str(Tfinal)+' time steps.')
        logging.info('A disruption will occur at time 1, it will affect '+
                     str(len(disruption['node']))+' nodes and '+
                     str(len(disruption['edge']))+' edges for '+
                     str(disruption['duration']) +' time steps.')
        
        logging.info("Starting time loop")
        for t in range(1, Tfinal+1):
            logging.info('Time t='+str(t))
            runOneTimeStep(transport_network=T, sc_network=G, firm_list=firm_list, 
                country_list=country_list, households=households,
                disruption=disruption,
                congestion=congestion,
                route_optimization_weight=route_optimization_weight,
                explicit_service_firm=explicit_service_firm,
                propagate_input_price_change=propagate_input_price_change,
                rationing_mode=rationing_mode,
                observer=obs,
                time_step=t,
                export_folder=exp_folder,
                export_flows=export_flows, 
                flow_types_to_export=flow_types_to_export,
                transport_edges = transport_edges,
                export_sc_flow_analysis=False,
                monetary_unit_transport_cost="USD", 
                monetary_unit_flow=monetary_units_in_model,
                cost_repercussion_mode=cost_repercussion_mode)
            logging.debug('End of t='+str(t))

            if (t > 1) and epsilon_stop_condition:
                if (households.extra_spending <= epsilon_stop_condition) & \
                   (households.consumption_loss <= epsilon_stop_condition):
                    logging.info('Household extra spending and consumption loss are at pre-disruption value. '\
                    +"Simulation stops.")
                    break

        computation_time = time.time()-t0
        logging.info("Time loop completed, {:.02f} min".format(computation_time/60))


        obs.evaluate_results(T, households, disruption, disruption_analysis['duration'],
         per_firm=export['impact_per_firm'])

        if export['time_series']:
            exportTimeSeries(obs, exp_folder)

        if export['criticality']:
            writeCriticalityResults(criticality_export_file, obs, disruption, 
                disruption_analysis['duration'], computation_time)

        if export['impact_per_firm']:
            writeResPerFirmResults(extra_spending_export_file, 
                missing_consumption_export_file, obs, disruption)

        # if export['agent_data']:
        #     exportAgentData(obs, export_folder=exp_folder)

        del obs
        
    if export['criticality']:
        criticality_export_file.close()

    logging.info("End of simulation")
