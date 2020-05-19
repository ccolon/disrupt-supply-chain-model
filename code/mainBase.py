# Check that the module is used correctly
import sys
if (len(sys.argv)<=1):
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
else:
    exp_folder = None

# Set logging parameters
logging_level = logging.DEBUG
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
with open(filepath_transport_parameters, "r") as yamlfile:
    transport_params = yaml.load(yamlfile, Loader=yaml.FullLoader)

## Creating the transport network consumes time
## To accelerate, we enable storing the transport network as pickle for later reuse
## With new input data, run the model with first arg = 0, it generates the pickle
## Then use first arg = 1, to skip network building and use directly the pickle
if sys.argv[1] == "0":
    pickle_filename = 'transport_network_base_pickle'
    if new_roads:
        pickle_filename = 'transport_network_modified_pickle'
        extra_road_log = " with extra roads"
    else:
        extra_road_log = ""
        filepath_extra_road_edges = None
    logging.info('Creating transport network'+extra_road_log+'.'+
        'Speeds: '+str(transport_params['speeds'])+
        ', travel_cost_of_time: '+str(transport_params['travel_cost_of_time']))
    T = createTransportNetwork(filepath_road_nodes, filepath_road_edges,
        transport_params, filepath_extra_road_edges=filepath_extra_road_edges)
    logging.info('Transport network'+extra_road_log+' created.'+
        'Nb of nodes: '+str(len(T.nodes))+
        ', Nb of edges: '+str(len(T.edges)))
    pickle.dump(T, open(os.path.join('tmp', pickle_filename), 'wb'))
    logging.info('Transport network saved in tmp folder: '+pickle_filename)
else:
    if new_roads:
        extra_road_log = " with extra roads"
        pickle_filename = 'transport_network_modified_pickle'
    else:
        extra_road_log = ""
        pickle_filename = 'transport_network_base_pickle'
    T = pickle.load(open(os.path.join('tmp', pickle_filename), 'rb'))
    logging.info('Transport network'+extra_road_log+' generated from temp file.'+
        'Speeds: '+str(transport_params['speeds'])+
        ', travel_cost_of_time: '+str(transport_params["travel_cost_of_time"]))
    
### Filter sectors
logging.info('Filtering the sectors based on their output. '+
    "Cutoff type is "+cutoff_sector_output['type']+
    ", cutoff value is "+str(cutoff_sector_output['value']))
sector_table = pd.read_csv(filepath_sector_table)
filtered_sectors = filterSector(sector_table, cutoff=cutoff_sector_output['value'],
    cutoff_type=cutoff_sector_output['type'],
    sectors_to_include=sectors_to_include)
logging.info('The filtered sectors are: '+str(filtered_sectors))

### Create firms
# Filter district sector combination
logging.info('Generating the firm table. '+
    'Districts included: '+str(districts_to_include)+
    ', district sector cutoff: '+str(district_sector_cutoff))
firm_table, odpoint_table, filtered_district_sector_table = \
    rescaleNbFirms(filepath_district_sector_importance, filepath_odpoints, sector_table,
        district_sector_cutoff, nb_top_district_per_sector,
        sectors_to_include=filtered_sectors, districts_to_include=districts_to_include)
#firm_table.to_csv(os.path.join("output", "Test", 'firm_table.csv'))
logging.info('Firm and OD tables generated')

# Creating the firms
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

# Loading the technical coefficients
import_code = sector_table.loc[sector_table['type']=='imports', 'sector'].iloc[0]
firm_list = loadTechnicalCoefficients(firm_list, filepath_tech_coef, io_cutoff, import_code)
logging.info('Technical coefficient loaded. io_cutoff: '+str(io_cutoff))

# Loading the inventories
firm_list = loadInventories(firm_list, inventory_duration_target=inventory_duration_target, 
    filepath_inventory_duration_targets=filepath_inventory_duration_targets, 
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

# Adding the firms onto the nodes of the transport network
T.locate_firms_on_nodes(firm_list)
logging.info('Firms located on the transport network')


### Create agents: Countries
logging.info('Creating country_list. Countries included: '+str(countries_to_include))
country_list = createCountries(filepath_imports, filepath_exports, 
    filepath_transit_matrix, filepath_entry_points, 
    present_sectors, countries_to_include=countries_to_include, 
    time_resolution=time_resolution,
    target_units=monetary_units_in_model, input_units=monetary_units_inputed)
logging.info('Country_list created: '+str([country.pid for country in country_list]))
# Linking the countries to the the transport network via their transit point.
# This creates "virtual nodes" in the transport network that corresponds to the countries.
# We create a copy of the transport network without such nodes, it will be used for plotting purposes
for country in country_list:
    T.connect_country(country)


### Specify the weight of a unit worth of good, which may differ according to sector, or even to each firm/countries
# Note that for imports, i.e. for the goods delivered by a country, and for transit flows, we do not disentangle sectors
# In this case, we use an average.
firm_list, country_list = loadTonUsdEquivalence(sector_table, firm_list, country_list)

### Create agents: Households
logging.info('Defining the final demand to each firm. time_resolution: '+str(time_resolution))
firm_table = defineFinalDemand(firm_table, odpoint_table, 
    filepath_population=filepath_population, filepath_final_demand=filepath_final_demand,
    time_resolution=time_resolution, 
    target_units=monetary_units_in_model, input_units=monetary_units_inputed)
logging.info('Creating households and loaded their purchase plan')
households = createHouseholds(firm_table)
logging.info('Households created')


### Create supply chain network
logging.info('The supply chain graph is being created. nb_suppliers_per_input: '+str(nb_suppliers_per_input))
G = nx.DiGraph()

logging.info('Tanzanian households are selecting their Tanzanian retailers (domestic B2C flows)')
households.select_suppliers(G, firm_list, mode='inputed')

logging.info('Tanzanian exporters are being selected by purchasing countries (export B2B flows)')
logging.info('and trading countries are being connected (transit flows)')
for country in country_list:
    country.select_suppliers(G, firm_list, country_list, sector_table)

logging.info('Tanzanian firms are selecting their Tanzanian and international suppliers (import B2B flows) (domestric B2B flows). Weight localisation is '+str(weight_localization))
for firm in firm_list:
    firm.select_suppliers(G, firm_list, country_list, nb_suppliers_per_input, weight_localization, 
        import_code=import_code)

logging.info('The nodes and edges of the supplier--buyer have been created')
if export['sc_network_summary']:
    exportSupplyChainNetworkSummary(G, firm_list, exp_folder)

### Coupling transportation network T and production network G
logging.info('The supplier--buyer graph is being connected to the transport network')
logging.info('Each B2B and transit edge is being linked to a route of the transport network')
logging.info('Routes for transit flows and import flows are being selected by trading countries finding routes to their clients')
for country in country_list:
    country.decide_routes(G, T)
logging.info('Routes for export flows and B2B domestic flows are being selected by Tanzanian firms finding routes to their clients')
for firm in firm_list:
    if firm.odpoint != -1:
        firm.decide_routes(G, T)
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
        exportFirmODPointTable(firm_list, firm_table, odpoint_table, filepath_road_nodes,
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
        propagate_input_price_change=propagate_input_price_change,
        rationing_mode=rationing_mode,
        observer=obs,
        time_step=0,
        export_folder=exp_folder,
        export_flows=export['flows'], 
        flow_types_to_export = flow_types_to_export,
        filepath_road_edges = filepath_road_edges,
        export_sc_flow_analysis=export['sc_flow_analysis'])

    if export['agent_data']:
        exportAgentData(observer, export_folder)

    logging.info("Simulation completed, "+str((time.time()-t0)/60)+" min")


else:
    logging.info("Criticality analysis. Defining the list of disruptions")
    disruption_list = defineDisruptionList(disruption_analysis, transport_network=T,
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

        logging.info("Simulating the initial state")
        runOneTimeStep(transport_network=T, sc_network=G, firm_list=firm_list, 
            country_list=country_list, households=households,
            disruption=None,
            congestion=congestion,
            propagate_input_price_change=propagate_input_price_change,
            rationing_mode=rationing_mode,
            observer=obs,
            time_step=0,
            export_folder=exp_folder,
            export_flows=export['flows'], 
            flow_types_to_export = flow_types_to_export,
            filepath_road_edges = filepath_road_edges,
            export_sc_flow_analysis=export['sc_flow_analysis'])

        if disruption == disruption_list[0]:
            if export['district_sector_table']:
                exportDistrictSectorTable(filtered_district_sector_table, export_folder=exp_folder)

            if export['firm_table'] or export['odpoint_table']:
                exportFirmODPointTable(firm_list, firm_table, odpoint_table, filepath_road_nodes,
            export_firm_table=export['firm_table'], export_odpoint_table=export['odpoint_table'], 
            export_folder=exp_folder)

            if export['country_table']:
                exportCountryTable(country_list, export_folder=exp_folder)

            if export['edgelist_table']:
                exportEdgelistTable(supply_chain_network=G, export_folder=exp_folder)

            if export['inventories']:
                exportInventories(firm_list, export_folder=exp_folder)

        obs.disruption_time = disruption['duration']
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
                propagate_input_price_change=propagate_input_price_change,
                rationing_mode=rationing_mode,
                observer=obs,
                time_step=t,
                export_folder=exp_folder,
                export_flows=False, 
                flow_types_to_export=flow_types_to_export,
                filepath_road_edges=filepath_road_edges,
                export_sc_flow_analysis=False)
            logging.debug('End of t='+str(t))

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
        
    logging.info("End of simulation")
