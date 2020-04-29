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
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

# If there is sth to export, then we create the output folder
exporting_sth = [
    export_log, export_criticality, export_impact_per_firm, export_time_series, export_flows,
    export_firm_table, export_odpoint_table, export_country_table, export_edgelist_table,
    export_inventories, export_district_sector_table, export_sc_network_summary
]
if any(exporting_sth):
    exp_folder = os.path.join('output', input_folder, timestamp)
    if not os.path.isdir(os.path.join('output', input_folder)):
        os.mkdir(os.path.join('output', input_folder))
    os.mkdir(exp_folder)
else:
    exp_folder = None

# Set logging parameters
if export_log:
    log_filename = os.path.join(exp_folder, 'exp.log')
    importlib.reload(logging)
    logging.basicConfig(
            filename=log_filename,
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    logging.getLogger().addHandler(logging.StreamHandler())
else:
    importlib.reload(logging)
    logging.basicConfig(
        level=logging.INFO, 
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
    additional_road_edges = None
    pickle_filename = 'transport_network_base_pickle'
    if new_roads:
        pickle_filename = 'transport_network_modified_pickle'
        extra_road_log = " with extra roads"
    else:
        extra_road_log = ""
    logging.info('Creating transport network'+extra_road_log+'.'+
        'Speeds: '+str(transport_params['speeds'])+
        ', travel_cost_of_time: '+str(transport_params['travel_cost_of_time']))
    T = createTransportNetwork(filepath_road_nodes, filepath_road_edges, 
        transport_params, additional_road_edges=filepath_extra_road_edges)
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
    

### Create firms
# Filter district sector combination
logging.info('Generating the firm table. Sector included: '+str(sectors_to_include)+
    ', districts included: '+str(districts_to_include)+
    ', district sector cutoff: '+str(district_sector_cutoff))
sector_table = pd.read_csv(filepath_sector_table)
firm_table, odpoint_table, filtered_district_sector_table = \
    rescaleNbFirms(filepath_district_sector_importance, filepath_odpoints, sector_table,
        district_sector_cutoff, nb_top_district_per_sector,
        sectors_to_include=sectors_to_include, districts_to_include=districts_to_include)
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
    random_mean_sd=None)
logging.info('Inventory duration targets loaded, inventory_duration_target: '+str(inventory_duration_target))
if extra_inventory_target:
    logging.info("Extra inventory duration: "+str(extra_inventory_target)+\
        " for inputs "+str(inputs_with_extra_inventories)+\
        " for buying sectors "+str(buying_sectors_with_extra_inventories))

# Adding the firms onto the nodes of the transport network
T.locate_firms_on_nodes(firm_list)
logging.info('Firms located on the transport network')


### Create agents: Countries
logging.info('Creating country_list. Countries included: '+str(countries_to_include))
country_list = createCountries(filepath_imports, filepath_exports, filepath_transit_matrix, filepath_transit_points, 
    present_sectors, countries_to_include=countries_to_include, time_resolution=time_resolution)
logging.info('Country_list created: '+str([country.pid for country in country_list]))
# Linking the countries to the the transport network via their transit point.
# This creates "virtual nodes" in the transport network that corresponds to the countries.
# We create a copy of the transport network without such nodes, it will be used for plotting purposes
for country in country_list:
    T.connect_country(country)
# T_noCountries =  T.subgraph([node for node in T.nodes if T.nodes[node]['type']!='virtual'])


### Specify the weight of a unit worth of good, which may differ according to sector, or even to each firm/countries
# Note that for imports, i.e. for the goods delivered by a country, and for transit flows, we do not disentangle sectors
# In this case, we use an average.
firm_list, country_list = loadTonUsdEquivalence(sector_table, firm_list, country_list)

### Create agents: Households
logging.info('Defining the final demand to each firm. time_resolution: '+str(time_resolution))
firm_table = defineFinalDemand(firm_table, odpoint_table, 
    filepath_population=filepath_population, filepath_final_demand=filepath_final_demand,
    time_resolution=time_resolution)
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
if export_sc_network_summary:
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


if disruption_analysis is None:
    logging.info("No disruption. Simulation of the initial state")
    t0 = time.time()

    # comments: not sure if the other initialization mode is (i) working and (ii) useful
    setInitialSCConditions(transport_network=T, sc_network=G, firm_list=firm_list, 
        country_list=country_list, households=households, initialization_mode="equilibrium")

    obs = Observer(firm_list, Tfinal)

    if export_district_sector_table:
        exportDistrictSectorTable(filtered_district_sector_table, export_folder=exp_folder)

    if export_firm_table or export_odpoint_table:
        exportFirmODPointTable(firm_list, firm_table, odpoint_table,
    export_firm_table=export_firm_table, export_odpoint_table=export_odpoint_table, export_folder=exp_folder)

    if export_country_table:
        exportCountryTable(country_list, export_folder=exp_folder)

    if export_edgelist_table:
        exportEdgelistTable(supply_chain_network=G, export_folder=exp_folder)

    if export_inventories:
        exportInventories(firm_list, export_folder=exp_folder)

    ### Run the simulation
    runOneTimeStep(transport_network=T, sc_network=G, firm_list=firm_list, 
        country_list=country_list, households=households,
        propagate_input_price_change=propagate_input_price_change,
        rationing_mode=rationing_mode,
        observer=obs,
        time_step=0,
        export_folder=exp_folder,
        export_flows=export_flows, 
        flow_types_to_export = present_sectors+['domestic_B2B', 'transit', 'import', 'export', 'total'],
        export_sc_flow_analysis=export_sc_flow_analysis, 
        export_agent_data=export_agent_data)

    logging.info("Initialization completed, "+str((time.time()-t0)/60)+" min")


else:
    disruption_list = defineDisruptionList(disruption_analysis, transport_network=T_noCountries,
        nodeedge_tested_topn=nodeedge_tested_topn, nodeedge_tested_skipn=nodeedge_tested_skipn)
    logging.info(str(len(disruption_list))+" "+disrupt_nodes_or_edges+" to be tested: "+str(disruption_list))


exit()
## do function in "simulations.py". One for initial condition, one for running t0
## see output, compare with TZA results
## do it for Cambodia

### Disruption Loop

if export_criticality:
    with open(os.path.join(exp_folder, 'criticality.csv'), "w") as myfile:
        myfile.write(
            'disrupted_node' \
            + ',' + 'disrupted_edge' \
            + ',' + 'disruption_duration' \
            + ',' + 'households_extra_spending' \
            + ',' + 'households_extra_spending_local' \
            + ',' + 'spending_recovered' \
            + ',' + 'countries_extra_spending' \
            + ',' + 'countries_consumption_loss' \
            + ',' + 'households_consumption_loss' \
            + ',' + 'households_consumption_loss_local' \
            + ',' + 'consumption_recovered' \
            + ',' + 'generalized_cost_normal' \
            + ',' + 'generalized_cost_disruption' \
            + ',' + 'generalized_cost_country_normal' \
            + ',' + 'generalized_cost_country_disruption' \
            + ',' + 'usd_transported_normal' \
            + ',' + 'usd_transported_disruption' \
            + ',' + 'tons_transported_normal' \
            + ',' + 'tons_transported_disruption' \
            + ',' + 'tonkm_transported_normal' \
            + ',' + 'tonkm_transported_disruption' \
            + ',' + 'computing_time' \
            + "\n")
    

for disrupted_stuff in disruption_list:
    if disruption_analysis['disrupt_nodes_or_edges'] == 'nodes':
        write_disrupted_stuff = str(disrupted_stuff) + ',' + 'NA'
    elif disruption_analysis['disrupt_nodes_or_edges'] == 'edges':
        write_disrupted_stuff = 'NA' + ',' + str(disrupted_stuff)

    t0 = time.time()
    
    ### Set initial conditions and create observer
    logging.info("Setting initial conditions")
    T.reinitialize_flows_and_disruptions()
    set_initial_conditions(G, firm_list, households, country_list, "equilibrium")
    obs = Observer(firm_list, Tfinal, exp_folder)
    logging.info("Initial conditions set")

    ### There are a number of export file that we export only once, after setting the initial conditions
    if disrupted_stuff == disruption_list[0]:
        logging.info("Exporting files")
        if export_firm_table or export_odpoint_table:
            imports = pd.Series({firm.pid: sum([val for key, val in firm.purchase_plan.items() if str(key)[0]=="C"]) for firm in firm_list}, name='imports')
            production_table = pd.Series({firm.pid: firm.production_target for firm in firm_list}, name='total_production')
            b2c_share = pd.Series({firm.pid: firm.clients[-1]['share'] if -1 in firm.clients.keys() else 0 for firm in firm_list}, name='b2c_share')
            export_share = pd.Series({firm.pid: sum([firm.clients[x]['share'] for x in firm.clients.keys() if isinstance(x, str)]) for firm in firm_list}, name='export_share')
            production_table = pd.concat([production_table, b2c_share, export_share, imports], axis=1)
            production_table['production_toC'] = production_table['total_production']*production_table['b2c_share']
            production_table['production_toB'] = production_table['total_production']-production_table['production_toC']
            production_table['production_exported'] = production_table['total_production']*production_table['export_share']
            production_table.index.name = 'id'
            firm_table = firm_table.merge(production_table.reset_index(), on="id", how="left")
            prod_per_sector_ODpoint_table = firm_table.groupby(['odpoint', 'sector_id'])['total_production'].sum().unstack().fillna(0).reset_index()
            odpoint_table = odpoint_table.merge(prod_per_sector_ODpoint_table.rename(columns={"odpoint":'od_point'}), on='od_point', how="left")

            if export_firm_table:
                firm_table.to_excel(os.path.join(exp_folder, 'firm_table.xlsx'), index=False)
            
            if export_odpoint_table:
                odpoint_table.to_excel(os.path.join(exp_folder, 'odpoint_table.xlsx'), index=False)
            
        if export_country_table:
            country_table = pd.DataFrame({
                'country_id':[country.pid for country in country_list],
                'country_name':[country.pid for country in country_list],
                'purchases':[sum(country.purchase_plan.values()) for country in country_list],
                'purchases_from_countries':[sum([value if str(key)[0]=='C' else 0 for key, value in country.purchase_plan.items()]) for country in country_list]
            })
            country_table['purchases_from_firms'] = country_table['purchases'] - country_table['purchases_from_countries']
            country_table.to_excel(os.path.join(exp_folder, 'country_table.xlsx'), index=False)

        if export_edgelist_table:
            edgelist_table = pd.DataFrame(extractEdgeList(G))
            edgelist_table.to_excel(os.path.join(exp_folder, 'edgelist_table.xlsx'), index=False)
            logging.info("Average distance all: "+str(edgelist_table['distance'].mean()))
            boolindex = (edgelist_table['supplier_odpoint']!=-1) & (edgelist_table['buyer_odpoint']!=-1)
            logging.info("Average distance only non virtual: "+str(edgelist_table.loc[boolindex, 'distance'].mean()))
            logging.info("Average weighted distance: "+str((edgelist_table['distance']*edgelist_table['flow']).sum()/edgelist_table['flow'].sum()))
            logging.info("Average weighted distance non virtual: "+str((edgelist_table.loc[boolindex, 'distance']*edgelist_table.loc[boolindex, 'flow']).sum()/edgelist_table.loc[boolindex, 'flow'].sum()))
            
        if export_inventories:
            # Evaluate total inventory per good type
            inventories = {}
            for firm in firm_list:
                for input_id, inventory in firm.inventory.items():
                    if input_id not in inventories.keys():
                        inventories[input_id] = inventory
                    else:
                        inventories[input_id] += inventory
                        
            pd.Series(inventories).to_excel(os.path.join(exp_folder, 'inventories.xlsx'))

    ### Run the simulation
    disruption_time = 2 # cannot be below 2
    obs.disruption_time = disruption_time
    if isinstance(disrupted_stuff, list):
        disrupted_stuff = disrupted_stuff
    else:
        disrupted_stuff = [disrupted_stuff]
    if disrupt_nodes_or_edges == 'nodes':
        disrupted_roads = {"edge_link":[], "node_nb":disrupted_stuff}
    elif disrupt_nodes_or_edges == 'edges':
        disrupted_roads = {"edge_link":disrupted_stuff, "node_nb":[]}

    flow_types_to_observe = present_sectors+['domestic', 'transit', 'import', 'export', 'total']
    logging.debug('Simulation will last '+str(Tfinal)+' time steps.')
    logging.info('A disruption will occur at time '+str(disruption_time)+
                 ', it will affect '+str(len(disrupted_roads['node_nb']))+
                 ' nodes and '+str(len(disrupted_roads['edge_link']))+ ' edges for '+str(disruption_duration) +' time steps.')
    
    
    logging.info("Starting time loop")
    for t in range(1, Tfinal+1):
        logging.info('Time t='+str(t))
        if (disruption_duration>0) and (t == disruption_time):
            if model_IO:
                if len(disrupted_roads['node_nb']) == 0:
                    logging.error('With model_IO, nodes should be disrupted')
                else:
                    sectoral_shocks = evaluate_sectoral_shock(firm_table, disrupted_roads['node_nb'])
                    logging.info("Shock on the following sectors: "+str(sectoral_shocks[sectoral_shocks>0]))
                    apply_sectoral_shocks(sectoral_shocks, firm_list)
            else:
                T.disrupt_roads(disrupted_roads, disruption_duration)
        if (model_IO) and (disruption_duration>0) and (t == disruption_time + disruption_duration):
            recover_from_sectoral_shocks(firm_list)
            
            
        allFirmsRetrieveOrders(G, firm_list)
        allFirmsPlanProduction(firm_list, G, price_fct_input=propagate_input_price_change)
        allFirmsPlanPurchase(firm_list)
        allAgentsSendPurchaseOrders(G, firm_list, households, country_list)
        allFirmsProduce(firm_list)
        allAgentsDeliver(G, firm_list, country_list, T, rationing_mode=rationing_mode)
        if export_flows:
            T.compute_flow_per_segment(flow_types_to_observe)
        if congestion:
            if (t==1):
                T.evaluate_normal_traffic()
            else:
                T.evaluate_congestion()
                if len(T.congestionned_edges)>0:
                    logging.info("Nb of congestionned segments: "+str(len(T.congestionned_edges)))
            for firm in firm_list:
                firm.add_congestion_malus2(G, T)
            for country in country_list:
                country.add_congestion_malus2(G, T)
        if export_flows:
            obs.collect_transport_flows(T_noCountries, t, flow_types_to_observe)
        if export_flows and (t==Tfinal):
            with open(os.path.join(exp_folder, 'flows.json'), 'w') as jsonfile:
                json.dump(obs.flows_snapshot, jsonfile)
        allAgentsReceiveProducts(G, firm_list, households, country_list, T)
        T.update_road_state()
        obs.collect_agent_data(firm_list, households, country_list, t)
        if export_flows and (t==1) and False: #legacy, should be removed, we shall do these kind of analysis outside of the core model
            obs.analyzeSupplyChainFlows(G, firm_list, exp_folder)
        logging.debug('End of t='+str(t))
    logging.info("Time loop completed, "+str((time.time()-t0)/60)+" min")


    obs.evaluate_results(T, households, disrupted_roads, disruption_duration, per_firm=export_impact_per_firm, export_folder=None)
    if export_time_series:
        exportTimeSeries(obs, exp_folder)

    if export_criticality:
        with open(os.path.join(exp_folder, 'criticality.csv'), "a") as myfile:
            myfile.write(write_disrupted_stuff \
                + ',' + str(disruption_duration) \
                + ',' + str(obs.households_extra_spending) \
                + ',' + str(obs.households_extra_spending_local) \
                + ',' + str(obs.spending_recovered) \
                + ',' + str(obs.countries_extra_spending) \
                + ',' + str(obs.countries_consumption_loss) \
                + ',' + str(obs.households_consumption_loss) \
                + ',' + str(obs.households_consumption_loss_local) \
                + ',' + str(obs.consumption_recovered) \
                + ',' + str(obs.generalized_cost_normal) \
                + ',' + str(obs.generalized_cost_disruption) \
                + ',' + str(obs.generalized_cost_country_normal) \
                + ',' + str(obs.generalized_cost_country_disruption) \
                + ',' + str(obs.usd_transported_normal) \
                + ',' + str(obs.usd_transported_disruption) \
                + ',' + str(obs.tons_transported_normal) \
                + ',' + str(obs.tons_transported_disruption) \
                + ',' + str(obs.tonkm_transported_normal) \
                + ',' + str(obs.tonkm_transported_disruption) \
                + ',' + str((time.time()-t0)/60) \

                + "\n")
    if export_impact_per_firm:
        if disrupted_stuff == disruption_list[0]:
            logging.debug('export extra spending and consumption with header')
            with open(os.path.join(exp_folder, 'extra_spending.csv'), 'w') as f:
                pd.DataFrame({str(disrupted_stuff): obs.households_extra_spending_per_firm}).transpose().to_csv(f, header=True)
            with open(os.path.join(exp_folder, 'extra_consumption.csv'), 'w') as f:
                pd.DataFrame({str(disrupted_stuff): obs.households_consumption_loss_per_firm}).transpose().to_csv(f, header=True)
        else:
            with open(os.path.join(exp_folder, 'extra_spending.csv'), 'a') as f:
                pd.DataFrame({str(disrupted_stuff): obs.households_extra_spending_per_firm}).transpose().to_csv(f, header=False)
            with open(os.path.join(exp_folder, 'extra_consumption.csv'), 'a') as f:
                pd.DataFrame({str(disrupted_stuff): obs.households_consumption_loss_per_firm}).transpose().to_csv(f, header=False)
    del obs
    
logging.info("End of simulation")
