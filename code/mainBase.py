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
    export_log, export_criticality, export_per_firm, export_time_series, export_flows,
    export_firm_table, export_odpoint_table, export_country_table, export_edgelist_table,
    export_inventories, export_district_sector_table
]
if any(exporting_sth):
    exp_folder = os.path.join('output', timestamp)
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
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

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
    logging.info('Creating transport network'+extra_road_log+'. Speeds: '+str(transport_params['speeds'])+', travel_cost_of_time: '+str(transport_params['travel_cost_of_time']))
    T = createTransportNetwork(filepath_road_nodes, filepath_road_edges, transport_params, additional_road_edges=filepath_extra_road_edges)
    logging.info('Transport network'+extra_road_log+' created. Nb of nodes: '+str(len(T.nodes))+', Nb of edges: '+str(len(T.edges)))
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
    logging.info('Transport network'+extra_road_log+' generated from temp file. Speeds: '+str(transport_params['speeds'])+', travel_cost_of_time: '+str(transport_params["travel_cost_of_time"]))
    


input_IO_filename = os.path.join('input', input_folder, 'input_IO.xlsx')

### Firm and OD table
nb_sectors = 'all'
odpoint_filename = os.path.join('input', input_folder, 'input_odpoints.xlsx')
logging.info('Generating the firm table. nb_sectors: '+str(nb_sectors)+', district sector cutoff: '+str(district_sector_cutoff))
firm_table, od_table = rescaleNbFirms3(filepath_district_sector_importance, odpoint_filename, 
    district_sector_cutoff, nb_top_district_per_sector,
    filepath_special_sectors,
    export_firm_table=export_firm_table, export_ODpoint_table=export_odpoint_table, 
    export_district_sector_table=export_district_sector_table, exp_folder=exp_folder)
logging.info('Firm and OD tables generated')

exit()
### should reimplement sector id as TRIgrammme, easier to debug, makes more sense (import), so mixing of integer and string

### Create agents: Firms
nb_firms = 'all'
logging.info('Creating firm_list. nb_firms: '+str(nb_firms)+', inventory_duration_target: '+str(inventory_duration_target)+', extra_inventory_target '+str(extra_inventory_target)+' reactivity_rate: '+str(reactivity_rate)+' utilization_rate: '+str(utilization_rate))
firm_list = createFirms(firm_table, nb_firms, inventory_duration_target, reactivity_rate, utilization_rate)
n = len(firm_list)
present_sectors = list(set([firm.sector for firm in firm_list]))
present_sectors.sort()
logging.info('Firm_list created, size is: '+str(n))
logging.info('Sectors present are: '+str(present_sectors))
tech_coef_filename = os.path.join('input', input_folder, "tech_coef_matrix.csv")
firm_list = loadTechnicalCoefficients(tech_coef_filename, firm_list, io_cutoff)
logging.info('Technical coefficient loaded. io_cutoff: '+str(io_cutoff))
if inventory_duration_target == 'inputed':
    ### XXX change xlsx to csv
    dic_sector_inventory = pd.read_excel(os.path.join('input', input_folder, 'input_inventory.xlsx'), encoding='utf-8').set_index(['sector', 'input_sector'])['selected_weekly_inventory']
    # if minimum_invent is not None:
    #     dic_sector_inventory[dic_sector_inventory<minimum_invent] = minimum_invent
    dic_sector_inventory = dic_sector_inventory.to_dict()
else:
    dic_sector_inventory = None
if inputs_with_extra_inventories == 'all':
    inputs_with_extra_inventories = present_sectors+['import']
firm_list = loadSectorSpecificInventories(firm_list, default_value=inventory_duration_target, dic_sector_inventory=dic_sector_inventory, random_draw=False, extra_inventory_target=extra_inventory_target, inputs_with_extra_inventories=inputs_with_extra_inventories)
logging.info('Specific inventory duration days loaded')
T.locate_firms_on_nodes(firm_list)
logging.info('Firms located on the transport network')

### Create agents: Countries
nb_countries = 13
time_resolution = 'week'
logging.info('Creating country_list. nb_countries: '+str(nb_countries))
country_list = createCountries(input_IO_filename, nb_countries, present_sectors, time_resolution)
firm_list, country_list = loadUsdPerTon(input_IO_filename, firm_list, country_list)
for country in country_list:
    T.connect_country(country)
logging.info('Country_list created: '+str([country.name for country in country_list]))
# Creating a version of the transport network without the virtual nodes and edges that connect countries.
T_toplot =  T.subgraph([node for node in T.nodes if T.nodes[node]['type']!='virtual'])



if nodeedge_tested == 'important':
    if disruption_duration==1:
        nodes_tested = pd.read_csv(os.path.join('input', input_folder, 'top_node_short_300.csv'), header=None).iloc[:,0].tolist()
        edges_tested = pd.read_csv(os.path.join('input', input_folder, 'top_edge_short_300.csv'), header=None).iloc[:,0].tolist()
    elif disruption_duration==2:
        nodes_tested = pd.read_csv(os.path.join('input', input_folder, 'top_node_duration2_300.csv'), header=None).iloc[:,0].tolist()
        edges_tested = pd.read_csv(os.path.join('input', input_folder, 'top_edge_long_300.csv'), header=None).iloc[:,0].tolist()
    elif disruption_duration==3:
        nodes_tested = pd.read_csv(os.path.join('input', input_folder, 'top_node_duration3_300.csv'), header=None).iloc[:,0].tolist()
        edges_tested = pd.read_csv(os.path.join('input', input_folder, 'top_edge_long_300.csv'), header=None).iloc[:,0].tolist()
    else:
        nodes_tested = pd.read_csv(os.path.join('input', input_folder, 'top_node_long_300.csv'), header=None).iloc[:,0].tolist()
        edges_tested = pd.read_csv(os.path.join('input', input_folder, 'top_edge_long_300.csv'), header=None).iloc[:,0].tolist() #############
        
elif nodeedge_tested == 'specific':
    node_tanga = [5305, 5397]
    node_malinyi = [2338, 2360]
    node_itakamara = [2310]
    node_wami = [2344]
    node_morogoro = [2359, 2358, 2346, 2340, 2325, 2302]
    node_all_morogoro = node_malinyi + node_itakamara + node_wami + node_morogoro
    nodes_tested = [node_all_morogoro, node_malinyi, node_itakamara, node_wami, node_morogoro]
    nodes_tested = [3502]
    nodes_tested = [1215]
    nodes_tested = [3510]
    nodes_tested = [5318]
    nodes_tested = [node_all_morogoro]
    nodes_tested = [node_tanga]

elif nodeedge_tested == 'all':
    nodes_tested = list(T_toplot.nodes)
    edges_tested = list(nx.get_edge_attributes(T_toplot, 'link').values())
    
elif nodeedge_tested == 'all_sorted':
    if (disruption_duration==1) and (~model_IO):
        nodes_tested = pd.read_csv(os.path.join('input', input_folder, 'all_nodes_short_ranked.csv'), header=None).iloc[:,0].tolist()
        edges_tested = pd.read_csv(os.path.join('input', input_folder, 'all_edges_short_ranked.csv'), header=None).iloc[:,0].tolist()
    else:
        nodes_tested = pd.read_csv(os.path.join('input', input_folder, 'all_nodes_long_ranked.csv'), header=None).iloc[:,0].tolist()
        edges_tested = pd.read_csv(os.path.join('input', input_folder, 'all_edges_long_ranked.csv'), header=None).iloc[:,0].tolist() #############
    
if isinstance(nodeedge_tested_topn, int):
    nodes_tested = nodes_tested[:nodeedge_tested_topn]
    edges_tested = edges_tested[:nodeedge_tested_topn]

if isinstance(nodeedge_tested_skipn, int):
    nodes_tested = nodes_tested[nodeedge_tested_skipn:]
    edges_tested = edges_tested[nodeedge_tested_skipn:]

### Create agents: Households
population_filename = os.path.join('input', input_folder, 'input_population.xlsx')
logging.info('Defining the final demand to each firm. time_resolution: '+str(time_resolution))
firm_table = defineFinalDemand(population_filename, input_IO_filename, firm_table, od_table, time_resolution, export_firm_table=export_firm_table, exp_folder=exp_folder)
logging.info('Creating households and loaded their purchase plan')
households = createHouseholds(firm_table)
logging.info('Households created')


### Create network
logging.info('The supplier--buyer graph is being created. nb_suppliers_per_sector: '+str(nb_suppliers_per_sector))
G = nx.DiGraph()
logging.info('Tanzanian households are selecting their Tanzanian retailers (domestic B2C flows)')
households.select_suppliers(G, firm_list, mode='inputed')
logging.info('Tanzanian exporters are being selected by purchasing countries (export B2B flows)')
logging.info('and trading countries are being connected (transit flows)')
pc_of_exporting_firms_per_sector = pd.read_excel(os.path.join('input', input_folder, 'input_exportingfirms.xlsx')).set_index('exporting_sector')['percentage']
for country in country_list:
    country.select_suppliers(G, firm_list, country_list, pc_of_exporting_firms_per_sector)
logging.info('Tanzanian firms are selecting their Tanzanian and international suppliers (import B2B flows) (domestric B2B flows). Weight localisation is '+str(weight_localization))
for firm in firm_list:
    firm.select_suppliers(G, firm_list, country_list, nb_suppliers_per_sector, weight_localization)
logging.info('The nodes and edges of the supplier--buyer have been created')

nb_F2F_links = 0
nb_F2H_lins = 0
nb_C2F_links = 0
nb_F2C_links = 0
nb_C2C_links = 0
for edge in G.edges:
    nb_F2F_links += int(isinstance(edge[0], Firm) and isinstance(edge[1], Firm))
    nb_F2H_lins += int(isinstance(edge[0], Firm) and isinstance(edge[1], Households))
    nb_C2F_links += int(isinstance(edge[0], Country) and isinstance(edge[1], Firm))
    nb_F2C_links += int(isinstance(edge[0], Firm) and isinstance(edge[1], Country))
    nb_C2C_links += int(isinstance(edge[0], Country) and isinstance(edge[1], Country))
logging.info("Nb firm to firm links: "+str(nb_F2F_links))
logging.info("Nb firm to households links: "+str(nb_F2H_lins))
logging.info("Nb country to firm links: "+str(nb_C2F_links))
logging.info("Nb firm to country links: "+str(nb_F2C_links))
logging.info("Nb country to country links: "+str(nb_C2C_links))
logging.info("Nb firm clients in firm records: "+str(sum([sum([1 if isinstance(client_id, int) else 0 for client_id in firm.clients.keys()]) for firm in firm_list])))
logging.info("Nb firm suppliers in firm records: "+str(sum([sum([1 if isinstance(supplier_id, int) else 0 for supplier_id in firm.suppliers.keys()]) for firm in firm_list])))

### Coupling transportation network T and production network G
logging.info('The supplier--buyer graph is being connected to the transport network')
logging.info('Each B2B and transit edge is being linked to a route of the transport network')
logging.info('Routes for transit flows and import flows are being selected by trading countries finding routes to their clients')
for country in country_list:
    country.decide_routes(G, T)
logging.info('Routes for export flows and B2B domestic flows are being selected by Tanzanian firms finding routes to their clients')
for firm in firm_list:
    if firm.location != -1:
        firm.decide_routes(G, T)
logging.info('The supplier--buyer graph is now connected to the transport network')


### Old disruption loop
if disrupt_nodes_or_edges == 'nodes':
    nodesedges_tested = nodes_tested
    logging.info("Nb of nodes tested: "+str(len(nodesedges_tested)))
elif disrupt_nodes_or_edges == 'edges':
    nodesedges_tested = edges_tested
    logging.info("Nb of edges tested: "+str(len(nodesedges_tested)))
logging.info(str(len(nodesedges_tested))+" nodes/edges to be tested: "+str(nodesedges_tested))

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
    

for disrupted_stuff in nodesedges_tested:
    if disrupt_nodes_or_edges == 'nodes':
        write_disrupted_stuff = str(disrupted_stuff) + ',' + 'NA'
    elif disrupt_nodes_or_edges == 'edges':
        write_disrupted_stuff = 'NA' + ',' + str(disrupted_stuff)
    t0 = time.time()
    
    ### Set initial conditions and create observer
    logging.info("Setting initial conditions")
    T.reinitialize_flows_and_disruptions()
    set_initial_conditions(G, firm_list, households, country_list, "equilibrium")
    obs = Observer(firm_list, Tfinal, exp_folder)
    logging.info("Initial conditions set")
    #obs.collect_data(firm_list, households, 0)

    ### There are a number of export file that we export only once, after setting the initial conditions
    if disrupted_stuff == nodesedges_tested[0]:
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
            prod_per_sector_ODpoint_table = firm_table.groupby(['location', 'sector_id'])['total_production'].sum().unstack().fillna(0).reset_index()
            od_table = od_table.merge(prod_per_sector_ODpoint_table.rename(columns={"location":'od_point'}), on='od_point', how="left")

            if export_firm_table:
                firm_table.to_excel(os.path.join(exp_folder, 'firm_table.xlsx'), index=False)
            
            if export_odpoint_table:
                od_table.to_excel(os.path.join(exp_folder, 'odpoint_table.xlsx'), index=False)
            
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
            boolindex = (edgelist_table['supplier_location']!=-1) & (edgelist_table['buyer_location']!=-1)
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
            obs.collect_data_flows(T_toplot, t, flow_types_to_observe)
        if export_flows and (t==Tfinal):
            with open(os.path.join(exp_folder, 'flows.json'), 'w') as jsonfile:
                json.dump(obs.flows_snapshot, jsonfile)
        allAgentsReceiveProducts(G, firm_list, households, country_list, T)
        T.update_road_state()
        obs.collect_data2(firm_list, households, country_list, t)
        if export_flows and (t==1) and False: #legacy, should be removed, we shall do these kind of analysis outside of the core model
            obs.analyzeFlows(G, firm_list, exp_folder)
        logging.debug('End of t='+str(t))
    logging.info("Time loop completed, "+str((time.time()-t0)/60)+" min")


    obs.evaluate_results(T, households, disrupted_roads, disruption_duration, per_firm=export_per_firm, export_folder=None)
    if export_time_series:
        obs.export_time_series(exp_folder)

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
    if export_per_firm:
        if disrupted_stuff == nodesedges_tested[0]:
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
    
logging.info("End of criticality analysis")
