import numpy as np
import networkx as nx
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import math
import json
import os


def load_dictionaries():
    with open('dictionaries.json', 'r') as fp:
        dictionaries = json.load(fp)
    return dictionaries
                                
                                 
def load_sectoral_IOmatrix(type='full', threshold=0):
    iomatrix = pd.read_excel('input_IO.xlsx', sheet_name=type+"_iotable")
    iomatrix = iomatrix.mask(iomatrix<=threshold, 0)
    return iomatrix


def load_sector_color():
    table = pd.read_excel('input_IO.xlsx', sheet_name="sector_color")
    dic = table.set_index("sector_id")['color'].to_dict()
    dic['transit'] = 'darkgrey'
    dic[999] = 'black'
    return dic


def load_sector_name(nb_sectors='all'):
    table = pd.read_excel('input_IO.xlsx', sheet_name="sector_name")
    if nb_sectors!='all':
        dic = table.set_index("sector_id")['sector_name'].iloc[:nb_sectors].to_dict()
    else:
        dic = table.set_index("sector_id")['sector_name'].to_dict()
    dic[999] = 'Total'
    dic['transit'] = 'Transit'
    return dic


def load_firm_data():
    return pd.read_excel('input_firms.xlsx')


def congestion_function(current_traffic, normal_traffic):
    if (current_traffic==0) & (normal_traffic==0):
        return 0
        
    elif (current_traffic>0) & (normal_traffic==0):
        return 0.5
    
    elif (current_traffic==0) & (normal_traffic>0):
        return 0
    
    elif (current_traffic < normal_traffic):
        return 0
    
    elif (current_traffic < 1.5*normal_traffic):
        return 0
    else:
        excess_traffic = current_traffic - 1.5*normal_traffic
        return 4 * (1 - math.exp(-(excess_traffic)))
        

def production_function(inputs, input_mix, function_type="Leontief"):
    # Leontief
    if function_type == "Leontief":
        try:
            return min([inputs[input_id] / input_mix[input_id] for input_id, val in input_mix.items()])
        except KeyError:
            return 0
            
    else:
        raise ValueError("Wrong mode selected")

        
def purchase_planning_function(estimated_need, inventory, safety_days=1, reactivity_rate=1):
    """Decide the quantity of each input to buy according to a dynamical rule
    """
    target_inventory = (1 + safety_days) * estimated_need
    if inventory >= target_inventory + estimated_need:
        return 0
    elif inventory >= target_inventory:
        return target_inventory + estimated_need - inventory
    else:
        return (1 - reactivity_rate) * estimated_need + reactivity_rate * (estimated_need + target_inventory - inventory)

    
def evaluate_safety_days(estimated_need, inventory):
    if estimated_need == 0:
        return None
    else:
        return inventory / estimated_need - 1


def set_initial_conditions(graph, firm_list, households, country_list, mode="equilibrium"):
    households.reset_variables()
    for edge in graph.in_edges(households):
        graph[edge[0]][households]['object'].reset_variables()
    for firm in firm_list:
        firm.reset_variables()
        for edge in graph.in_edges(firm):
            graph[edge[0]][firm]['object'].reset_variables()
    for country in country_list:
        country.reset_variables()
        for edge in graph.in_edges(country):
            graph[edge[0]][country]['object'].reset_variables()
            
    if mode=="equilibrium":
        initilize_at_equilibrium(graph, firm_list, households, country_list)
    elif mode =="dynamic":
        initializeFirmsHouseholds(G, firm_list, households)
    else:
        print("Wrong mode chosen")


def initilize_at_equilibrium(graph, firm_list, households, country_list):
    firm_connectivity_matrix = nx.adjacency_matrix(graph.subgraph(list(graph.nodes)[:-1]), weight='weight', nodelist=firm_list).todense()
    import_weight_per_firm = [sum([graph[supply_edge[0]][supply_edge[1]]['weight'] for supply_edge in graph.in_edges(firm) if str(graph[supply_edge[0]][supply_edge[1]]['object'].supplier_id)[0] == 'C']) for firm in firm_list]
    n = len(firm_list)

    households.consumption = households.purchase_plan
    households.tot_consumption = households.final_demand * np.sum(list(households.purchase_plan.values()))
    households.spending = households.consumption
    households.tot_spending = households.tot_consumption
    households.extra_spending_per_sector = {key:0 for key, val in households.extra_spending_per_sector.items()}
    
    # Build final demand vector, of length n
    # Purchases made by other countries are considered as final demand
    final_demand_vector = np.zeros((n, 1))
    for firm_id, quantity in households.purchase_plan.items():
        final_demand_vector[(firm_id,0)] += quantity
    for country in country_list:
        for firm_id, quantity in country.purchase_plan.items():
            if type(firm_id) == str:
                continue #we dismiss transit flows
            else:
                final_demand_vector[(firm_id,0)] += quantity
    eq_production_vector = np.linalg.solve(np.eye(n) - firm_connectivity_matrix, final_demand_vector)

    households.send_purchase_orders(graph)
    for country in country_list:
        country.send_purchase_orders(graph)

    # Compute costs
    #input costs
    domestic_input_cost_vector = np.multiply(firm_connectivity_matrix.sum(axis=0).reshape((n,1)), eq_production_vector)
    import_input_cost_vector = np.multiply(np.array(import_weight_per_firm).reshape((n,1)), eq_production_vector)
    input_cost_vector = domestic_input_cost_vector + import_input_cost_vector
    #transport costs
    proportion_of_transport_cost_vector = 0.2*np.ones((n,1))
    transport_cost_vector = np.multiply(eq_production_vector, proportion_of_transport_cost_vector)
    #compute other costs based on margin
    margin = np.array([firm.target_margin for firm in firm_list]).reshape((n,1))
    other_cost_vector = np.multiply(eq_production_vector, (1-margin)) - input_cost_vector - transport_cost_vector
    
    for firm in firm_list:
        firm.production_target = eq_production_vector[(firm.pid, 0)]
        firm.production = firm.production_target
        firm.eq_production_capacity = firm.production_target / firm.utilization_rate
        firm.production_capacity = firm.eq_production_capacity
        firm.evaluate_input_needs()
        firm.eq_needs = firm.input_needs
        firm.inventory = {input_id: need * (1+firm.safety_days[input_id]) for input_id, need in firm.input_needs.items()}
        firm.decide_purchase_plan()
        firm.send_purchase_orders(graph)
        firm.eq_finance['sales'] = firm.production
        firm.eq_finance['costs']['input'] = input_cost_vector[(firm.pid,0)]
        firm.eq_finance['costs']['transport'] = transport_cost_vector[(firm.pid,0)]
        firm.eq_finance['costs']['other'] = other_cost_vector[(firm.pid,0)]
        firm.eq_profit = firm.eq_finance['sales'] - sum(firm.eq_finance['costs'].values())
        firm.finance['sales'] = firm.eq_finance['sales']
        firm.finance['costs']['input'] = firm.eq_finance['costs']['input']
        firm.finance['costs']['transport'] = firm.eq_finance['costs']['transport']
        firm.finance['costs']['other'] = firm.eq_finance['costs']['other']
        firm.profit = firm.eq_profit
        firm.delta_price_input = 0
        #the following is just to set once for all the share of sales of each client
    for firm in firm_list:
        firm.retrieve_orders(graph)
        firm.aggregate_orders()
        firm.eq_total_order = firm.total_order
        firm.calculate_client_share_in_sales()
        
    # Reset price to 1
    reset_prices(graph)

        
def reset_prices(graph):
    # set prices to 1
    for edge in graph.edges:
        graph[edge[0]][edge[1]]['object'].price = 1
            
            
def generate_weights(nb_values):
    rdm_values = np.random.uniform(0,1, size=nb_values)
    return list(rdm_values / sum(rdm_values))

def generate_weights_from_list(list_nb):
    sum_list = sum(list_nb)
    return [nb/sum_list for nb in list_nb]

def allFirmsSendPurchaseOrders(G, firm_list):
    for firm in firm_list:
        firm.send_purchase_orders(G)
        
        
def allAgentsSendPurchaseOrders(G, firm_list, households, country_list):
    households.send_purchase_orders(G)
    for firm in firm_list:
        firm.send_purchase_orders(G)
    for country in country_list:
        country.send_purchase_orders(G)
        
        
def allFirmsRetrieveOrders(G, firm_list):
    for firm in firm_list:
        firm.retrieve_orders(G)
        
def allFirmsPlanProduction(firm_list, graph, price_fct_input=True):
    for firm in firm_list:
        firm.aggregate_orders()
        firm.decide_production_plan()
        if price_fct_input:
            firm.calculate_price(graph, firm_list)
        
def allFirmsPlanPurchase(firm_list):
    for firm in firm_list:
        firm.evaluate_input_needs()
        firm.decide_purchase_plan() #mode="reactive"
        
        
def initializeFirmsHouseholds(G, firm_list, households):
    '''For dynamic initialization'''
    # Initialize dictionary
    for firm in firm_list:
        firm.inventory = {input_id: 0 for input_id, mix in firm.input_mix.items()}
        firm.input_needs = firm.inventory
    # Initialize orders
    households.send_purchase_orders(G)
    allFirmsRetrieveOrders(G, firm_list)
    allFirmsPlanProduction(firm_list, G)
    allFirmsPlanPurchase(firm_list)
    for i in range(0,10):
        allAgentsSendPurchaseOrders(G, firm_list, country_list)
        allFirmsRetrieveOrders(G, firm_list)
        allFirmsPlanProduction(firm_list, G)
        allFirmsPlanPurchase(firm_list)
    # Initialize inventories
    for firm in firm_list:
        firm.inventory = firm.input_needs
    # Initialize production plan
    for firm in firm_list:
        firm.production_target = firm.total_order

        
def allFirmsProduce(firm_list):
    for firm in firm_list:
        firm.produce()
        
def allFirmsDeliver(G, firm_list, T, rationing_mode):
    for firm in firm_list:
        firm.deliver_products(G, T, rationing_mode=rationing_mode)

def allAgentsDeliver(G, firm_list, country_list, T, rationing_mode):
    for firm in firm_list:
        firm.deliver_products(G, T, rationing_mode)
    for country in country_list:
        country.deliver_products(G, T)
        
        
def allAgentsReceiveProducts(G, firm_list, households, country_list, T):
    for firm in firm_list:
        firm.receive_products_and_pay(G, T)
    households.receive_products_and_pay(G)
    for country in country_list:
        country.receive_products_and_pay(G, T)
    for firm in firm_list:
        firm.evaluate_profit(G)
        
        
def allAgentsPrintInfo(firm_list, households):
    for firm in firm_list:
        firm.print_info()
    households.print_info()

    
def rescale_values(input_list, minimum=0.1, maximum=1, max_val=None, alpha=1):
    max_val = max_val or max(input_list)
    min_val = min(input_list)
    if max_val == min_val:
        return [0.5 * maximum] * len(input_list)
    else:
        return [minimum + (((val - min_val) / (max_val - min_val))**alpha) * (maximum - minimum) for val in input_list]


def plot_firm_on_map(all_edges_data, all_nodes_data, firm_list):
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_aspect('equal')
    all_edges_data.plot(ax=ax, color='grey', zorder=0) # plot all roads
    
    firm_locations = pd.Series([firm.location for firm in firm_list], name="nb_firms")
    nb_firms_per_location = firm_locations.value_counts().reset_index().rename(columns={"index":"nodenumber"})
    nodes_to_plot = pd.merge(nb_firms_per_location, all_nodes_data, how='inner', on='nodenumber')[['nb_firms', 'geometry']].set_geometry('geometry')
    nodes_to_plot.plot(ax=ax, color='blue', markersize=rescale_values(nodes_to_plot['nb_firms'], 10, 30))
    plt.show()

    
def prepare_tables_for_plotting(transport_network, firm_list):
    transport_edges = gpd.GeoDataFrame(data={
        'type': list(nx.get_edge_attributes(transport_network, "type").values()),
        'link': list(nx.get_edge_attributes(transport_network, "link").values()),
        'node_tuple': list(nx.get_edge_attributes(transport_network, "node_tuple").values()), 
        'geometry': list(nx.get_edge_attributes(transport_network, "geometry").values()) 
    }).set_geometry('geometry')
    
    transport_nodes = gpd.GeoDataFrame(data={
        'type': list(nx.get_node_attributes(transport_network, "type").values()),
        'nodenumber': list(nx.get_node_attributes(transport_network, "nodenumber").values()), 
        'geometry': list(nx.get_node_attributes(transport_network, "geometry").values())
    }).set_geometry('geometry')

    firm_nodes = gpd.GeoDataFrame(data={
        'pid': [firm.pid for firm in firm_list], 
        'geometry': [firm.geometry for firm in firm_list]
    }).set_geometry('geometry')
    
    return {'transport_edges':transport_edges, 'transport_nodes':transport_nodes, 'firm_nodes':firm_nodes}


def generate_basemap(transport_network, firm_list, plot_size, plot_firms=True):
    basemap = prepare_tables_for_plotting(transport_network, firm_list)
    fig, ax = plt.subplots(figsize=(plot_size, plot_size))
    ax.set_aspect('equal')
    basemap['transport_edges'].plot(ax=ax, color='lightgrey', zorder=0) # plot all roads
    if plot_firms:
        basemap['firm_nodes'].plot(ax=ax, color='blue', markersize=30, zorder=100)
    return fig, ax, basemap


def compute_distance(x0, y0, x1, y1):
    return math.sqrt((x1-x0)**2+(y1-y0)**2)


def compute_distance_from_arcmin(x0, y0, x1, y1):
    EW_dist = (x1-x0)*112.5
    NS_dist = (y1-y0)*111
    return math.sqrt(EW_dist**2+NS_dist**2)
    
    
def extract_country_transit_point(country_data, country_id):
    transit_point = country_data.loc[country_id, 'transit_point'].split()
    transit_point = pd.Series(transit_point).str.extract('([0-9]*)\:').iloc[:,0].tolist()
    transit_point = [int(nodenumber) for nodenumber in transit_point]
    return transit_point


def evaluate_sectoral_shock(firm_table, disrupted_node):
    disrupted_sectoral_production = firm_table[firm_table['location'].isin(disrupted_node)].groupby('sector_id')['total_production'].sum()
    normal_sectoral_production = firm_table.groupby('sector_id')['total_production'].sum()
    consolidated_table = pd.concat([disrupted_sectoral_production, normal_sectoral_production], axis=1).fillna(0)
    return consolidated_table.iloc[:,0] / consolidated_table.iloc[:,1]
    

def apply_sectoral_shocks(sectoral_shock, firm_list):
    for firm in firm_list:
        if firm.sector in sectoral_shock.index:
            firm.production_capacity = firm.production_target * (1 - sectoral_shock[firm.sector])


def recover_from_sectoral_shocks(firm_list):
    for firm in firm_list:
        firm.production_capacity = firm.eq_production_capacity
