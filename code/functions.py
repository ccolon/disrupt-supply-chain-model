import numpy as np
import networkx as nx
import pandas as pd
import geopandas as gpd
import math
import json
import os
import random

def identify_special_transport_nodes(transport_nodes, special):
    res = transport_nodes.dropna(subset=['special'])
    res = res.loc[res['special'].str.contains(special), "id"].tolist()
    return res


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

        
def purchase_planning_function(estimated_need, inventory, inventory_duration_target=1, reactivity_rate=1):
    """Decide the quantity of each input to buy according to a dynamical rule
    """
    target_inventory = (1 + inventory_duration_target) * estimated_need
    if inventory >= target_inventory + estimated_need:
        return 0
    elif inventory >= target_inventory:
        return target_inventory + estimated_need - inventory
    else:
        return (1 - reactivity_rate) * estimated_need + reactivity_rate * (estimated_need + target_inventory - inventory)

    
def evaluate_inventory_duration(estimated_need, inventory):
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



def buildFinalDemandVector(households, country_list, firm_list):
    '''Create a numpy.Array of the final demand per firm, including exports

    Households and countries should already have set their purchase plan

    Returns
    -------
    numpy.Array of dimension (len(firm_list), 1)
    '''
    if len(households.purchase_plan) == 0:
        raise ValueError('Households purchase plan is empty')

    if (len(country_list)>0) \
        & (sum([len(country.purchase_plan) for country in country_list]) == 0):
        raise ValueError('The purchase plan of all countries is empty')

    final_demand_vector = np.zeros((len(firm_list), 1))
    # Collect households final demand. They buy only from firms.
    for firm_id, quantity in households.purchase_plan.items(): 
        final_demand_vector[(firm_id,0)] += quantity
    # Collect country final demand. They buy from firms and countries.
    # We need to filter the demand directed to firms only.
    for country in country_list:
        for supplier_id, quantity in country.purchase_plan.items():
            if isinstance(supplier_id, int): # we only consider purchase from firms
                final_demand_vector[(supplier_id,0)] += quantity

    return final_demand_vector


def initilize_at_equilibrium(graph, firm_list, households, country_list):
    """Initialize the supply chain network at the input--output equilibrium

    We will use the matrix forms to solve the following equation for X (production):
    D + E + AX = X + I
    where:
        D: final demand from households
        E: exports
        I: imports
        X: firm productions
        A: the input-output matrix
    These vectors and matrices are in the firm-and-country space.

    Parameters
    ----------
    graph : NetworkX.DiGraph
        The supply chain network. Nodes are firms, countries, or households. 
        Edges are commercial links.

    firm_list : list of Firms
        List of firms

    households : Households
        households

    country_list : list of Countries
        List of countries

    Returns
    -------
    Nothing
    """

    # Get the wieghted connectivity matrix.
    # Weight is the sectoral technical coefficient, if there is only one supplier for the input
    # It there are several, the technical coefficient is multiplied by the share of input of
    # this type that the firm buys to this supplier.
    firm_connectivity_matrix = nx.adjacency_matrix(
        graph.subgraph(list(graph.nodes)[:-1]), 
        weight='weight', 
        nodelist=firm_list
    ).todense()
    # Imports are considered as "a sector". We get the weight per firm for these inputs.
    # !!! aren't I computing the same thing as the IMP tech coef?
    import_weight_per_firm = [
        sum([
            graph[supply_edge[0]][supply_edge[1]]['weight'] 
            for supply_edge in graph.in_edges(firm) 
            if graph[supply_edge[0]][supply_edge[1]]['object'].category == 'import'
        ]) 
        for firm in firm_list
    ]
    n = len(firm_list)

    # Build final demand vector per firm, of length n
    # Exports are considered as final demand
    final_demand_vector = buildFinalDemandVector(households, country_list, firm_list)

    # Solve the input--output equation
    eq_production_vector = np.linalg.solve(
        np.eye(n) - firm_connectivity_matrix, 
        final_demand_vector
    )

    # Initialize households variables
    households.initialize_var_on_purchase_plan()

    # Compute costs
    ## Input costs
    domestic_input_cost_vector = np.multiply(
        firm_connectivity_matrix.sum(axis=0).reshape((n,1)), 
        eq_production_vector
    )
    import_input_cost_vector = np.multiply(
        np.array(import_weight_per_firm).reshape((n,1)), 
        eq_production_vector
    )
    input_cost_vector = domestic_input_cost_vector + import_input_cost_vector
    ## Transport costs
    proportion_of_transport_cost_vector = 0.2*np.ones((n,1))
    transport_cost_vector = np.multiply(eq_production_vector, proportion_of_transport_cost_vector)
    ## Compute other costs based on margin
    margin = np.array([firm.target_margin for firm in firm_list]).reshape((n,1))
    other_cost_vector = np.multiply(eq_production_vector, (1-margin))\
         - input_cost_vector - transport_cost_vector
    
    # Based on these calculus, update agents variables
    ## Firm operational variables
    for firm in firm_list:
        firm.initialize_ope_var_using_eq_production(
            eq_production=eq_production_vector[(firm.pid, 0)]
        )
    ## Firm financial variables
    for firm in firm_list:
        firm.initialize_fin_var_using_eq_cost(
            eq_production=eq_production_vector[(firm.pid, 0)], 
            eq_input_cost=input_cost_vector[(firm.pid,0)],
            eq_transport_cost=transport_cost_vector[(firm.pid,0)], 
            eq_other_cost=other_cost_vector[(firm.pid,0)]
        )
    ## Commercial links: agents set their order
    households.send_purchase_orders(graph)
    for country in country_list:
        country.send_purchase_orders(graph)
    for firm in firm_list:
        firm.send_purchase_orders(graph)
    ##the following is just to set once for all the share of sales of each client
    for firm in firm_list:
        firm.retrieve_orders(graph)
        firm.aggregate_orders()
        firm.eq_total_order = firm.total_order
        firm.calculate_client_share_in_sales()
        
    ## Set price to 1
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


def determine_suppliers_and_weights(potential_supplier_pid,
    nb_selected_suppliers, firm_list, mode):
    
    # Get importance for each of them
    if "importance_export" in mode.keys():
        importance_of_each = rescale_values([
            firm_list[firm_pid].importance * mode['importance_export']['bonus']
            if firm_list[firm_pid].odpoint in mode['importance_export']['export_odpoints']
            else firm_list[firm_pid].importance
            for firm_pid in potential_supplier_pid
        ])
    elif "importance" in mode.keys():
        importance_of_each = rescale_values([
            firm_list[firm_pid].importance 
            for firm_pid in potential_supplier_pid
        ])

    # Select supplier
    prob_to_be_selected = np.array(importance_of_each)
    prob_to_be_selected /= prob_to_be_selected.sum()
    selected_supplier_ids = np.random.choice(potential_supplier_pid, 
        p=prob_to_be_selected, size=nb_selected_suppliers, replace=False).tolist()

    # Compute weights, based on importance only
    supplier_weights = generate_weights_from_list([
        firm_list[firm_pid].importance 
        for firm_pid in selected_supplier_ids
    ])

    return selected_supplier_ids, supplier_weights


def identify_firms_in_each_sector(firm_list):
    firm_id_each_sector = pd.DataFrame({
        'firm': [firm.pid for firm in firm_list],
        'sector': [firm.sector for firm in firm_list]})
    dic_sector_to_firmid = firm_id_each_sector\
        .groupby('sector')['firm']\
        .apply(lambda x: list(x))\
        .to_dict()
    return dic_sector_to_firmid


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
        
# def allFirmsDeliver(G, firm_list, T, rationing_mode, route_optimization_weight):
#     for firm in firm_list:
#         firm.deliver_products(G, T, rationing_mode, route_optimization_weight)

def allAgentsDeliver(G, firm_list, country_list, T, rationing_mode, explicit_service_firm,
    monetary_unit_transport_cost="USD", monetary_unit_flow="mUSD", cost_repercussion_mode="type1"):
    for firm in firm_list:
        firm.deliver_products(G, T, rationing_mode,
            monetary_unit_transport_cost="USD", monetary_unit_flow="mUSD", 
            cost_repercussion_mode=cost_repercussion_mode, 
            explicit_service_firm=explicit_service_firm)
    for country in country_list:
        country.deliver_products(G, T,
            monetary_unit_transport_cost="USD", monetary_unit_flow="mUSD", 
            cost_repercussion_mode=cost_repercussion_mode, 
            explicit_service_firm=explicit_service_firm)
        
        
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


def compute_distance(x0, y0, x1, y1):
    return math.sqrt((x1-x0)**2+(y1-y0)**2)


def compute_distance_from_arcmin(x0, y0, x1, y1):
    EW_dist = (x1-x0)*112.5
    NS_dist = (y1-y0)*111
    return math.sqrt(EW_dist**2+NS_dist**2)
    

def evaluate_sectoral_shock(firm_table, disrupted_node):
    disrupted_sectoral_production = firm_table[firm_table['odpoint'].isin(disrupted_node)].groupby('sector_id')['total_production'].sum()
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



