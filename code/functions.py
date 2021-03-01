import numpy as np
import networkx as nx
import pandas as pd
import geopandas as gpd
import math
import json
import os
import random
import logging


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


def set_initial_conditions(graph, firm_list, household_list, country_list, mode="equilibrium"):
    # Reset the variables of all agents, and those of their input commercial links
    for household in firm_list:
        household.reset_variables()
        for edge in graph.in_edges(household):
            graph[edge[0]][household]['object'].reset_variables()
    for firm in firm_list:
        firm.reset_variables()
        for edge in graph.in_edges(firm):
            graph[edge[0]][firm]['object'].reset_variables()
    for country in country_list:
        country.reset_variables()
        for edge in graph.in_edges(country):
            graph[edge[0]][country]['object'].reset_variables()
            
    if mode=="equilibrium":
        initilize_at_equilibrium(graph, firm_list, household_list, country_list)
    # elif mode =="dynamic": #deprecated
    #     initializeFirmsHouseholds(G, firm_list, household_list)
    else:
        print("Wrong mode chosen")



def build_final_demand_vector(household_list, country_list, firm_list):
    '''Create a numpy.Array of the final demand per firm, including exports

    Households and countries should already have set their purchase plan

    Returns
    -------
    numpy.Array of dimension (len(firm_list), 1)
    '''
    final_demand_vector = np.zeros((len(firm_list), 1))

    # Collect households final demand. They buy only from firms.
    for household in household_list:
        for retailer_id, quantity in household.purchase_plan.items():
            final_demand_vector[(retailer_id,0)] += quantity

    # Collect country final demand. They buy from firms and countries.
    # We need to filter the demand directed to firms only.
    for country in country_list:
        for supplier_id, quantity in country.purchase_plan.items():
            if isinstance(supplier_id, int): # we only consider purchase from firms, not from other countries
                final_demand_vector[(supplier_id,0)] += quantity

    return final_demand_vector


"""def buildFinalDemandVector(household, country_list, firm_list):
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
"""

def initilize_at_equilibrium(graph, firm_list, household_list, country_list):
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
    final_demand_vector = build_final_demand_vector(household_list, country_list, firm_list)

    # Solve the input--output equation
    eq_production_vector = np.linalg.solve(
        np.eye(n) - firm_connectivity_matrix, 
        final_demand_vector
    )

    # Initialize households variables
    for household in household_list:
        household.initialize_var_on_purchase_plan()

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
    proportion_of_transport_cost_vector = 0.2*np.ones((n,1)) #XXX
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
    for household in household_list:
        household.send_purchase_orders(graph)
    for country in country_list:
        country.send_purchase_orders(graph)
    for firm in firm_list:
        firm.send_purchase_orders(graph)
    ##the following is just to set once for all the share of sales of each client
    for firm in firm_list:
        firm.retrieve_orders(graph)
        firm.aggregate_orders(print_info=True)
        firm.eq_total_order = firm.total_order
        firm.calculate_client_share_in_sales()
        
    ## Set price to 1
    reset_prices(graph)

        
def reset_prices(graph):
    # set prices to 1
    for edge in graph.edges:
        graph[edge[0]][edge[1]]['object'].price = 1
            
            
def generate_weights(nb_suppliers, importance_of_each=None):
    # if there is only one supplier, retunr 1
    if nb_suppliers == 1:
        return [1]

    # if there are several and importance are provided, choose according to importance
    if importance_of_each:
        return [x/sum(importance_of_each) for x in importance_of_each]

    # otherwise choose random values
    else:
        rdm_values = np.random.uniform(0,1, size=nb_suppliers)
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
        
        
def allAgentsSendPurchaseOrders(G, firm_list, household_list, country_list):
    for household in household_list:
        household.send_purchase_orders(G)
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
    for country in country_list:
        country.deliver_products(G, T,
            monetary_unit_transport_cost="USD", monetary_unit_flow="mUSD", 
            cost_repercussion_mode=cost_repercussion_mode, 
            explicit_service_firm=explicit_service_firm)
    for firm in firm_list:
        firm.deliver_products(G, T, rationing_mode,
            monetary_unit_transport_cost="USD", monetary_unit_flow="mUSD", 
            cost_repercussion_mode=cost_repercussion_mode, 
            explicit_service_firm=explicit_service_firm)
        
        
def allAgentsReceiveProducts(G, firm_list, household_list, country_list, T):
    for household in household_list:
        household.receive_products_and_pay(G, T)
    for firm in firm_list:
        firm.receive_products_and_pay(G, T)
    for country in country_list:
        country.receive_products_and_pay(G, T)
    for firm in firm_list:
        firm.evaluate_profit(G)
        
        
def allAgentsPrintInfo(firm_list, households):
    for firm in firm_list:
        firm.print_info()
    households.print_info()

    
def rescale_values(input_list, minimum=0.1, maximum=1, max_val=None, alpha=1, normalize=False):
    max_val = max_val or max(input_list)
    min_val = min(input_list)
    if max_val == min_val:
        res = [0.5 * maximum] * len(input_list)
    else:
        res = [
            minimum + (((val - min_val) / (max_val - min_val))**alpha) * (maximum - minimum) 
            for val in input_list
        ]
    if normalize:
        res = [x / sum(res) for x in res]
    return res


def compute_distance(x0, y0, x1, y1):
    return math.sqrt((x1-x0)**2+(y1-y0)**2)


def compute_distance_from_arcmin(x0, y0, x1, y1):
    # This is a very approximate way to convert arc distance into km
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


def transformUSDtoTons(monetary_flow, monetary_unit, usd_per_ton):
    # Load monetary units
    monetary_unit_factor = {
        "mUSD": 1e6,
        "kUSD": 1e3,
        "USD": 1
    }
    factor = monetary_unit_factor[monetary_unit]

    #sector_to_usdPerTon = sector_table.set_index('sector')['usd_per_ton']

    return monetary_flow / (usd_per_ton/factor)


def calculate_distance_between_agents(agentA, agentB):
    if (agentA.odpoint == -1) or (agentB.odpoint == -1):
        logging.warning("Try to calculate distance between agents, but one of them does not have real odpoint")
        return 1
    else:
        return compute_distance_from_arcmin(agentA.long, agentA.lat, agentB.long, agentB.lat)


def determine_nb_suppliers(nb_suppliers_per_input, max_nb_of_suppliers=None):
    '''Draw 1 or 2 depending on the 'nb_suppliers_per_input' parameters

    nb_suppliers_per_input is a float number between 1 and 2

    max_nb_of_suppliers: maximum value not to exceed
    '''
    if (nb_suppliers_per_input < 1) or (nb_suppliers_per_input > 2):
        raise ValueError("'nb_suppliers_per_input' should be between 1 and 2")

    if nb_suppliers_per_input == 1:
        nb_suppliers = 1

    elif nb_suppliers_per_input == 2:
        nb_suppliers = 2

    else:
        if random.uniform(0,1) < nb_suppliers_per_input-1:
            nb_suppliers = 2
        else:
            nb_suppliers = 1

    if max_nb_of_suppliers:
        nb_suppliers = min(nb_suppliers, max_nb_of_suppliers)

    return nb_suppliers


def select_supplier_from_list(agent, firm_list, 
    nb_suppliers_to_choose, potential_firm_ids, 
    distance, importance, weight_localization,
    force_same_odpoint=False):
    # reduce firm to choose to local ones
    if force_same_odpoint:
        same_odpoint_firms = [
            firm_id 
            for firm_id in potential_firm_ids 
            if firm_list[firm_id].odpoint == agent.odpoint
        ]
        if len(same_odpoint_firms) > 0:
            potential_firm_ids = same_odpoint_firms
        #     logging.info('retailer available locally at odpoint '+str(agent.odpoint)+
        #         " for "+firm_list[potential_firm_ids[0]].sector)
        # else:
        #     logging.warning('force_same_odpoint but no retailer available at odpoint '+str(agent.odpoint)+
        #         " for "+firm_list[potential_firm_ids[0]].sector)

    # distance weight
    if distance:
        distance_to_each = rescale_values([
            calculate_distance_between_agents(agent, firm_list[firm_id]) 
            for firm_id in potential_firm_ids
        ])
        distance_weight = 1 / (np.array(distance_to_each)**weight_localization)

    # importance weight
    if importance:
        importance_of_each = rescale_values([firm_list[firm_id].importance for firm_id in potential_firm_ids])
        importance_weight = np.array(importance_of_each)

    # create weight vector based on choice
    if importance and distance:
        prob_to_be_selected = importance_weight * distance_weight
    elif importance and not distance:
        prob_to_be_selected = importance_weight
    elif not importance and distance:
        prob_to_be_selected = distance_weight
    else:
        prob_to_be_selected = np.ones((1,len(potential_firm_ids)))
    prob_to_be_selected /= prob_to_be_selected.sum()

    # perform the random choice
    selected_supplier_id = np.random.choice(
        potential_firm_ids, 
        p=prob_to_be_selected, 
        size=nb_suppliers_to_choose, 
        replace=False
    ).tolist()
    # Choose weight if there are multiple suppliers
    if importance:
        supplier_weights = generate_weights(nb_suppliers_to_choose, importance_of_each)
    else:
        supplier_weights = generate_weights(nb_suppliers_to_choose)


    # return
    return selected_supplier_id, supplier_weights


def agent_decide_initial_routes(agent, graph, transport_network, transport_modes,
        account_capacity, monetary_unit_flow):
    for edge in graph.out_edges(agent):
        if edge[1].pid == -1: # we do not create route for households
            continue
        elif edge[1].odpoint == -1: # we do not create route for service firms if explicit_service_firms = False
            continue
        else:
            # Get the id of the orign and destination node
            origin_node = agent.odpoint
            destination_node = edge[1].odpoint
            # Define the type of transport mode to use by looking in the transport_mode table
            if agent.agent_type == 'firm':
                cond_from = (transport_modes['from'] == "domestic")
            elif agent.agent_type == 'country':
                cond_from = (transport_modes['from'] == agent.pid)
            else:
                raise ValueError("'agent' must be a Firm or a Country")
            if edge[1].agent_type in['firm', 'household']: #see what is the other end
                cond_to = (transport_modes['to'] == "domestic")
            elif edge[1].agent_type == 'country':
                cond_to = (transport_modes['to'] == edge[1].pid)
            else:
                raise ValueError("'edge[1]' must be a Firm or a Country")
                # we have not implemented the "sector" condition
            transport_mode = transport_modes.loc[cond_from & cond_to, "transport_mode"].iloc[0]
            graph[agent][edge[1]]['object'].transport_mode = transport_mode
            # Choose the route and the corresponding mode
            route, selected_mode = agent.choose_route(
                transport_network=transport_network, 
                origin_node=origin_node, 
                destination_node=destination_node, 
                possible_transport_modes=transport_mode
            )
            # Store it into commercial link object
            graph[agent][edge[1]]['object'].storeRouteInformation(
                route=route,
                transport_mode=selected_mode,
                main_or_alternative="main",
                transport_network=transport_network
            )
            # Update the "current load" on the transport network
            # if current_load exceed burden, then add burden to the weight
            if account_capacity:
                new_load_in_usd = graph[agent][edge[1]]['object'].order
                new_load_in_tons = transformUSDtoTons(new_load_in_usd, monetary_unit_flow, agent.usd_per_ton)
                transport_network.update_load_on_route(route, new_load_in_tons)


        
def agent_receive_products_and_pay(agent, graph, transport_network):
    # reset variable
    if agent.agent_type == 'country':
        agent.extra_spending = 0
        agent.consumption_loss = 0
    elif agent.agent_type == 'household':
        agent.reset_variables()

    # for each incoming link, receive product and pay
    # the way differs between service and shipment
    for edge in graph.in_edges(agent): 
        if graph[edge[0]][agent]['object'].product_type in ['services', 'utility', 'transport']:
            agent_receive_service_and_pay(agent, graph[edge[0]][agent]['object'])
        else:
            agent_receive_shipment_and_pay(agent, graph[edge[0]][agent]['object'], transport_network)


def agent_receive_service_and_pay(agent, commercial_link):
    # Always available, same price
    quantity_delivered = commercial_link.delivery
    commercial_link.payment = quantity_delivered * commercial_link.price
    if agent.agent_type == 'firm':
        agent.inventory[commercial_link.product] += quantity_delivered
    # Update indicator
    agent_update_indicator(agent, quantity_delivered, commercial_link.price, commercial_link)


def agent_update_indicator(agent, quantity_delivered, price, commercial_link):
    """When receiving product, agents update some internal variables

    Parameters
    ----------
    """
    if agent.agent_type == "country":
        agent.extra_spending += quantity_delivered * (price - commercial_link.eq_price)
        agent.consumption_loss += commercial_link.delivery - quantity_delivered

    elif agent.agent_type == 'household':
        agent.consumption_per_retailer[commercial_link.supplier_id] = quantity_delivered
        agent.tot_consumption += quantity_delivered
        agent.spending_per_retailer[commercial_link.supplier_id] = quantity_delivered * price
        agent.tot_spending += quantity_delivered * price
        agent.extra_spending += quantity_delivered * (price - commercial_link.eq_price)
        agent.consumption_loss = (agent.purchase_plan[commercial_link.supplier_id] - quantity_delivered) * \
                    commercial_link.eq_price
        # if consum_loss >= 1e-6:
        #     logging.debug("Household "+agent.pid+" Firm "+
        #         str(commercial_link.supplier_id)+" supposed to deliver "+
        #         str(agent.purchase_plan[commercial_link.supplier_id])+
        #         " but delivered "+str(quantity_delivered)
        #     )
    # Log if quantity received differs from what it was supposed to be
    if abs(commercial_link.delivery - quantity_delivered) > 1e-6:
        logging.debug("Agent "+str(agent.pid)+": quantity delivered by "+
            str(commercial_link.supplier_id)+" is "+str(quantity_delivered)+
            ". It was supposed to be "+str(commercial_link.delivery)+".")



def agent_receive_shipment_and_pay(agent, commercial_link, transport_network):
    """Firm look for shipments in the transport nodes it is located
    It takes those which correspond to the commercial link 
    It receives them, thereby removing them from the transport network
    Then it pays the corresponding supplier along the commecial link
    """
    # Look at available shipment
    available_shipments = transport_network.node[agent.odpoint]['shipments']
    if commercial_link.pid in available_shipments.keys():
        # Identify shipment
        shipment = available_shipments[commercial_link.pid]
        # Get quantity and price
        quantity_delivered = shipment['quantity']
        price = shipment['price']
        # Remove shipment from transport
        transport_network.remove_shipment(commercial_link)
        # Make payment
        commercial_link.payment = quantity_delivered * price
        # If firm, add to inventory
        if agent.agent_type == 'firm':
            agent.inventory[commercial_link.product] += quantity_delivered

    # If none is available, log it
    else:
        logging.debug("Agent "+str(agent.pid)+
            ": no shipment available for commercial link "+
            str(commercial_link.pid)+' ('+commercial_link.product+')'
        )
        quantity_delivered = 0
        price = 1

    agent_update_indicator(agent, quantity_delivered, price, commercial_link)



def agent_receive_shipment_and_payOLD(agent, commercial_link, transport_network):
    """Firm look for shipments in the transport nodes it is located
    It takes those which correspond to the commercial link 
    It receives them, thereby removing them from the transport network
    Then it pays the corresponding supplier along the commecial link
    """
    # quantity_intransit = commercial_link.delivery
    # Get delivery and update price
    quantity_delivered = 0
    price = 1
    if commercial_link.pid in transport_network.node[agent.odpoint]['shipments'].keys():
        quantity_delivered += transport_network.node[agent.odpoint]['shipments'][commercial_link.pid]['quantity']
        price = transport_network.node[agent.odpoint]['shipments'][commercial_link.pid]['price']
        transport_network.remove_shipment(commercial_link)

    if agent.agent_type == 'firm':
        # Add to inventory
        agent.inventory[commercial_link.product] += quantity_delivered

    elif agent.agent_type == "country":
        # Increment extra spending
        agent.extra_spending += quantity_delivered * (price - commercial_link.eq_price)
        # Increment consumption loss
        agent.consumption_loss += commercial_link.delivery - quantity_delivered

    elif agent.agent_type == 'household':
        agent.consumption[commercial_link.supplier_id] = quantity_delivered
        agent.tot_consumption += quantity_delivered
        agent.spending[commercial_link.supplier_id] = quantity_delivered * commercial_link.price
        agent.tot_spending += quantity_delivered * commercial_link.price
        agent.extra_spending += quantity_delivered * (commercial_link.price - commercial_link.eq_price)
        consum_loss = (agent.purchase_plan[commercial_link.supplier_id] - quantity_delivered) * \
                    commercial_link.eq_price
        agent.consumption_loss += consum_loss
        if consum_loss >= 1e-6:
            logging.debug("Household "+agent.pid+" Firm "+
                str(commercial_link.supplier_id)+" supposed to deliver "+
                str(agent.purchase_plan[commercial_link.supplier_id])+
                " but delivered "+str(quantity_delivered)
            )

    # Log if quantity received differs from what it was supposed to be
    if abs(commercial_link.delivery - quantity_delivered) > 1e-6:
        logging.debug("Agent "+str(agent.pid)+": quantity delivered by "+
            str(commercial_link.supplier_id)+" is "+str(quantity_delivered)+
            ". It was supposed to be "+str(commercial_link.delivery)+".")

    # Make payment
    commercial_link.payment = quantity_delivered * price