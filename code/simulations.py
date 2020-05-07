# Core function of the simulation loop

import logging
from functions import *
from export_functions import *
from check_functions import *

def setInitialSCConditions(transport_network, sc_network, firm_list, 
    country_list, households, initialization_mode="equilibrium"):
    """
    Set the initial supply chain conditions and reinitialize the transport network at 0.

    Parameters
    ----------
    transport_network : TransportNetwork
        Transport network graph
    sc_network : networkx.DiGraph
        Supply chain network graph
    firm_list : list of Firms
        List of firms
    country_list : list of Countries
        List of countries
    households : Households
        Households

    Returns
    -------
    Nothing
    """
    logging.info("Setting initial supply-chain conditions")
    transport_network.reinitialize_flows_and_disruptions()
    set_initial_conditions(sc_network, firm_list, households, country_list,
        initialization_mode)
    logging.info("Initial supply-chain conditions set")


def runOneTimeStep(transport_network, sc_network, firm_list, 
    country_list, households, observer,
    propagate_input_price_change=True,
    rationing_mode="household_first",
    time_step=0,
    export_folder=None,
    export_flows=False, 
    flow_types_to_export=['total'],
    filepath_road_edges="",
    export_sc_flow_analysis=False, 
    export_agent_data=False):
    """
    Run one time step

    Parameters
    ----------
    transport_network : TransportNetwork
        Transport network graph
    sc_network : networkx.DiGraph
        Supply chain network graph
    firm_list : list of Firms
        List of firms
    country_list : list of Countries
        List of countries
    households : Households
        Households
    observer : Observer
        Observer
    propagate_input_price_change : Boolean
        Whether or not firms should readjust their price to changes in input prices
    rationing_mode : string
        How firms ration their clients if they cannot meet all demand. Possible values:
        - 'equal': all clients are equally rationned in proportion of their order
        - 'household_first': if the firm sells to both households and other firms, 
        then households are served first
    time_step : int
        Used by observer to know the time of its observation
    export_folder : string or None
        Where output are exported, if any


    Returns
    -------
    Nothing
    """

    allFirmsRetrieveOrders(sc_network, firm_list)

    allFirmsPlanProduction(firm_list, sc_network, price_fct_input=propagate_input_price_change)
    
    allFirmsPlanPurchase(firm_list)

    allAgentsSendPurchaseOrders(sc_network, firm_list, households, country_list)
    
    allFirmsProduce(firm_list)
    
    allAgentsDeliver(sc_network, firm_list, country_list, transport_network, 
        rationing_mode=rationing_mode)
    
    if export_flows: #should be done at this stage, while the goods are on their way
        transport_network.compute_flow_per_segment(flow_types_to_export)
        # obs.collect_transport_flows(T_noCountries, time_step=0, flow_types_to_export=flow_types_to_export)
        observer.collect_transport_flows(transport_network, 
            time_step=time_step, flow_types=flow_types_to_export)
        exportTransportFlows(observer, export_folder)
        exportTransportFlowsShp(observer, export_folder, time_step=time_step, 
            filepath_road_edges=filepath_road_edges)
    
    if export_sc_flow_analysis: #should be done at this stage, while the goods are on their way
        analyzeSupplyChainFlows(sc_network, firm_list, export_folder)
    
    allAgentsReceiveProducts(sc_network, firm_list, households, 
        country_list, transport_network)
    
    if export_agent_data:
        observer.collect_agent_data(firm_list, households, country_list, 
            time_step=time_step)
        exportAgentData(observer, export_folder)

    compareProductionPurchasePlans(firm_list, country_list, households)