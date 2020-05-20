import geopandas as gpd
import pandas as pd
import logging
import os
import networkx as nx
import math
import numpy as np
import yaml

from class_firm import Firm
from class_households import Households
from class_transport_network import TransportNetwork
from class_country import Country


def createTransportNetwork(filepath_road_nodes, filepath_road_edges, 
    transport_params, filepath_extra_road_edges=None):
    """Create the transport network object

    It uses one shapefile for the nodes and another for the edges.
    Note that therea are strong constraints on these files, in particular on their attributes. 
    We can optionally use an additional edge shapefile, which contains extra road segments. Useful for scenario testing.

    Parameters
    ----------
    filepath_road_nodes : string
        Path of the shapefile of the road nodes 

    filepath_road_edges : string
        Path of the shapefile of the road edges 

    transport_params : dictionary
        Transport parameters. Should be in a specific format.

    filepath_extra_road_edges : string
        Path of the shapefile of any extra road edges to include. Default to None.

    Returns
    -------
    TransportNetwork
    """

    # Create the transport network object, and set the "unit_cost" parameter, which is the ton.km cost of transporting sth
    T = TransportNetwork()
    T.graph['unit_cost'] = transport_params['transport_cost_per_tonkm']
    
    # Load node and edge data
    road_nodes = gpd.read_file(filepath_road_nodes)
    road_nodes.index = road_nodes['id']
    road_edges = gpd.read_file(filepath_road_edges)
    road_edges.index = road_edges['id']

    # Check conformity
    if (road_nodes['id'].duplicated().sum()>0):
        raise ValueError('The following node ids are duplicated: '+
            road_nodes.loc[road_nodes['id'].duplicated(), "id"])
    if (road_edges['id'].duplicated().sum()>0):
        raise ValueError('The following edge ids are duplicated: '+
            road_edges.loc[road_edges['id'].duplicated(), "id"])
    edge_ends = set(road_edges['end1'].tolist()+road_edges['end2'].tolist())
    edge_ends_not_in_node_data = edge_ends - set(road_nodes['id'])
    if len(edge_ends_not_in_node_data)>0:
        raise KeyError("The following node id are given as 'end1' or 'end2' in edge data \
            but are not in the node data: "+str(edge_ends_not_in_node_data))

    # Add additional edges, if any
    if filepath_extra_road_edges is not None:
        new_road_edges = gpd.read_file(filepath_extra_road_edges)
        new_road_edges.index = new_road_edges['id']
        # new_road_edges.index = [road_edges.index.max()+1+item for item in list(range(new_road_edges.shape[0]))]
        road_edges = pd.concat([road_edges, new_road_edges])
        #road_edges = road_edges.append(new_road_edges.reindex(), verify_integrity=True)
    
    # Compute how much it costs to transport one USD worth of good on each road edge
    # compute travel time. Differentiate between paved and unpaved roads
    road_edges['travel_time'] = road_edges['km'] * (
            (road_edges['surface']=='unpaved') / transport_params['speeds']['road']['unpaved'] +\
            (road_edges['surface']=='paved') / transport_params['speeds']['road']['paved']
        )
    # compute cost of travel time
    road_edges['cost_travel_time'] = road_edges['travel_time'] * \
        transport_params["travel_cost_of_time"]
    # compute variability cost. Differentiate between paved and unpaved roads
    road_edges['cost_variability'] = road_edges['cost_travel_time'] * \
        transport_params['variability_coef'] * (
            (road_edges['surface']=='unpaved') * transport_params["variability"]['road']['unpaved'] +\
            (road_edges['surface']=='paved') * transport_params['variability']['road']['paved']
        )
    # finally, add up both cost to get the cost of time of each road edges
    road_edges['time_cost'] = road_edges['cost_travel_time'] + road_edges['cost_variability']
    # Load the nodes and edges on the transport network object
    for road in road_edges['id']:
        T.add_transport_edge_with_nodes(road, road_edges, road_nodes)

    return T

    
def filterSector(sector_table, cutoff=0.1, cutoff_type="percentage", 
    sectors_to_include="all"):
    """Filter the sector table to sector whose output is largen than the cutoff value

    Parameters
    ----------
    sector_table : pandas.DataFrame
        Sector table
    cutoff : float
        Cutoff value for selecting the sectors
        If cutoff_type="percentage", the sector's output divided by all sectors' output is used
        If cutoff_type="absolute", the sector's absolute output, in USD, is used
    sectors_to_include : list of string or 'all'
        list of the sectors preselected by the user. Default to "all"

    Returns
    -------
    list of filtered sectors
    """
    if cutoff_type == "percentage":
        rel_output = sector_table['output'] / sector_table['output'].sum()
        filtered_sectors = sector_table.loc[rel_output > cutoff, "sector"].tolist()

    elif cutoff_type == "absolute":
        filtered_sectors = sector_table.loc[sector_table['output'] > cutoff, "sector"].tolist()

    else:
        raise ValueError("cutoff type should be 'percentage' or 'absolute'")    

    if len(filtered_sectors) == 0:
        raise ValueError("The cutoff value is so high that it filtered out all sectors")

    if isinstance(sectors_to_include, list):
        if (len(set(sectors_to_include) - set(filtered_sectors)) > 0):
            selected_but_filteredout_sectors = list(set(sectors_to_include) - set(filtered_sectors))
            logging.info("The following sectors were specifically selected but were filtered out"+
                str(selected_but_filteredout_sectors))

        return list(set(sectors_to_include) & set(filtered_sectors))

    else:
        return filtered_sectors


    
def rescaleNbFirms(filepath_district_sector_importance, filepath_odpoints, 
    sector_table,
    district_sector_cutoff, nb_top_district_per_sector, 
    sectors_to_include="all", districts_to_include="all"):
    """Generate the firm data

    It uses the district_sector_importance table, the odpoints, the cutoff values to generate the list of firms.
    It generates a tuple of 3 pandas.DataFrame:
    - firm_table
    - odpoint_table
    - filter_district_sector_table

    Parameters
    ----------
    filepath_district_sector_importance : string
        Path for the district_sector_importance table
    filepath_odpoints : string
        Path for the odpoint table
    sector_table : pandas.DataFrame
        Sector table
    district_sector_cutoff : float
        Cutoff value for selecting the combination of district and sectors. 
        For agricultural sector it is divided by two.
    nb_top_district_per_sector : None or integer
        Nb of extra district to keep based on importance rank per sector
    sectors_to_include : list of string or 'all'
        list of the sectors to include. Default to "all"
    districts_to_include : list of string or 'all'
        list of the districts to include. Default to "all"

    Returns
    -------
    tuple(pandas.DataFrame, pandas.DataFrame, pandas.DataFrame)
    """

    # Load 
    table_district_sector_importance = pd.read_csv(filepath_district_sector_importance)

    # Filter out combination with 0 importance
    table_district_sector_importance = \
        table_district_sector_importance[table_district_sector_importance['importance']!=0]

    # Keep only selected sectors, if applicable
    if isinstance(sectors_to_include, list):
        table_district_sector_importance = \
            table_district_sector_importance[table_district_sector_importance['sector'].isin(sectors_to_include)]
    elif (sectors_to_include!='all'):
        raise ValueError("'sectors_to_include' should be a list of string or 'all'")

    # Keep only selected districts, if applicable
    if isinstance(districts_to_include, list):
        table_district_sector_importance = \
            table_district_sector_importance[table_district_sector_importance['district'].isin(districts_to_include)]
    elif (districts_to_include!='all'):
        raise ValueError("'districts_to_include' should be a list of string or 'all'")

    logging.info('Nb of combinations (district, sector) before cutoff: '+str(table_district_sector_importance.shape[0]))

    # Filter district-sector combination that are above the cutoff value
    agri_sectors = sector_table.loc[sector_table['type']=="agriculture", "sector"].tolist()
    if len(agri_sectors)>0:
        logging.info('Cutoff is '+str(district_sector_cutoff/2)+" for agriculture sectors, "+
            str(district_sector_cutoff)+" otherwise")
        boolindex_overthreshold = table_district_sector_importance['importance'] >= district_sector_cutoff
        boolindex_agri = (table_district_sector_importance['sector'].isin(agri_sectors)) & \
             (table_district_sector_importance['importance'] >= district_sector_cutoff/2)
        filtered_district_sector_table = table_district_sector_importance[boolindex_overthreshold | boolindex_agri].copy()
    else:
        logging.info('Cutoff is '+str(district_sector_cutoff))
        boolindex_overthreshold = table_district_sector_importance['importance']>= district_sector_cutoff
        filtered_district_sector_table = table_district_sector_importance[boolindex_overthreshold].copy()
    logging.info('Nb of combinations (district, sector) after cutoff: '+str(filtered_district_sector_table.shape[0]))

    # Add the top district of each sector
    if isinstance(nb_top_district_per_sector, int):
        if nb_top_district_per_sector > 0:
            top_district_sector = pd.concat([
                table_district_sector_importance[table_district_sector_importance['sector']==sector].nlargest(nb_top_district_per_sector, 'importance')
                for sector in table_district_sector_importance['sector'].unique()
            ])
            filtered_district_sector_table = pd.concat([filtered_district_sector_table, top_district_sector]).drop_duplicates()
        logging.info('Nb of combinations (district, sector) after adding top '+
            str(nb_top_district_per_sector)+": "+
            str(filtered_district_sector_table.shape[0]))
    
    # Generate the OD sector table
    # It only contains the OD points for the filtered district
    table_odpoints = pd.read_csv(filepath_odpoints)
    table_odpoints['nb_points_same_district'] = table_odpoints['district'].map(table_odpoints['district'].value_counts())
    od_sector_table = pd.merge(table_odpoints, filtered_district_sector_table, how='inner', on='district')
    od_sector_table['importance'] = od_sector_table['importance'] / od_sector_table['nb_points_same_district']
    logging.info('Initial nb of OD points: '+str(table_odpoints.shape[0]))
    logging.info('Filtered OD points: '+str(od_sector_table["odpoint"].nunique()))
    logging.info("Av OD point per district: {:.2g}".
        format(od_sector_table["odpoint"].nunique()/od_sector_table["district"].nunique()))
    logging.info("Av sector per OD point: {:.2g}".
        format(od_sector_table.shape[0]/od_sector_table["odpoint"].nunique()))
    
    # Create firm table
    # To generate the firm table, remove utilities, transport, and services
    # Should be changed!!! The flow of those firms should be virtual.
    # But they should be spatially represented! Because they buy real stuff!
    service_sectors = sector_table.loc[sector_table['type'].isin(
            ['utility', 'transport', 'services']
        ), 'sector'].tolist()
    service_sectors_present = od_sector_table.loc[
            od_sector_table['sector'].isin(service_sectors), 
            'sector'
        ].drop_duplicates().tolist()
    od_sector_table_noservice = od_sector_table[
        ~od_sector_table['sector'].isin(service_sectors_present)
    ]
    # We want two firms of the same sector in the same OD point,
    # so duplicates rows, generate a unique id
    firm_table = pd.concat(
        [od_sector_table_noservice, od_sector_table_noservice], 
        axis = 0, ignore_index=True
    )
    # Add utilities, transport, and services as virtuval firms
    if len(service_sectors_present)>1:
        firm_table_services = pd.DataFrame({
            'odpoint':-1,
            'importance': 1/2,
            "sector": service_sectors_present*2
        })
        firm_table = pd.concat([firm_table, firm_table_services], 
            axis=0, ignore_index=True, sort=True)

    # Format firm table
    firm_table = firm_table.sort_values('importance', ascending=False)
    firm_table = firm_table.sort_values(['district', 'sector'])
    firm_table['id'] = list(range(firm_table.shape[0]))
    firm_table = firm_table[['id', 'sector', 'odpoint', 'importance', 'district', 'long', 'lat']]
    
    # Create OD table
    od_table = od_sector_table.copy()
    od_table = od_table[['odpoint', 'district', 'nb_points_same_district', 'long', 'lat']]
    od_table = od_table.drop_duplicates().sort_values('odpoint')

    logging.info('Nb of od points chosen: '+str(len(set(firm_table['odpoint'])))+
        ', final nb of firms chosen: '+str(firm_table.shape[0]))
    # logging.debug(firm_table.groupby('sector')['id'].count())

    return firm_table, od_table, filtered_district_sector_table
    
    
    
def createFirms(firm_table, keep_top_n_firms=None, reactivity_rate=0.1, utilization_rate=0.8):
    """Create the firms

    It uses firm_table from rescaleNbFirms

    Parameters
    ----------
    firm_table: pandas.DataFrame
        firm_table from rescaleNbFirms

    keep_top_n_firms: None (default) or integer
        (optional) can be specified if we want to keep only the first n firms, for testing purposes

    reactivity_rate: float
        Determines the speed at which firms try to reach their inventory duration target. Default to 0.1.

    utilization_rate: float
        Set the utilization rate, which determines the production capacity at the input-output equilibrium.

    Returns
    -------
    list of Firms
    """

    if isinstance(keep_top_n_firms, int):
        firm_table = firm_table.iloc[:keep_top_n_firms,:]

    logging.debug('Creating firm_list')
    ids = firm_table['id'].tolist()
    firm_table = firm_table.set_index('id')
    firm_list= [
        Firm(i, 
             sector=firm_table.loc[i, "sector"], 
             odpoint=firm_table.loc[i, "odpoint"], 
             importance=firm_table.loc[i, 'importance'],
             # geometry=firm_table.loc[i, 'geometry'],
             long=float(firm_table.loc[i, 'long']),
             lat=float(firm_table.loc[i, 'lat']),
             utilization_rate=utilization_rate
        )
        for i in ids
    ]
    # We add a bit of noise to the long and lat coordinates
    # It allows to visually disentangle firms located at the same odpoint when plotting the map.
    for firm in firm_list:
        firm.add_noise_to_geometry()
        
    return firm_list
    
    
def loadTechnicalCoefficients(firm_list, filepath_tech_coef, io_cutoff=0.1, import_sector_name=None):
    """Load the input mix of the firms' Leontief function

    Parameters
    ----------
    firm_list : pandas.DataFrame
        the list of Firms generated from the createFirms function

    filepath_tech_coef : string
        Filepath to the matrix of technical coefficients

    io_cutoff : float
        Filters out technical coefficient below this cutoff value. Default to 0.1.

    imports : None or string
        Give the name of the import sector. If None, then the import technical coefficient is discarded. Default to None.
    
    Returns
    -------
    list of Firms
    """

    # Load technical coefficient matrix from data
    tech_coef_matrix = pd.read_csv(filepath_tech_coef, index_col=0)
    tech_coef_matrix = tech_coef_matrix.mask(tech_coef_matrix<=io_cutoff, 0)
    
    # We select only the technicial coefficient between sectors that are actually represented in the economy
    # Note that, when filtering out small sector-district combination, some sector may not be present.
    sector_present = list(set([firm.sector for firm in firm_list]))
    if import_sector_name:
        tech_coef_matrix = tech_coef_matrix.loc[sector_present + [import_sector_name], sector_present]
    else:
        tech_coef_matrix = tech_coef_matrix.loc[sector_present, sector_present]
    
    # Load input mix
    for firm in firm_list:
        firm.input_mix = tech_coef_matrix.loc[tech_coef_matrix.loc[:,firm.sector] != 0, firm.sector].to_dict()
    
    return firm_list
    


def loadInventories(firm_list, inventory_duration_target=2, 
    filepath_inventory_duration_targets=None, extra_inventory_target=None, 
    inputs_with_extra_inventories=None, buying_sectors_with_extra_inventories=None,
    min_inventory=1, random_mean_sd=None):
    """Load inventory duration target

    If inventory_duration_target is an integer, it is uniformly applied to all firms.
    If it its "inputed", then we use the targets defined in the file filepath_inventory_duration_targets. In that case,
    targets are sector specific, i.e., it varies according to the type of input and the sector of the buying firm.
    If both cases, we can add extra units of inventories:
    - uniformly, e.g., all firms have more inventories of all inputs,
    - to specific inputs, all firms have extra agricultural inputs,
    - to specific buying firms, e.g., all manufacturing firms have more of all inputs,
    - to a combination of both. e.g., all manufacturing firms have more of agricultural inputs.
    We can also add some noise on the distribution of inventories. Not yet imlemented.

    Parameters
    ----------
    firm_list : pandas.DataFrame
        the list of Firms generated from the createFirms function

    inventory_duration_target : "inputed" or integer
        Inventory duration target uniformly applied to all firms and all inputs.
        If 'inputed', uses the specific values from the file specified by 
        filepath_inventory_duration_targets

    extra_inventory_target : None or integer
        If specified, extra inventory duration target.

    inputs_with_extra_inventories : None or list of sector
        For which inputs do we add inventories.

    buying_sectors_with_extra_inventories : None or list of sector
        For which sector we add inventories.

    min_inventory : int
        Set a minimum inventory level

    random_mean_sd: None
        Not yet implemented.

    Returns
    -------
        list of Firms
    """

    if isinstance(inventory_duration_target, int):
        for firm in firm_list:
            firm.inventory_duration_target = {input_sector: inventory_duration_target for input_sector in firm.input_mix.keys()}
    
    elif inventory_duration_target=='inputed':
        dic_sector_inventory = pd.read_csv(filepath_inventory_duration_targets).set_index(['buying_sector', 'input_sector'])['inventory_duration_target'].to_dict()
        for firm in firm_list:
            firm.inventory_duration_target = {
                input_sector: dic_sector_inventory[(firm.sector, input_sector)]
                for input_sector in firm.input_mix.keys() 
            }

    else:
        raise ValueError("Unknown value entered for 'inventory_duration_target'")

    # if random_mean_sd:
    #     if random_draw:
    #         for firm in firm_list:
    #             firm.inventory_duration_target = {}
    #             for input_sector in firm.input_mix.keys():
    #                 mean = dic_sector_inventory[(firm.sector, input_sector)]['mean']
    #                 sd = dic_sector_inventory[(firm.sector, input_sector)]['sd']
    #                 mu = math.log(mean/math.sqrt(1+sd**2/mean**2))
    #                 sigma = math.sqrt(math.log(1+sd**2/mean**2))
    #                 safety_day = np.random.log(mu, sigma)
    #                 firm.inventory_duration_target[input_sector] = safety_day

    # Add extra inventories if needed. Not the best programming mabye...
    if isinstance(extra_inventory_target, int):
        if isinstance(inputs_with_extra_inventories, list) and (buying_sectors_with_extra_inventories=='all'):
            for firm in firm_list:
                firm.inventory_duration_target = {
                    input_sector: firm.inventory_duration_target[input_sector]+extra_inventory_target 
                    if (input_sector in inputs_with_extra_inventories) else firm.inventory_duration_target[input_sector]
                    for input_sector in firm.input_mix.keys() 
                }

        elif (inputs_with_extra_inventories=='all') and isinstance(buying_sectors_with_extra_inventories, list):
            for firm in firm_list:
                firm.inventory_duration_target = {
                    input_sector: firm.inventory_duration_target[input_sector]+extra_inventory_target 
                    if (firm.sector in buying_sectors_with_extra_inventories) else firm.inventory_duration_target[input_sector]
                    for input_sector in firm.input_mix.keys() 
                }

        elif isinstance(inputs_with_extra_inventories, list) and isinstance(buying_sectors_with_extra_inventories, list):
            for firm in firm_list:
                firm.inventory_duration_target = {
                    input_sector: firm.inventory_duration_target[input_sector]+extra_inventory_target 
                    if ((input_sector in inputs_with_extra_inventories) and (firm.sector in buying_sectors_with_extra_inventories)) else firm.inventory_duration_target[input_sector]
                    for input_sector in firm.input_mix.keys() 
                }

        elif (inputs_with_extra_inventories=='all') and (buying_sectors_with_extra_inventories=='all'):
            for firm in firm_list:
                firm.inventory_duration_target = {
                    input_sector: firm.inventory_duration_target[input_sector]+extra_inventory_target 
                    for input_sector in firm.input_mix.keys() 
                }

        else:
            raise ValueError("Unknown value given for 'inputs_with_extra_inventories' or 'buying_sectors_with_extra_inventories'.\
                Should be a list of string or 'all'")           

    if (min_inventory > 0):
        for firm in firm_list:
            firm.inventory_duration_target = {
                input_sector: max(min_inventory, inventory)
                for input_sector, inventory in firm.inventory_duration_target.items()
            }
    # inventory_table = pd.DataFrame({
    #     'id': [firm.pid for firm in firm_list],
    #     'buying_sector': [firm.sector for firm in firm_list],
    #     'inventories': [firm.inventory_duration_target for firm in firm_list]
    # })
    # inventory_table.to_csv('inventory_check.csv')
    # logging.info("Inventories: "+str({firm.pid: firm.inventory_duration_target for firm in firm_list}))
    return firm_list


def loadTonUsdEquivalence(sector_table, firm_list, country_list):
    """Load equivalence between usd and ton

    It updates the firm_list and country_list.
    It updates the 'usd_per_ton' attribute of firms, based on their sector.
    It updates the 'usd_per_ton' attribute of countries, it gives the average.
    Note that this will be applied only to goods that are delivered by those agents.

    sector_table : pandas.DataFrame
        Sector table
    firm_list : list(Firm objects)
        list of firms
    country_list : list(Country objects)
        list of countries

    :return: (list(Firm objects), list(Country objects))
    """
    sector_to_usdPerTon = sector_table.set_index('sector')['usd_per_ton']
    for firm in firm_list:
        firm.usd_per_ton = sector_to_usdPerTon[firm.sector]
    for country in country_list:
        country.usd_per_ton = sector_to_usdPerTon.mean()
    return firm_list, country_list


def rescaleMonetaryValues(values, time_resolution="week", target_units="mUSD", input_units="USD"):
    """Rescale monetary values using the approriate time scale and monetary units

    Parameters
    ----------
    values : pandas.Series, pandas.DataFrame, float
        Values to transform

    time_resolution : 'day', 'week', 'month', 'year'
        The number in the input table are yearly figure

    target_units : 'USD', 'kUSD', 'mUSD'
        Moneraty units to which valuess are converted

    input_units : 'USD', 'kUSD', 'mUSD'
        Moneraty units of the inputed values

    Returns
    -------
    same type as values
    """
    # Rescale according to the time period chosen
    periods = {'day': 365, 'week': 52, 'month': 12, 'year': 1}
    values = values / periods[time_resolution]

    # Change units
    units = {"USD": 1, "kUSD": 1e3, "mUSD":1e6}
    values = values * units[input_units] / units[target_units]
    
    return values



def createCountries(filepath_imports, filepath_exports, filepath_transit_matrix, filepath_entry_points,
    present_sectors, countries_to_include='all', 
    time_resolution="week", target_units="mUSD", input_units="USD"):
    """Create the countries

    Parameters
    ----------
    filepath_imports : string
        path to import table csv

    filepath_exports : string
        path to export table csv

    filepath_transit_matrix : string
        path to transit matrix csv

    filepath_entry_points : string
        path to the table of transit points csv

    present_sectors : list of string
        list which sectors are included. Output of the rescaleFirms functions.

    countries_to_include : list of string or "all"
        List of countries to include. Default to "all", which select all sectors.

    time_resolution : see rescaleMonetaryValues
    target_units : see rescaleMonetaryValues
    input_units : see rescaleMonetaryValues

    Returns
    -------
    list of Countries
    """
    import_table = rescaleMonetaryValues(
        pd.read_csv(filepath_imports, index_col=0),
        time_resolution=time_resolution,
        target_units=target_units,
        input_units=input_units
    )
    export_table = rescaleMonetaryValues(
        pd.read_csv(filepath_exports, index_col=0),
        time_resolution=time_resolution,
        target_units=target_units,
        input_units=input_units
    )
    transit_matrix = rescaleMonetaryValues(
        pd.read_csv(filepath_transit_matrix, index_col=0),
        time_resolution=time_resolution,
        target_units=target_units,
        input_units=input_units
    )
    entry_point_table = pd.read_csv(filepath_entry_points)

    # Keep only selected countries, if applicable
    if isinstance(countries_to_include, list):
        import_table = import_table.loc[countries_to_include, present_sectors]
        export_table = export_table.loc[countries_to_include, present_sectors]
        transit_matrix = transit_matrix.loc[countries_to_include, countries_to_include]
    elif (countries_to_include=='all'):
        import_table = import_table.loc[:, present_sectors]
        export_table = export_table.loc[:, present_sectors]
    else:
        raise ValueError("'countries_to_include' should be a list of string or 'all'")
    
    logging.info("Total imports per "+time_resolution+" is "+
        "{:.01f} ".format(import_table.sum().sum())+target_units)
    logging.info("Total exports per "+time_resolution+" is "+
        "{:.01f} ".format(export_table.sum().sum())+target_units)
    logging.info("Total transit per "+time_resolution+" is "+
        "{:.01f} ".format(transit_matrix.sum().sum())+target_units)

    country_list = []
    total_imports = import_table.sum().sum()
    for country in import_table.index.tolist():
        # transit points
        entry_points = entry_point_table.loc[
            entry_point_table['country']==country, 'entry_point'
        ].astype(int).tolist()

        # imports, i.e., sales of countries
        qty_sold = import_table.loc[country,:].to_dict()
        supply_importance = sum(qty_sold.values()) / total_imports
        
        # exports, i.e., purchases from countries
        qty_purchased = export_table.loc[country,:].to_dict()
        
        # transits
        # Note that transit are not given per sector, so, if we only consider a few sector, the full transit flows will still be used
        transit_from = transit_matrix.loc[:, country]
        transit_from = transit_from[transit_from>0].to_dict()
        transit_to = transit_matrix.loc[country, :]
        transit_to = transit_to[transit_to>0].to_dict()
        
        # create the list of Country object
        country_list += [Country(pid=country,
                                qty_sold=qty_sold,
                                qty_purchased=qty_purchased,
                                entry_points=entry_points,
                                transit_from=transit_from,
                                transit_to=transit_to,
                                supply_importance=supply_importance
                        )]
    return country_list


def defineFinalDemand(firm_table, od_table, 
    filepath_population, filepath_final_demand,
    time_resolution='week', target_units="mUSD", input_units="USD"):
    """Allocate a final demand to each firm. It updates the firm_table

    Parameters
    ----------
    firm_table : pandas.DataFrame
        firm_table from rescaleNbFirms function

    od_table: pandas.DataFrame
        od_table from rescaleNbFirms function

    filepath_population: string
        path to population csv

    filepath_final_demand : string
        path to final demand csv

    time_resolution : see rescaleMonetaryValues
    target_units : see rescaleMonetaryValues
    input_units : see rescaleMonetaryValues

    Returns
    -------
    firm_table
    """
    # Compute population allocated to each od point
    population_per_district = pd.read_csv(filepath_population)
    od_table = pd.merge(od_table, population_per_district, on='district', how='left')
    od_table['population'] = od_table['population'] / od_table['nb_points_same_district']
    od_table['perc_population'] = od_table['population'] / od_table['population'].sum()
    logging.info('Population in district with firms: '+str(int(od_table['population'].sum()))+', total population is: '+str(population_per_district['population'].sum()))
    
    # Compute population allocated to each firm
    firm_table = firm_table[['id', 'sector', 'odpoint', 'importance']].merge(od_table, 
                                                                      on='odpoint', 
                                                                      how='left')
    firm_table = firm_table.merge(firm_table.groupby(['odpoint', 'sector'])['id'].count().reset_index().rename(columns={'id':'nb_firms_same_point_same_sector'}),
                      on=['odpoint', 'sector'],
                      how='left')
    firm_table.loc[firm_table['odpoint']==-1, 'perc_population'] = 1
    firm_table['final_demand_weight'] = firm_table['perc_population'] / firm_table['nb_firms_same_point_same_sector']
    
    # Weight will not add up to 1 in some if not all sectors, because not all sectors are present in each od point. We renormalize.
    firm_table['final_demand_weight'] = firm_table['final_demand_weight'] / firm_table['sector'].map(firm_table.groupby('sector')['final_demand_weight'].sum())
    
    # Check that weight sum up to 1
    sum_of_sectoral_final_demand_weight = firm_table.groupby('sector')['final_demand_weight'].sum()
    if ((sum_of_sectoral_final_demand_weight-1.0)>1e-6).any():
        logging.warning('The final demand of some sectors is problematic: '+str(sum_of_sectoral_final_demand_weight))
    
    # Rescale final demand according to the time period and monetary unit chosen
    final_demand_per_sector = pd.read_csv(filepath_final_demand)
    final_demand_per_sector["final_demand"] = rescaleMonetaryValues(
        final_demand_per_sector["final_demand"],
        time_resolution=time_resolution,
        target_units=target_units,
        input_units=input_units
    )
    # Allocate actual final demand to each firm based on its weight
    firm_table['final_demand'] = firm_table['sector'].map(final_demand_per_sector.set_index('sector')['final_demand'])
    firm_table['final_demand'] = firm_table['final_demand'] * firm_table['final_demand_weight']
    logging.info("Total final demand per "+time_resolution+" is "+
        "{:.01f} ".format(firm_table['final_demand'].sum())+target_units)
    return firm_table
    

def createHouseholds(firm_table):
    """Create Households objecvt

    :param firm_table: firm_table from rescaleNbFirms and defineFinalDemand functions
    :type firm_table: pandas.DataFrame

    :return: Households object
    """
    households = Households()
    households.final_demand_per_sector = firm_table.groupby('sector')['final_demand'].sum().to_dict()
    households.purchase_plan = firm_table[['id', 'final_demand']].set_index('id')['final_demand'].to_dict()
    households.extra_spending_per_sector = {key: 0 for key, val in households.final_demand_per_sector.items()}
    return households
    

def extractEdgeList(graph):
    dic_commercial_links = nx.get_edge_attributes(graph, "object")
    dic_commercial_links = {key: value for key, value in dic_commercial_links.items() if (isinstance(key[0], Firm) and isinstance(key[1], Firm))}
    dic_commercial_links = [{
        'supplier_id':key[0].pid, 'buyer_id':key[1].pid,
        'supplier_odpoint':key[0].odpoint, 'buyer_odpoint':key[1].odpoint,
        'supplier_sector':key[0].sector, 'buyer_sector':key[1].sector, 
        'distance':key[0].distance_to_other(key[1]),
        'flow':commercial_link.order
    } for key, commercial_link in dic_commercial_links.items()]
    return dic_commercial_links


def defineDisruptionList(disruption_analysis, transport_network, 
    nodeedge_tested_topn=None, nodeedge_tested_skipn=None):
    """Create list of infrastructure to disrupt

    Parameters
    ----------
    disruption_analysis : None or dic
        See parameters_default commments

    transport_network : TransportNetwork object
        Transport network

    nodeedge_tested_topn : None or integer
        Nb of node/edge to test in the list. If None the full list is tested.

    nodeedge_tested_topn : None or integer
        Nb of node/edge to skip in the list.

    Returns
    -------
    list of node/edge ids
    """
    if disruption_analysis is None:
        raise ValueError("'disruption_analysis' is None, cannot define the disruption list")

    # if a list is directly provided, then it is the list
    if isinstance(disruption_analysis['nodeedge_tested'], list):
        disruption_list = disruption_analysis['nodeedge_tested']

    # if 'all' is indicated, need to retrieve all node or edge ids from the transport network
    elif disruption_analysis['nodeedge_tested'] == 'all':
        actual_transport_network =  transport_network.subgraph([
            node 
            for node in transport_network.nodes 
            if transport_network.nodes[node]['type']!='virtual'
            ])
        if disruption_analysis['disrupt_nodes_or_edges'] == "nodes":
            disruption_list = list(actual_transport_network.nodes)
        elif disruption_analysis['disrupt_nodes_or_edges'] == "edges":
            disruption_list = list(nx.get_edge_attributes(actual_transport_network, 'id').values())
        else:
            raise ValueError("disruption_analysis['disrupt_nodes_or_edges'] should be 'nodes' or 'edges'")

    # if a filepath is given, then load it. It is the list
    elif isinstance(disruption_analysis['nodeedge_tested'], str):
        if disruption_analysis['nodeedge_tested'][-4:] == ".csv":
            disruption_list = pd.read_csv(
                disruption_analysis['nodeedge_tested'], 
                header=None
            ).iloc[:,0].tolist()
        else:
            raise ValueError("If defining a path to a file in 'nodeedge_tested', it should be a csv")

    else:
        raise ValueError("'nodeedge_tested' should be list of node/edge ids, a path to a csv file, or 'all'")

    # keep only some
    if isinstance(nodeedge_tested_topn, int):
        disruption_list = disruption_list[:nodeedge_tested_topn]

    # skip some
    if isinstance(nodeedge_tested_skipn, int):
        disruption_list = disruption_list[nodeedge_tested_skipn:]

    # reformat disruption list. It is a list of dic
    # [{"node":[1,2,3], "edge"=[], "duration"=2}, {"node":[4], "edge"=[], "duration"=1}]
    if disruption_analysis['disrupt_nodes_or_edges'] == "nodes":
        disruption_list = [
            {"node":disrupted_stuff, "edge":[], "duration":disruption_analysis["duration"]}
            if isinstance(disrupted_stuff, list)
            else {"node":[disrupted_stuff], "edge":[], "duration":disruption_analysis["duration"]}
            for disrupted_stuff in disruption_list
        ]
    elif disruption_analysis['disrupt_nodes_or_edges'] == "edges":
        disruption_list = [
            {"node":[], "edge":disrupted_stuff, "duration":disruption_analysis["duration"]}
            if isinstance(disrupted_stuff, list)
            else {"node":[], "edge":[disrupted_stuff], "duration":disruption_analysis["duration"]}
            for disrupted_stuff in disruption_list
        ]

    return disruption_list


def exportSupplyChainNetworkSummary(sc_graph, firm_list, export_folder):
    """
    Export simple indicators of the supply chain network

    It exports a 'sc_network_summary.csv' file in the export_folder.

    Parameters
    ----------
    sc_graph : networkx.DiGraph
        Supply chain graph
    firm_list : list of Firm
        Generated by createFirm function
    export_folder : string
        Path of the directory to export the file

    Returns
    -------
    int
        0
    """
    nb_F2F_links = 0
    nb_F2H_lins = 0
    nb_C2F_links = 0
    nb_F2C_links = 0
    nb_C2C_links = 0
    for edge in sc_graph.edges:
        nb_F2F_links += int(isinstance(edge[0], Firm) and isinstance(edge[1], Firm))
        nb_F2H_lins += int(isinstance(edge[0], Firm) and isinstance(edge[1], Households))
        nb_C2F_links += int(isinstance(edge[0], Country) and isinstance(edge[1], Firm))
        nb_F2C_links += int(isinstance(edge[0], Firm) and isinstance(edge[1], Country))
        nb_C2C_links += int(isinstance(edge[0], Country) and isinstance(edge[1], Country))

    nb_clients_known_by_firm = sum([sum([1 if isinstance(client_id, int) else 0 for client_id in firm.clients.keys()]) for firm in firm_list])
    nb_suppliers_known_by_firm = sum([sum([1 if isinstance(supplier_id, int) else 0 for supplier_id in firm.suppliers.keys()]) for firm in firm_list])
    res = {
        "Nb firm to firm links": nb_F2F_links,
        "Nb firm to households links": nb_F2H_lins,
        "Nb country to firm links": nb_C2F_links,
        "Nb firm to country links": nb_F2C_links,
        "Nb country to country links": nb_C2C_links,
        "Nb clients in firm objects (check)": nb_clients_known_by_firm,
        "Nb suppliers in firm objects (check)": nb_suppliers_known_by_firm
    }
    logging.info("Nb firm to firm links: "+str(nb_F2F_links))
    logging.info("sc_network_summary.csv exported")
    pd.Series(res).to_csv(os.path.join(export_folder, "sc_network_summary.csv"))

    return 0

