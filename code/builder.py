import geopandas as gpd
import pandas as pd
import logging
import os
import networkx as nx
import math
import numpy as np
import yaml

from functions import extract_country_transit_point

from class_firm import Firm
from class_households import Households
from class_transport_network import TransportNetwork
from class_country import Country


def createTransportNetwork(filepath_road_nodes, filepath_road_edges, transport_params, filepath_extra_road_edges=None):
    """Create the transport network object

    It uses one shapefile for the nodes and another for the edges.
    Note that therea are strong constraints on these files, in particular on their attributes. 
    We can optionally use an additional edge shapefile, which contains extra road segments. Useful for scenario testing.

    :param filepath_road_nodes: Path of the shapefile of the road nodes 
    :type filepath_road_nodes: string

    :param filepath_road_edges: Path of the shapefile of the road edges 
    :type filepath_road_edges: string

    :param transport_params: Transport parameters. Should be in a specific format.
    :type transport_params: dictionary

    :param filepath_extra_road_edges: Path of the shapefile of any extra road edges to include. Default to None.
    :type filepath_extra_road_edges: string

    :return: TransportNetwork
    """

    # Create the transport network object, and set the "unit_cost" parameter, which is the ton.km cost of transporting sth
    T = TransportNetwork()
    T.graph['unit_cost'] = transport_params['transport_cost_per_tonkm']
    
    # Load node and edge data
    road_nodes = gpd.read_file(filepath_road_nodes)
    road_edges = gpd.read_file(filepath_road_edges)

    # Add additional edges, if any
    if filepath_extra_road_edges is not None:
        new_road_edges = gpd.read_file(filepath_extra_road_edges)
        new_road_edges.index = [road_edges.index.max()+1+item for item in list(range(new_road_edges.shape[0]))]
        road_edges = road_edges.append(new_road_edges.reindex(), verify_integrity=True)
    
    # Compute how much it costs to transport one USD worth of good on each road edge
    travel_time_paved = road_edges['kmpaved']/transport_params['speeds']['road']['paved']
    travel_time_unpaved = road_edges['kmunpaved']/transport_params["speeds"]['road']['unpaved']
    road_edges['cost_travel_time'] = (travel_time_paved + travel_time_unpaved) * transport_params["travel_cost_of_time"]
    road_edges['cost_variability'] = transport_params['variability_coef'] * (
            travel_time_paved*transport_params["variability"]['road']['paved'] + 
            travel_time_unpaved*transport_params["variability"]['road']['unpaved']
        )
    road_edges['time_cost'] = road_edges['cost_travel_time'] + road_edges['cost_variability']

    # Load the nodes and edges on the transport network object
    for road in road_edges['link']:
        T.add_transport_edge_with_nodes(road, road_edges, road_nodes)
    return T


def rescaleNbFirms2(sector_ODpoint_filename, nb_sectors, importance_threshold, export_firm_table=False, export_ODpoint_table=False, exp_folder=None):
    
    odpoints_with_firms = pd.read_excel(sector_ODpoint_filename)
    odpoints_with_firms['nodenumber'] = odpoints_with_firms['nodenumber'].astype(int)
    logging.info('Nb of combinations (od points, sectors): '+str(odpoints_with_firms.shape[0]))
    logging.info('Nb of firms if no threshold: '+str(2*odpoints_with_firms.shape[0]))
    
    # Remove selected sectors if asked
    if nb_sectors != 'all':
        odpoints_with_firms = odpoints_with_firms[odpoints_with_firms['sector_id'] < (nb_sectors+1)]
    
    # Put two firms each time firm_importance > threshold, 0 otherwise
    logging.info('Treshold is '+str(importance_threshold/2)+" for agriculture, "+str(importance_threshold)+" otherwise")
    odpoints_with_firms['nb_firms'] = 0
    odpoints_with_firms.loc[odpoints_with_firms['firm_importance2']>=importance_threshold, 'nb_firms'] = 2
    odpoints_with_firms.loc[(odpoints_with_firms['sector_id']==1) & (odpoints_with_firms['firm_importance2']>=(importance_threshold/2)), 'nb_firms'] = 2
    
    # Remove points without firms
    odpoints_with_firms = odpoints_with_firms[odpoints_with_firms['nb_firms']!=0]
    
    # To generate the firm table, duplicates rows with 2 firms, divide by 2 firm_importance, and generate a unique id
    firm_table = pd.concat([odpoints_with_firms, odpoints_with_firms[odpoints_with_firms['nb_firms']==2]], axis=0)
    firm_table['firm_importance2'] = firm_table['firm_importance2'] / firm_table['nb_firms']
    firm_table['firm_importance'] = firm_table['firm_importance2']
    logging.info('Nb of od points chosen: '+str(len(set(firm_table['nodenumber'])))+', final nb of firms chosen: '+str(firm_table.shape[0]))

    # Create firm table
    firm_table = pd.concat([firm_table, firm_table['geometry'].astype(str).str.extract('\((?P<long>.*) (?P<lat>.*)\)')], axis=1)
    firm_table = firm_table.sort_values('firm_importance', ascending=False)
    firm_table = firm_table.sort_values(['region', 'sector_id'])
    firm_table['id'] = list(range(firm_table.shape[0]))
    renaming = {'nodenumber':'location'}
    col_to_export = ['id', 'sector_id', 'location', 'firm_importance', 'geometry', 'long', 'lat']
    firm_table = firm_table.rename(columns=renaming)[col_to_export]
    
    if export_firm_table:
        firm_table.to_excel(os.path.join(exp_folder, 'firm_table.xlsx'), index=False)
        logging.info('firm_table.xlsx exported')
    
    # Create OD table
    od_table = odpoints_with_firms.copy()
    od_table = od_table.rename(columns={'nodenumber':'od_point'})[['od_point', 'loc_small_code', 'nb_points_same_district']]
    od_table = od_table.drop_duplicates().sort_values('od_point')
    if export_ODpoint_table:
        od_table.to_excel(os.path.join(exp_folder, 'odpoint_table.xlsx'), index=False)
        logging.info('odpoint_table.xlsx exported')
    
    return firm_table, od_table
    
    
    
def rescaleNbFirms3(filepath_district_sector_importance, filepath_odpoints, 
    district_sector_cutoff, nb_top_district_per_sector, 
    agri_sectors=None, service_sectors=None,
    export_firm_table=False, export_ODpoint_table=False, export_district_sector_table=False, exp_folder=None):
    """Generate the firm data

    It uses the district_sector_importance table, the odpoints, the cutoff values to generate the list of firms.

    :param filepath_district_sector_importance: Path for the district_sector_importance table
    :type filepath_district_sector_importance: string

    :param filepath_odpoints: Path for the odpoint table
    :type filepath_odpoints: string

    :param district_sector_cutoff: Cutoff value for selecting the combination of district and sectors. For agricultural sector it is divided by two.
    :type district_sector_cutoff: float

    :param nb_top_district_per_sector: Nb of extra district to keep based on importance rank per sector
    :type nb_top_district_per_sector: None or integer

    :param agri_sectors: list of the sectors that pertains to agriculture. If None no special treatment is given to agriculture.
    :type agri_sectors: list of string or None

    :param service_sectors: list of the sectors that pertains to services. If None no special treatment is given to services.
    :type service_sectors: list of string or None

    :return: tuple(pandas.DataFrame, pandas.DataFrame)
    """

    # Load 
    table_district_sector_importance = pd.read_csv(filepath_district_sector_importance)
    table_district_sector_importance = table_district_sector_importance[table_district_sector_importance['importance']!=0]
    logging.info('Nb of combinations (od points, sectors): '+str(table_district_sector_importance.shape[0]))

    # Filter district-sector combination that are above the cutoff value
    if agri_sectors:
        logging.info('Treshold is '+str(district_sector_cutoff/2)+" for agriculture sectors, "+str(district_sector_cutoff)+" otherwise")
        boolindex_overthreshold = table_district_sector_importance['importance']>= district_sector_cutoff
        boolindex_agri = (table_district_sector_importance['sector'].isin(agri_sectors)) & (table_district_sector_importance['importance'] >= district_sector_cutoff/2)
        filtered_district_sector = table_district_sector_importance[boolindex_overthreshold | boolindex_agri].copy()
    else:
        logging.info('Treshold is '+str(district_sector_cutoff))
        boolindex_overthreshold = table_district_sector_importance['importance']>= district_sector_cutoff
        filtered_district_sector = table_district_sector_importance[boolindex_overthreshold].copy()

    # Add the top district of each sector
    if isinstance(nb_top_district_per_sector, int):
        if nb_top_district_per_sector > 0:
            top_district_sector = pd.concat([
                table_district_sector_importance[table_district_sector_importance['sector']==sector].nlargest(nb_top_district_per_sector, 'importance')
                for sector in table_district_sector_importance['sector'].unique()
            ])
            filtered_district_sector = pd.concat([filtered_district_sector, top_district_sector]).drop_duplicates()
    if export_district_sector_table:
        filtered_district_sector.to_excel(os.path.join(exp_folder, 'filtered_district_sector.xlsx'), index=False)
    
    # Generate the OD sector table
    table_odpoints = pd.read_csv(filepath_odpoints)
    table_odpoints['nb_points_same_district'] = table_odpoints['district'].map(table_odpoints['district'].value_counts())
    od_sector_table = pd.merge(table_odpoints, filtered_district_sector, how='inner', on='district')
    od_sector_table['importance'] = od_sector_table['importance'] / od_sector_table['nb_points_same_district']
    
    # Create firm table
    # To generate the firm table, duplicates rows with 2 firms, divide by 2 firm_importance, and generate a unique id
    # Remove utilities, transport, and services
    if service_sectors:
        od_sector_table = od_sector_table[~od_sector_table['sector'].isin(service_sectors)]
        firm_table = od_sector_table.copy()    
        firm_table_services = pd.DataFrame({
            'odpoint':-1,
            'importance': 1/2,
            "sector": service_sectors*2
        })
        firm_table_services.index = [firm_table.index.max()+1+item for item in list(range(firm_table_services.shape[0]))]
        firm_table = pd.concat([firm_table, firm_table_services], sort=True)
    else:
        firm_table = od_sector_table.copy()

    firm_table = firm_table.sort_values('importance', ascending=False)
    firm_table = firm_table.sort_values(['district', 'sector'])
    firm_table['id'] = list(range(firm_table.shape[0]))
    firm_table = firm_table[['id', 'sector', 'odpoint', 'importance', 'district', 'geometry', 'long', 'lat']]
    
    if export_firm_table:
        firm_table.to_excel(os.path.join(exp_folder, 'firm_table.xlsx'), index=False)
        logging.info('firm_table.xlsx exported')
    
    # Create OD table
    od_table = od_sector_table.copy()
    od_table = od_table[['odpoint', 'district', 'nb_points_same_district', 'geometry', 'long', 'lat']]
    od_table = od_table.drop_duplicates().sort_values('odpoint')
    if export_ODpoint_table:
        od_table.to_excel(os.path.join(exp_folder, 'odpoint_table.xlsx'), index=False)
        logging.info('odpoint_table.xlsx exported')
    
    logging.info('Nb of od points chosen: '+str(len(set(firm_table['odpoint'])))+
        ', final nb of firms chosen: '+str(firm_table.shape[0]))


    return firm_table, od_table
    
    
    
def createFirms(firm_data, keep_top_n_firms=None, reactivity_rate=0.1, utilization_rate=0.8):
    """Create the firms

    It uses firm_table from rescaleNbFirms3

    :param firm_data: firm_table from rescaleNbFirms3
    :type firm_data: pandas.DataFrame

    :param keep_top_n_firms: (optional) can be specified if we want to keep only the first n firms, for testing purposes
    :type keep_top_n_firms: None (default) or integer


    :param reactivity_rate: Determines the speed at which firms try to reach their inventory duration target. Default to 0.1.
    :type reactivity_rate: float

    :param utilization_rate: Set the utilization rate, which determines the production capacity at the input-output equilibrium.
    :type utilization_rate: float

    :return: list of Firms
    """

    if isinstance(keep_top_n_firms, int):
        firm_data = firm_data.iloc[:keep_top_n_firms,:]

    logging.debug('Creating firm_list')
    ids = firm_data['id'].tolist()
    firm_data = firm_data.set_index('id')
    firm_list= [
        Firm(i, 
             sector=firm_data.loc[i, "sector"], 
             location=firm_data.loc[i, "odpoint"], 
             importance=firm_data.loc[i, 'importance'],
             geometry=firm_data.loc[i, 'geometry'],
             long=float(firm_data.loc[i, 'long']),
             lat=float(firm_data.loc[i, 'lat']),
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

    :param firm_list: the list of Firms generated from the createFirms function
    :type firm_list: pandas.DataFrame

    :param filepath_tech_coef: Filepath to the matrix of technical coefficients
    :type filepath_tech_coef: string

    :param io_cutoff: Filters out technical coefficient below this cutoff value. Default to 0.1.
    :type io_cutoff: float

    :param imports: Give the name of the import sector. If None, then the import technical coefficient is discarded. Default to None.
    :type imports: None or string

    :return: list of Firms
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
    


def loadInventories(firm_list, inventory_duration_target=2, filepath_inventory_duration_targets=None,
    extra_inventory_target=None, inputs_with_extra_inventories=None, buying_sectors_with_extra_inventories=None,
    random_mean_sd=None):
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

    :param firm_list: the list of Firms generated from the createFirms function
    :type firm_list: pandas.DataFrame

    :param inventory_duration_target: Inventory duration target uniformly applied to all firms and all inputs.
    If 'inputed', uses the specific values from the file specified by filepath_inventory_duration_targets
    :type inventory_duration_target: "inputed" or integer

    :param extra_inventory_target: If specified, extra inventory duration target.
    :type extra_inventory_target: None or integer

    :param inputs_with_extra_inventories: For which inputs do we add inventories.
    :type inputs_with_extra_inventories: None or list of sector

    :param buying_sectors_with_extra_inventories: For which sector we add inventories.
    :type buying_sectors_with_extra_inventories: None or list of sector

    :param random_mean_sd: Not yet implemented.
    :type random_mean_sd: None

    :return: list of Firms
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

    # inventory_table = pd.DataFrame({
    #     'id': [firm.pid for firm in firm_list],
    #     'buying_sector': [firm.sector for firm in firm_list],
    #     'inventories': [firm.inventory_duration_target for firm in firm_list]
    # })
    # inventory_table.to_csv('inventory_check.csv')
    # logging.info("Inventories: "+str({firm.pid: firm.inventory_duration_target for firm in firm_list}))
    return firm_list
    


def loadUsdPerTon(input_IO_filename, firm_list, country_list):
    sectorId_to_usdPerTon = pd.read_excel(input_IO_filename, sheet_name='sector_usdperton').set_index('sector_id')['usd_per_ton']
    for firm in firm_list:
        firm.usd_per_ton = sectorId_to_usdPerTon[firm.sector]
    for country in country_list:
        country.usd_per_ton = sectorId_to_usdPerTon.mean()
    return firm_list, country_list


def createCountries(input_IO_filename, nb_countries, present_sectors, time_resolution):
    periods = {'day': 365, 'week': 52, 'month': 12, 'year': 1}
    
    country_data = pd.read_excel(input_IO_filename, sheet_name="country_name", dtype={'transit_points':str}, index_col=0)
    import_data = pd.read_excel(input_IO_filename, sheet_name="imports")
    export_data = pd.read_excel(input_IO_filename, sheet_name="exports")
    transit_data = pd.read_excel(input_IO_filename, sheet_name="transits", index_col=0)
    present_sectors_as_str = [str(sector) for sector in present_sectors]
    
    if nb_countries != 'all':
        country_to_include = ['C'+str(num) for num in range(nb_countries+1)]
    else:
        country_to_include = country_data.index.tolist()
        
    transit_matrix = transit_data.loc[country_to_include, country_to_include]
    
    country_list = []
    total_imports = import_data.set_index('country_id').loc[country_to_include, present_sectors_as_str].sum().sum() / periods[time_resolution]
    for country_id in country_to_include:
        
        transit_points = extract_country_transit_point(country_data, country_id)
        
        qty_sold = import_data.loc[import_data['country_id'] == country_id, present_sectors_as_str].transpose().iloc[:,0]
        qty_sold = (qty_sold / periods[time_resolution]).to_dict()
        supply_importance = sum(qty_sold.values()) / total_imports
        
        qty_purchased = export_data.loc[import_data['country_id'] == country_id, present_sectors_as_str].transpose().iloc[:,0]
        qty_purchased = (qty_purchased / periods[time_resolution]).to_dict()
        qty_purchased = {int(key): val for key, val in qty_purchased.items()}
        
        transit_from = transit_matrix.loc[:,country_id] / periods[time_resolution]
        transit_from = transit_from * len(present_sectors) / 22
        transit_from = transit_from[transit_from>0].to_dict()
        transit_to = transit_matrix.loc[country_id, :] / periods[time_resolution]
        transit_to = transit_to * len(present_sectors) / 22
        transit_to = transit_to[transit_to>0].to_dict()
        
        country_list += [Country(name=country_data.loc[country_id, 'country_name'],
                                pid=country_id,
                                qty_sold=qty_sold,
                                qty_purchased=qty_purchased,
                                transit_points=transit_points,
                                transit_from=transit_from,
                                transit_to=transit_to,
                                supply_importance=supply_importance
                        )]
    return country_list


def defineFinalDemand(population_filename, input_IO_filename, firm_table, od_table, time_resolution='week', export_firm_table=False, exp_folder=None):
    # Compute population allocated to each od point
    population_per_district = pd.read_excel(population_filename)
    od_table = pd.merge(od_table, population_per_district, on='loc_small_code', how='left')
    od_table['population'] = od_table['population'] / od_table['nb_points_same_district']
    od_table['perc_population'] = od_table['population'] / od_table['population'].sum()
    logging.info('Population in district with firms: '+str(int(od_table['population'].sum()))+', total population is: '+str(population_per_district['population'].sum()))
    
    # Compute population allocated to each firm
    col_to_keep = ['id', 'sector_id', 'location', 'firm_importance', 'geometry', 'long', 'lat']
    firm_table = pd.merge(firm_table[col_to_keep], od_table.rename(columns={'od_point':'location'}), on='location', how='left')
    firm_table = pd.merge(firm_table,
                      firm_table.groupby(['location', 'sector_id'])['id'].count().reset_index().rename(columns={'id':'nb_firms_same_point_same_sector'}),
                      on=['location', 'sector_id'],
                      how='left')
    firm_table.loc[firm_table['location']==-1, 'perc_population'] = 1
    firm_table['final_demand_weight'] = firm_table['perc_population'] / firm_table['nb_firms_same_point_same_sector']
    
    # Weight will not add up to 1 in some if not all sectors, because not all sectors are present in each od point. We renormalize.
    firm_table['final_demand_weight'] = firm_table['final_demand_weight'] / firm_table['sector_id'].map(firm_table.groupby('sector_id')['final_demand_weight'].sum())
    
    # Check that weight sum up to 1
    sum_of_sectoral_final_demand_weight = firm_table.groupby('sector_id')['final_demand_weight'].sum()
    if ((sum_of_sectoral_final_demand_weight-1.0)>1e-6).any():
        logging.warning('The final demand of some sectors is problematic: '+str(sum_of_sectoral_final_demand_weight))
    
    # Allocate actual final demand, in dollars, using final consumption data of IO table, and take into account a time resolution
    periods = {'day': 365, 'week': 52, 'month': 12, 'year': 1}
    final_demand_per_sector = pd.read_excel(input_IO_filename, sheet_name='final_demand')
    firm_table['final_demand'] = firm_table['sector_id'].map(final_demand_per_sector.set_index('sector_id')['final_demand'])
    firm_table['final_demand'] = firm_table['final_demand'] * firm_table['final_demand_weight'] / periods[time_resolution]
    logging.info('Every '+time_resolution+', the total final demand is '+str(int(firm_table['final_demand'].sum())))
    actual_final_demand_per_sector = firm_table.groupby('sector_id')['final_demand'].sum()
        # for sector in actual_final_demand_per_sector.index:
        #     logging.debug(dic['sectorId_to_sectorName'][sector]+': '+str(int(actual_final_demand_per_sector[sector])))
    
    if export_firm_table:
        firm_table.to_excel(os.path.join(exp_folder, 'firm_table.xlsx'), index=False)
        logging.info('firm_table.xlsx exported')
    
    return firm_table
    

def createHouseholds(firm_data):
    households = Households()
    households.final_demand_per_sector = firm_data.groupby('sector_id')['final_demand'].sum().to_dict()
    households.purchase_plan = firm_data[['id', 'final_demand']].set_index('id')['final_demand'].to_dict()
    households.extra_spending_per_sector = {key: 0 for key, val in households.final_demand_per_sector.items()}
    return households
    
    
def extractEdgeList(graph):
    dic_commercial_links = nx.get_edge_attributes(graph, "object")
    dic_commercial_links = {key: value for key, value in dic_commercial_links.items() if (isinstance(key[0], Firm) and isinstance(key[1], Firm))}
    dic_commercial_links = [{
        'supplier_id':key[0].pid, 'buyer_id':key[1].pid,
        'supplier_location':key[0].location, 'buyer_location':key[1].location,
        'supplier_sector':key[0].sector, 'buyer_sector':key[1].sector, 
        'distance':key[0].distance_to_other(key[1]),
        'flow':commercial_link.order
    } for key, commercial_link in dic_commercial_links.items()]
    return dic_commercial_links