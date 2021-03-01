import os
import logging
import json
import pandas as pd
import geopandas as gpd
from builder import extractEdgeList


def exportFirmODPointTable(firm_list, firm_table, household_table, filepath_road_nodes,
    export_firm_table, export_odpoint_table, export_folder):

    # Get info per firm
    imports = pd.Series({
        firm.pid: sum([val for key, val in firm.purchase_plan.items() if str(key)[0]=="C"]) 
        for firm in firm_list
    }, name='imports')
    production_table = pd.Series({
        firm.pid: firm.production_target 
        for firm in firm_list
    }, name='total_production')
    b2c_share = pd.Series({
        firm.pid: sum([val['share'] for key, val in firm.clients.items() if str(key)[0]=="h"])
        for firm in firm_list
    }, name='b2c_share')
    export_share = pd.Series({
        firm.pid: sum([val['share'] for key, val in firm.clients.items() if str(key)[0]=="C"])
        for firm in firm_list
    }, name='export_share')
    production_table = pd.concat([production_table, b2c_share, export_share, imports], axis=1)
    production_table['production_toC'] = production_table['total_production']*production_table['b2c_share']
    production_table['production_toB'] = production_table['total_production']-production_table['production_toC']
    production_table['production_exported'] = production_table['total_production']*production_table['export_share']
    production_table.index.name = 'id'
    firm_table = firm_table.merge(production_table.reset_index(), on="id", how="left")

    # Get info per od point
    odpoint_table_firm = firm_table.groupby('odpoint', as_index=False)['id'].count()
    odpoint_table_household = household_table.groupby('odpoint', as_index=False)['population'].sum()
    odpoint_table = pd.concat([
        odpoint_table_firm.rename(columns={'id': 'nb_firms'}),
        odpoint_table_household.rename(columns={'id': 'pop_in_households'}),
    ], sort=True)
    prod_per_sector_ODpoint_table = firm_table.groupby(['odpoint', 'sector'])['total_production'].sum().unstack().fillna(0).reset_index()
    odpoint_table = odpoint_table.merge(prod_per_sector_ODpoint_table, on='odpoint', how="left")
    odpoint_table = odpoint_table.merge(
        firm_table.groupby('odpoint', as_index=False)[
            ['total_production', 'production_toC', 'production_toB', 'production_exported']
        ].sum(),
        on="odpoint",
        how="left").fillna(0)

    if export_firm_table:
        firm_table.to_csv(os.path.join(export_folder, 'firm_table.csv'), index=False)
    
    if export_odpoint_table:
        odpoint_table.to_csv(os.path.join(export_folder, 'odpoint_production_table.csv'), index=False)
        exportODPointLayer(odpoint_table, filepath_road_nodes, export_folder)


def exportODPointLayer(odpoint_table, filepath_road_nodes, export_folder):
    road_nodes = gpd.read_file(filepath_road_nodes)
    res = road_nodes[['id', 'geometry']].merge(odpoint_table, left_on='id', right_on="odpoint", how='right')
    res.to_file(os.path.join(export_folder, 'odpoints_output.geojson'), driver='GeoJSON')



def exportDistrictSectorTable(filtered_district_sector_table, export_folder):
    filtered_district_sector_table.to_csv(os.path.join(export_folder, 'filtered_district_sector_table.csv'), index=False)


def exportCountryTable(country_list, export_folder):
    country_table = pd.DataFrame({
        'country':[country.pid for country in country_list],
        'purchases':[sum(country.purchase_plan.values()) for country in country_list],
        'purchases_from_countries':[sum([value if isinstance(key, str) else 0 for key, value in country.purchase_plan.items()]) for country in country_list],
        'purchases_from_firms':[sum([value if isinstance(key, int) else 0 for key, value in country.purchase_plan.items()]) for country in country_list]
    })
    # country_table['purchases_from_firms'] = \
    #     country_table['purchases'] - country_table['purchases_from_countries']
    country_table.to_csv(os.path.join(export_folder, 'country_table.csv'), index=False)


def exportEdgelistTable(supply_chain_network, export_folder):
    # Concern only Firm-Firm relationships, not Country, not Households
    edgelist_table = pd.DataFrame(extractEdgeList(supply_chain_network))
    edgelist_table.to_csv(os.path.join(export_folder, 'edgelist_table.csv'), index=False)

    # Log some key indicators, taking into account all firms
    av_km_btw_supplier_buyer = edgelist_table['distance'].mean()
    logging.info("Average distance between supplier and buyer: "+
        "{:.01f}".format(av_km_btw_supplier_buyer) + " km")
    av_km_btw_supplier_buyer_weighted_by_flow = \
        (edgelist_table['distance'] * edgelist_table['flow']).sum() / \
        edgelist_table['flow'].sum()
    logging.info("Average distance between supplier and buyer, weighted by the traded quantity: "+
        "{:.01f}".format(av_km_btw_supplier_buyer_weighted_by_flow) + " km")

    # Log some key indicators, excluding service firms
    boolindex = (edgelist_table['supplier_odpoint'] !=-1 ) & (edgelist_table['buyer_odpoint'] != -1)
    av_km_btw_supplier_buyer_no_service = edgelist_table.loc[boolindex, 'distance'].mean()
    logging.info("Average distance between supplier and buyer, excluding service firms: "+
        "{:.01f}".format(av_km_btw_supplier_buyer_no_service) + " km")
    av_km_btw_supplier_buyer_weighted_by_flow_no_service = \
        (edgelist_table.loc[boolindex, 'distance'] * edgelist_table.loc[boolindex, 'flow']).sum() / \
        edgelist_table.loc[boolindex, 'flow'].sum()
    logging.info("Average distance between supplier and buyer, excluding service firms"+
        ", weighted by the traded quantity: "+
        "{:.01f}".format(av_km_btw_supplier_buyer_weighted_by_flow_no_service) + " km")
    

def exportInventories(firm_list, export_folder):
    # Evaluate total inventory per good type
    inventories = {}
    for firm in firm_list:
        for input_id, inventory in firm.inventory.items():
            if input_id not in inventories.keys():
                inventories[input_id] = inventory
            else:
                inventories[input_id] += inventory
                
    pd.Series(inventories).to_csv(os.path.join(export_folder, 'inventories.csv'))


def exportTransportFlows(observer, export_folder):
    with open(os.path.join(export_folder, 'transport_flows.json'), 'w') as jsonfile:
        json.dump(observer.flows_snapshot, jsonfile)


def exportSpecificFlows(observer, export_folder):
    export_filename = os.path.join(export_folder, 'specific_flows.csv')
    pd.DataFrame(observer.specific_flows).transpose().to_csv(export_filename)


def exportTransportFlowsLayer(observer, export_folder, time_step, transport_edges):
    #extract flows of the desired time step
    flow_table = pd.DataFrame(observer.flows_snapshot[time_step]).transpose()
    flow_table['id'] = flow_table.index.astype(int)
    # print(flow_table)
    transport_edges_with_flows = transport_edges.merge(flow_table, on='id', how='left')
    transport_edges_with_flows.to_file(
        os.path.join(export_folder, 'flow_table_'+str(time_step)+'.geojson'), 
        driver='GeoJSON'
    )
    
def exportShipmentsLayer(observer, export_folder, time_step, transport_edges):
    #extract flows of the desired time step
    output_layer = transport_edges.copy()
    output_layer['shipments'] = output_layer['id'].map(observer.shipments_snapshot[time_step])
    output_layer.to_file(
        os.path.join(export_folder, 'shipments_'+str(time_step)+'.geojson'), 
        driver='GeoJSON'
    )

def exportAgentData(observer, export_folder):
    agent_data = {
        'firms': observer.firms,
        'countries': observer.countries,
        'households': observer.households
    }
    with open(os.path.join(export_folder, 'agent_data.json'), 'w') as jsonfile:
        json.dump(agent_data, jsonfile)


def analyzeSupplyChainFlows(sc_network, firm_list, export_folder):
    # Collect all flows
    io_flows = [[sc_network[edge[0]][edge[1]]['object'].delivery, sc_network[edge[0]][edge[1]]['object'].supplier_id, sc_network[edge[0]][edge[1]]['object'].buyer_id] for edge in sc_network.edges]
    io_flows = pd.DataFrame(columns=['quantity', 'supplier_id', 'buyer_id'], data=io_flows)
    
    # Analyze domestic B2C flows
    domestic_flows = io_flows[(io_flows['supplier_id'].apply(lambda x: isinstance(x, int))) & (io_flows['buyer_id'].apply(lambda x: isinstance(x, int)))]
    dic_firmid_to_sectorid = {firm.pid: firm.sector for firm in firm_list}
    domestic_b2c_flows = domestic_flows[domestic_flows['buyer_id'] == -1].copy()
    domestic_b2c_flows['from_sector'] = domestic_flows['supplier_id'].map(dic_firmid_to_sectorid)
    domestic_b2c_flows_per_sector = domestic_b2c_flows.groupby('from_sector')['quantity'].sum().reset_index()
    
    # Analyze domestic B2B flows
    domestic_b2b_flows = domestic_flows[domestic_flows['buyer_id'] >= 0].copy()
    domestic_b2b_flows['from_sector'] = domestic_b2b_flows['supplier_id'].map(dic_firmid_to_sectorid)
    domestic_b2b_flows['to_sector'] = domestic_b2b_flows['buyer_id'].map(dic_firmid_to_sectorid)
    domestic_b2b_flows_per_sector = domestic_b2b_flows.groupby(['from_sector', 'to_sector'])['quantity'].sum().reset_index()
    
    # Produce B2B io sector-to-sector matrix
    domestic_sectors = list(domestic_b2c_flows_per_sector['from_sector'].sort_values())
    observed_io_matrix = pd.DataFrame(index=domestic_sectors, columns=domestic_sectors, data=0)
    for i in range(domestic_b2b_flows_per_sector.shape[0]):
        observed_io_matrix.loc[domestic_b2b_flows_per_sector['from_sector'].iloc[i], domestic_b2b_flows_per_sector['to_sector'].iloc[i]] = domestic_b2b_flows_per_sector['quantity'].iloc[i]

    # Analyze import B2B flows
    import_flows = io_flows[(io_flows['supplier_id'].apply(lambda x: isinstance(x, str))) & (io_flows['buyer_id'].apply(lambda x: isinstance(x, int)))]
    import_b2b_flows = import_flows[import_flows['buyer_id'] >= 0].copy()
    import_b2b_flows_per_country = import_b2b_flows.groupby('supplier_id')['quantity'].sum().reset_index()
    import_b2b_flows['to_sector'] = import_b2b_flows['buyer_id'].map(dic_firmid_to_sectorid)
    import_b2b_flows_per_sector = import_b2b_flows.groupby('to_sector')['quantity'].sum().reset_index()
    
    # Analyze import B2C flows
    import_b2c_flows = import_flows[import_flows['buyer_id'] == -1].copy()

    # Analyze export flows
    export_flows = io_flows[(io_flows['supplier_id'].apply(lambda x: isinstance(x, int))) & (io_flows['buyer_id'].apply(lambda x: isinstance(x, str)))].copy()
    export_flows_per_country = export_flows.groupby('buyer_id')['quantity'].sum().reset_index()
    export_flows['from_sector'] = export_flows['supplier_id'].map(dic_firmid_to_sectorid)
    export_flows_per_sector = export_flows.groupby('from_sector')['quantity'].sum().reset_index()

    # Analyze transit flows
    transit_flows = io_flows[(io_flows['supplier_id'].apply(lambda x: isinstance(x, str))) & (io_flows['buyer_id'].apply(lambda x: isinstance(x, str)))].copy()
    transit_countries = pd.Series(list(set(transit_flows['supplier_id']) | set(transit_flows['buyer_id']))).sort_values().tolist()
    country_to_country_transit_matrix = pd.DataFrame(index=transit_countries, columns=transit_countries, data=0)
    for i in range(transit_flows.shape[0]):
        country_to_country_transit_matrix.loc[transit_flows['supplier_id'].iloc[i], transit_flows['buyer_id'].iloc[i]] = transit_flows['quantity'].iloc[i]
    
    # Form final consumption
    final_consumption = domestic_b2c_flows_per_sector.append(pd.DataFrame(index=['import'], data={'from_sector':'import', 'quantity':import_b2c_flows['quantity'].sum()}))
    
    # Enrich io matrix with import and export flows
    observed_io_matrix = pd.concat([
        pd.concat([
            observed_io_matrix, 
            import_b2b_flows_per_sector.set_index('to_sector').rename(columns={'quantity':'total'}).transpose()
            ], axis=0, sort=True),
        export_flows_per_sector.set_index('from_sector').rename(columns={'quantity':'total'})],
        axis=1, sort=True).fillna(0)
    
    # Regional io matrix
    # legacy, should be removed, we shall do these kind of analysis outside of the core model
    if False:
        dic_firmid_to_region = {firm.pid: dic_odpoint_to_region[firm.odpoint] for firm in firm_list}

        domestic_b2b_flows['from_region'] = domestic_b2b_flows['supplier_id'].map(dic_firmid_to_region)
        domestic_b2b_flows['to_region'] = domestic_b2b_flows['buyer_id'].map(dic_firmid_to_region)

        domestic_b2b_flows_per_region = domestic_b2b_flows.groupby(['from_region', 'to_region'])['quantity'].sum().reset_index()

        regions = pd.Series(list(set(dic_odpoint_to_region.values()))).sort_values().tolist()
        region_to_region_io_matrix = pd.DataFrame(index=regions, columns=regions, data=0)
        for i in range(domestic_b2b_flows_per_region.shape[0]):
            region_to_region_io_matrix.loc[domestic_b2b_flows_per_region['from_region'].iloc[i], domestic_b2b_flows_per_region['to_region'].iloc[i]] = domestic_b2b_flows_per_region['quantity'].iloc[i]

    # Export Report
    final_consumption.to_csv(os.path.join(export_folder, 'initial_final_consumption.csv'), index=False)
    observed_io_matrix.to_csv(os.path.join(export_folder, 'initial_sector_io_matrix.csv'), index=True)
    if False: # legacy, should be removed, we shall do these kind of analysis outside of the core model
        region_to_region_io_matrix.to_csv(os.path.join(export_folder, 'initial_region_to_region_io_matrix.csv'), index=True)
    country_to_country_transit_matrix.to_csv(os.path.join(export_folder, 'initial_transit_matrix.csv'), index=True)
    import_b2b_flows_per_country.to_csv(os.path.join(export_folder, 'initial_import_b2b_flows_per_country.csv'), index=False)
    export_flows_per_country.to_csv(os.path.join(export_folder, 'initial_export_flows_per_country.csv'), index=False)


def exportFirmTS(what, observer, export_folder):
    ts = pd.DataFrame(
        {t: \
            {firm_id: val[what] for firm_id, val in observer.firms[t].items()} 
        for t in observer.firms.keys()}
    ).transpose()
    ts.to_csv(os.path.join(export_folder, 'firm_'+what+'_ts.csv'), sep=',')
    return ts


def exportCountriesTS(what, observer, export_folder):
    ts = pd.DataFrame(
        {t: \
            {country_id: val[what] for country_id, val in observer.countries[t].items()} 
        for t in observer.firms.keys()}
    ).transpose()
    ts.to_csv(os.path.join(export_folder, 'countries_'+what+'_ts.csv'), sep=',')
    return ts


def exportHouseholdsTS(what, observer, export_folder):
    ts = pd.DataFrame(
        {t: val[what] for t, val in observer.households.items()}
    ).transpose()
    ts.to_csv(os.path.join(export_folder, 'households_'+what+'_ts.csv'), sep=',')
    return ts


def exportTimeSeries(observer, export_folder):  
    # Export time series per agent
    firm_production_ts = exportFirmTS('production', observer, export_folder)
    firm_profit_ts = exportFirmTS('profit', observer, export_folder)
    firm_transportcost_ts = exportFirmTS('transport_cost', observer, export_folder)
    firm_inputcost_ts = exportFirmTS('input_cost', observer, export_folder)
    firm_othercost_ts = exportFirmTS('other_cost', observer, export_folder)

    firm_avinventoryduration_ts = pd.DataFrame(
        {t: \
            {firm_id: sum(val['inventory_duration'].values())/len(val['inventory_duration'].values()) for firm_id, val in observer.firms[t].items()} 
        for t in observer.firms.keys()}
    ).transpose()
    firm_avinventoryduration_ts.to_csv(os.path.join(export_folder, 'firm_avinventoryduration_ts.csv'), sep=',')
    
    households_consumption_ts = exportHouseholdsTS('consumption', observer, export_folder)
    households_spending_ts = exportHouseholdsTS('spending', observer, export_folder)

    countries_spending_ts = exportCountriesTS('spending', observer, export_folder)

    # Export aggregated time series
    agg_df = pd.DataFrame({
        'firm_production': firm_production_ts.sum(axis=1),
        'firm_profit': firm_profit_ts.sum(axis=1),
        'firm_transportcost': firm_transportcost_ts.mean(axis=1),
        'firm_avinventoryduration': firm_avinventoryduration_ts.mean(axis=1),
        'households_consumption': households_consumption_ts.sum(axis=1),
        'households_spending': households_spending_ts.sum(axis=1),
        "countres_spending": countries_spending_ts.sum(axis=1)
    })
    agg_df.to_csv(os.path.join(export_folder, 'aggregate_ts.csv'), sep=',', index=False)


def initializeCriticalityExportFile(export_folder):
    criticality_export_file = open(os.path.join(export_folder, 'criticality.csv'), "w")
    criticality_export_file.write(
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
    return criticality_export_file


def initializeResPerFirmExportFile(export_folder, firm_list):
    extra_spending_export_file = open(os.path.join(export_folder, 'hh_extra_spending_per_firm.csv'), 'w')
    missing_consumption_export_file = open(os.path.join(export_folder, 'hh_missing_consumption_per_firm.csv'), 'w')
    list_firm_id = [str(firm.pid) for firm in firm_list]
    extra_spending_export_file.write('disrupted_node,disrupted_edge,'+','.join(list_firm_id)+"\n")
    missing_consumption_export_file.write('disrupted_node,disrupted_edge,'+','.join(list_firm_id)+"\n")
    return extra_spending_export_file, missing_consumption_export_file


def writeCriticalityResults(criticality_export_file, observer, disruption, 
    disruption_duration, computation_time):
    criticality_export_file.write(str(disruption['node']) \
        + ', '+ str(disruption['edge']) \
        + ',' + str(disruption_duration) \
        + ',' + str(observer.households_extra_spending) \
        + ',' + str(observer.households_extra_spending_local) \
        + ',' + str(observer.spending_recovered) \
        + ',' + str(observer.countries_extra_spending) \
        + ',' + str(observer.countries_consumption_loss) \
        + ',' + str(observer.households_consumption_loss) \
        + ',' + str(observer.households_consumption_loss_local) \
        + ',' + str(observer.consumption_recovered) \
        + ',' + str(observer.generalized_cost_normal) \
        + ',' + str(observer.generalized_cost_disruption) \
        + ',' + str(observer.generalized_cost_country_normal) \
        + ',' + str(observer.generalized_cost_country_disruption) \
        + ',' + str(observer.usd_transported_normal) \
        + ',' + str(observer.usd_transported_disruption) \
        + ',' + str(observer.tons_transported_normal) \
        + ',' + str(observer.tons_transported_disruption) \
        + ',' + str(observer.tonkm_transported_normal) \
        + ',' + str(observer.tonkm_transported_disruption) \
        + ',' + str(computation_time/60) \
        + "\n")

def writeResPerFirmResults(extra_spending_export_file, missing_consumption_export_file, 
    observer, disruption):
    val = [str(x) for x in observer.households_extra_spending_per_firm.tolist()]
    extra_spending_export_file.write(
        str(disruption['node'])+','+str(disruption['edge'])+','+",".join(val)+"\n"
    )
    val = [str(x) for x in observer.households_consumption_loss_per_firm.tolist()]
    missing_consumption_export_file.write(
        str(disruption['node'])+','+str(disruption['edge'])+','+",".join(val)+"\n"
    )
    # pd.DataFrame({str(disruption): obs.households_extra_spending_per_firm}).transpose().to_csv(extra_spending_export_file, header=False, mode='a')
    # pd.DataFrame({str(disruption): obs.households_consumption_loss_per_firm}).transpose().to_csv(f, header=False)


def exportParameters(exp_folder):
    copy_command = {
        'nt': 'copy',
        'posix': "cp"
    }
    os.system(copy_command[os.name]+" "+
        os.path.join('parameter', "parameters.py")+" "+
        os.path.join(exp_folder, "parameters.txt")
    )
    logging.info("Parameters exported in "+os.path.join(exp_folder, "parameters.txt"))