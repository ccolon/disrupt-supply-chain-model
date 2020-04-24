import os
import logging
import pandas as pd
from builder import extractEdgeList


def exportFirmODPointTable(firm_list, firm_table, odpoint_table,
    export_firm_table, export_odpoint_table, export_folder):

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
    prod_per_sector_ODpoint_table = firm_table.groupby(['odpoint', 'sector'])['total_production'].sum().unstack().fillna(0).reset_index()
    odpoint_table = odpoint_table.merge(prod_per_sector_ODpoint_table, on='odpoint', how="left")

    if export_firm_table:
        firm_table.to_csv(os.path.join(export_folder, 'firm_table.csv'), index=False)
    
    if export_odpoint_table:
        odpoint_table.to_csv(os.path.join(export_folder, 'odpoint_production_table.csv'), index=False)


def exportCountryTable(country_list, export_folder):
    country_table = pd.DataFrame({
        'country':[country.pid for country in country_list],
        'purchases':[sum(country.purchase_plan.values()) for country in country_list],
        'purchases_from_countries':[sum([value if str(key)[0]=='C' else 0 for key, value in country.purchase_plan.items()]) for country in country_list]
    })
    country_table['purchases_from_firms'] = \
        country_table['purchases'] - country_table['purchases_from_countries']
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
