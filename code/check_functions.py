import pandas as pd


def compareProductionPurchasePlans(firm_list, country_list, household_list):
    # Create dictionary to map firm id and country id into sector
    dic_agent_id_to_sector = {firm.pid: firm.sector for firm in firm_list}
    for country in country_list:
        dic_agent_id_to_sector[country.pid] = "IMP"

    # Evalute purchase plans
    ## of firms
    df = pd.DataFrame({firm.pid: firm.purchase_plan for firm in firm_list})
    df["tot_purchase_planned_by_firms"] = df.sum(axis=1)
    df['input_sector'] = df.index.map(dic_agent_id_to_sector)
    df_firms = df.groupby('input_sector')["tot_purchase_planned_by_firms"].sum()

    ## of countries
    df = pd.DataFrame({country.pid: country.purchase_plan for country in country_list})
    df["tot_purchase_planned_by_countries"] = df.sum(axis=1)
    df['input_sector'] = df.index.map(dic_agent_id_to_sector)
    df_countries = df.groupby('input_sector')["tot_purchase_planned_by_countries"].sum()
 
    ## of households
    df = pd.DataFrame({household.pid: household.purchase_plan for household in household_list})
    df["tot_purchase_planned_by_households"] = df.sum(axis=1)
    # df = pd.DataFrame({"tot_purchase_planned_by_households": households.purchase_plan})
    df['input_sector'] = df.index.map(dic_agent_id_to_sector)
    df_households = df.groupby('input_sector')["tot_purchase_planned_by_households"].sum()

    ## concat
    df_purchase_plan = pd.concat([df_firms, df_countries, df_households], axis=1, sort=True)

    # Evalute productions/sales
    ## of firms
    df = pd.DataFrame({"tot_production_per_firm":
        {firm.pid: firm.production for firm in firm_list}
    })
    df['sector'] = df.index.map(dic_agent_id_to_sector)
    df_firms = df.groupby('sector')["tot_production_per_firm"].sum()

    ## of countries
    df = pd.DataFrame({"tot_production_per_country":
        {country.pid: country.qty_sold for country in country_list}
    })
    df['sector'] = df.index.map(dic_agent_id_to_sector)
    df_countries = df.groupby('sector')["tot_production_per_country"].sum()

    ## concat
    df_sales = pd.concat([df_firms, df_countries], axis=1, sort=True)

    # Compare
    res = pd.concat([df_purchase_plan, df_sales], axis=1, sort=True)
    res['dif'] = res["tot_purchase_planned_by_firms"] \
        + res["tot_purchase_planned_by_countries"] \
        + res["tot_purchase_planned_by_households"] \
        - res['tot_production_per_firm'] - res['tot_production_per_country']
    boolindex_unbalanced = res['dif'] > 1e-6
    if boolindex_unbalanced.sum() > 0:
        logging.warn("Sales does not equate purchases for sectors: "+
            str(res.index[boolindex_unbalanced].tolist()))


def compareDeliveredVsReceived():
    # not finished
    qty_delivered_by_firm_per_sector = {}
    for firm in firm_list:
        if firm.sector not in qty_delivered_by_firm_per_sector.keys():
            qty_delivered_by_firm_per_sector[firm.sector] = 0
        qty_delivered_by_firm = 0
        for edge in G.out_edges(firm):
            qty_delivered_by_firm_per_sector[firm.sector] += \
                G[firm][edge[1]]['object'].delivery

    qty_bought_by_household_per_sector = {}
    for edge in G.in_edges(households):
        if edge[0].sector not in qty_bought_by_household_per_sector.keys():
            qty_bought_by_household_per_sector[edge[0].sector] = 0
        qty_bought_by_household_per_sector[firm.sector] += \
            G[edge[0]][households]['object'].delivery

    qty_ordered_by_firm_per_sector = {}
    for firm in firm_list:
        if firm.sector not in qty_ordered_by_firm_per_sector.keys():
            qty_delivered_by_firm_per_sector[firm.sector] = 0
