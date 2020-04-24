import os
import json
import pandas as pd
import networkx as nx
import geopandas as gpd

from functions import rescale_values


class Observer(object):
    
    def __init__(self, firm_list, Tfinal=0, exp_folder=None):
        self.firms = {}
        self.households = {}
        self.countries = {}
        sector_list = list(set([firm.sector for firm in firm_list]))
        self.disruption_time = 2
        self.production = pd.DataFrame(index=range(0,Tfinal+1), 
                                   columns=["firm_"+str(firm.pid) for firm in firm_list]+['total'], 
                                   data=0)
        self.delta_price = pd.DataFrame(index=range(0,Tfinal+1), 
                                   columns=["firm_"+str(firm.pid) for firm in firm_list]+['average'], 
                                   data=0)
        self.profit = pd.DataFrame(index=range(0,Tfinal+1), 
                                   columns=["firm_"+str(firm.pid) for firm in firm_list]+['total'], 
                                   data=0)
        self.consumption = pd.DataFrame(index=range(0,Tfinal+1), 
                                   columns=["sector_"+str(sector_id) for sector_id in sector_list]+['total'],
                                   data=0)
        self.spending = pd.DataFrame(index=range(0,Tfinal+1), 
                                   columns=["sector_"+str(sector_id) for sector_id in sector_list]+['total'],
                                   data=0)
        self.av_inventory_duration = pd.DataFrame(index=range(0,Tfinal+1), 
                                   columns=["firm_"+str(firm.pid) for firm in firm_list]+['average'], 
                                   data=0)
        self.flows_snapshot = {}
        self.disrupted_nodes = {}
        self.disrupted_edges = {}
        self.households_extra_spending = 0
        self.households_extra_local = 0
        self.households_extra_spending_per_firm = {}
        self.spending_recovered = True
        self.households_consumption_loss = 0
        self.households_consumption_loss_local = 0
        self.households_consumption_loss_per_firm = {}
        self.consumption_recovered = True
        self.exp_folder = exp_folder or ''
        self.firm_to_sector = {firm.pid: firm.sector for firm in firm_list}
        self.generalized_cost_normal = 0
        self.generalized_cost_disruption = 0
        self.generalized_cost_country_normal = 0
        self.generalized_cost_country_disruption = 0
        self.usd_transported_normal = 0
        self.usd_transported_disruption = 0
        self.tons_transported_normal = 0
        self.tons_transported_disruption = 0
        self.tonkm_transported_normal = 0
        self.tonkm_transported_disruption = 0
    
    def collect_agent_data(self, firm_list, households, country_list, t):
        self.firms[t] = {firm.pid: {
            'production': firm.production,
            'profit': firm.profit,
            'transport_cost': firm.finance['costs']['transport'],
            'inventory_duration': firm.current_inventory_duration,
            'generalized_transport_cost': firm.generalized_transport_cost,
            'usd_transported': firm.usd_transported,
            'tons_transported': firm.tons_transported,
            'tonkm_transported': firm.tonkm_transported
        } for firm in firm_list}
        self.countries[t] = {country.pid: {
            'generalized_transport_cost': country.generalized_transport_cost,
            'usd_transported': country.usd_transported,
            'tons_transported': country.tons_transported,
            'tonkm_transported': country.tonkm_transported,
            'extra_spending': country.extra_spending,
            'consumption_loss': country.consumption_loss
        } for country in country_list}
        self.households[t] = {
            'spending': households.spending,
            'consumption': households.consumption
        }
        
        
    def collect_transport_flows(self, transport_network, time_step, flow_types_to_export=None):
        self.disrupted_nodes[time_step] = [node for node, val in nx.get_node_attributes(transport_network, "disruption_duration").items() if val > 0]
        self.disrupted_edges[time_step] = [nx.get_edge_attributes(transport_network, "link")[edge] for edge, val in nx.get_edge_attributes(transport_network, "disruption_duration").items() if val > 0]
        flow_types_to_export = flow_types_to_export or ['total']
        self.flows_snapshot[time_step] = {
            str(transport_network[edge[0]][edge[1]]['link']): {str(sector): transport_network[edge[0]][edge[1]]["flow_"+str(sector)] for sector in flow_types_to_export}
            for edge in transport_network.edges
        }
        
        
    def compute_sectoral_IO_table(self, graph):
        for edge in graph.edges:
            graph[edge[0]][edge[1]]['object'].delivery
        
    
    def get_ts_feature_agg_all_agents(self, feature, agent_type):
        if agent_type=='firm':
            return pd.Series(index=list(self.firms.keys()), data=[sum([val[feature] for firm_id, val in self.firms[t].items()]) for t in self.firms.keys()])
        elif agent_type=='country':
            return pd.Series(index=list(self.countries.keys()), data=[sum([val[feature] for country_id, val in self.countries[t].items()]) for t in self.countries.keys()])
        elif agent_type=='firm+country':
            tot_firm = pd.Series(index=list(self.firms.keys()), data=[sum([val[feature] for firm_id, val in self.firms[t].items()]) for t in self.firms.keys()])
            tot_country = pd.Series(index=list(self.countries.keys()), data=[sum([val[feature] for country_id, val in self.countries[t].items()]) for t in self.countries.keys()])
            return tot_firm+tot_country
        else:
            raise ValueError("'agent_type' should be 'firm', 'country', or 'firm+country'")
    
        
    def evaluate_results(self, transport_network, households, disrupted_roads, disruption_duration, per_firm=False, export_folder=None):
        self.households_extra_spending = households.extra_spending
        print("Impact of disruption-induced price change on households:", '{:.4f}'.format(self.households_extra_spending))
        tot_spending_ts = pd.Series({t: sum(val['spending'].values()) for t, val in self.households.items()})
        self.spending_recovered = (tot_spending_ts.iloc[-1] - tot_spending_ts.iloc[0]) < 1e-6
        if self.spending_recovered:
            print("\tHouseholds spending has recovered")
        else:
            print("\tHouseholds spending has not recovered")
            
        self.households_consumption_loss = households.consumption_loss
        print("Impact of disruption-induced input shortages on households:", '{:.4f}'.format(self.households_consumption_loss))
        tot_consumption_ts = pd.Series({t: sum(val['consumption'].values()) for t, val in self.households.items()})
        self.consumption_recovered = (tot_consumption_ts.iloc[-1] - tot_consumption_ts.iloc[0]) < 1e-6
        if self.consumption_recovered:
            print("\tHouseholds consumption has recovered")
        else:
            print("\tHouseholds consumption has not recovered")
        
        tot_ts = self.get_ts_feature_agg_all_agents('generalized_transport_cost', 'firm+country')
        self.generalized_cost_normal = tot_ts.loc[1]
        self.generalized_cost_disruption = tot_ts.loc[self.disruption_time]
        
        tot_ts = self.get_ts_feature_agg_all_agents('generalized_transport_cost', 'country')
        self.generalized_cost_country_normal = tot_ts.loc[1]
        self.generalized_cost_country_disruption = tot_ts.loc[self.disruption_time]
        
        tot_ts = self.get_ts_feature_agg_all_agents('usd_transported', 'firm+country')
        self.usd_transported_normal = tot_ts.loc[1]
        self.usd_transported_disruption = tot_ts.loc[self.disruption_time]

        tot_ts = self.get_ts_feature_agg_all_agents('tons_transported', 'firm+country')
        self.tons_transported_normal = tot_ts.loc[1]
        self.tons_transported_disruption = tot_ts.loc[self.disruption_time]
        
        tot_ts = self.get_ts_feature_agg_all_agents('tonkm_transported', 'firm+country')
        self.tonkm_transported_normal = tot_ts.loc[1]
        self.tonkm_transported_disruption = tot_ts.loc[self.disruption_time]
        
        tot_ts = self.get_ts_feature_agg_all_agents('extra_spending', 'country')
        self.countries_extra_spending = tot_ts.sum()
        
        tot_ts = self.get_ts_feature_agg_all_agents('consumption_loss', 'country')
        self.countries_consumption_loss = tot_ts.sum()
        
        # Measure local impact
        firm_id_in_disrupted_nodes = [firm_id for disrupted_node in disrupted_roads['node_nb'] for firm_id in transport_network.node[disrupted_node]['firms_there']]

        extra_spending_per_firm = pd.DataFrame({t: val['spending'] for t, val in self.households.items()}).transpose()
        extra_spending_per_firm = extra_spending_per_firm.sum() - extra_spending_per_firm.shape[0]*extra_spending_per_firm.iloc[0,:]
        self.households_extra_spending_local = extra_spending_per_firm[firm_id_in_disrupted_nodes].sum()
        
        consumption_loss_per_firm = pd.DataFrame({t: val['consumption'] for t, val in self.households.items()}).transpose()
        consumption_loss_per_firm = -(consumption_loss_per_firm.sum() - consumption_loss_per_firm.shape[0]*consumption_loss_per_firm.iloc[0,:])
        self.households_consumption_loss_local = consumption_loss_per_firm[firm_id_in_disrupted_nodes].sum()
        
        if per_firm:
            self.households_extra_spending_per_firm = extra_spending_per_firm
            self.households_extra_spending_per_firm[self.households_extra_spending_per_firm<1e-6] = 0
            self.households_consumption_loss_per_firm = consumption_loss_per_firm
            self.households_consumption_loss_per_firm[self.households_consumption_loss_per_firm<1e-6] = 0

        if export_folder is not None:
            pd.DataFrame(index=["value", "recovered"], data={
                "agg_spending":[self.households_extra_spending, self.spending_recovered], 
                "agg_consumption":[self.households_consumption_loss, self.consumption_recovered]
            }).to_csv(os.path.join(export_folder, "results.csv"))


    def export_time_series(self, exp_folder):  
        # Calculus
        firm_production_ts = pd.DataFrame({t: {firm_id: val['production'] for firm_id, val in self.firms[t].items()} for t in self.firms.keys()}).transpose()
        firm_profit_ts = pd.DataFrame({t: {firm_id: val['profit'] for firm_id, val in self.firms[t].items()} for t in self.firms.keys()}).transpose()
        firm_transportcost_ts = pd.DataFrame({t: {firm_id: val['transport_cost'] for firm_id, val in self.firms[t].items()} for t in self.firms.keys()}).transpose()
        firm_avinventoryduration_ts = pd.DataFrame({t: {firm_id: sum(val['inventory_duration'].values())/len(val['inventory_duration'].values()) for firm_id, val in self.firms[t].items()} for t in self.firms.keys()}).transpose()
        households_consumption_ts = pd.DataFrame({t: val['consumption'] for t, val in self.households.items()}).transpose()
        households_spending_ts = pd.DataFrame({t: val['spending'] for t, val in self.households.items()}).transpose()
        # Export
        firm_production_ts.to_csv(os.path.join(exp_folder, 'firm_production_ts.csv'), sep=',')
        firm_profit_ts.to_csv(os.path.join(exp_folder, 'firm_profit_ts.csv'), sep=',')
        firm_transportcost_ts.to_csv(os.path.join(exp_folder, 'firm_transportcost_ts.csv'), sep=',')
        firm_avinventoryduration_ts.to_csv(os.path.join(exp_folder, 'firm_avinventoryduration_ts.csv'), sep=',')
        households_consumption_ts.to_csv(os.path.join(exp_folder, 'households_consumption_ts.csv'), sep=',')
        households_spending_ts.to_csv(os.path.join(exp_folder, 'households_spending_ts.csv'), sep=',')
        # Get aggregate time series
        agg_df = pd.DataFrame({
            'firm_production': firm_production_ts.sum(axis=1),
            'firm_profit': firm_profit_ts.sum(axis=1),
            'firm_transportcost': firm_transportcost_ts.mean(axis=1),
            'firm_avinventoryduration': firm_avinventoryduration_ts.mean(axis=1),
            'households_consumption': households_consumption_ts.sum(axis=1),
            'households_spending': households_spending_ts.sum(axis=1)
        })
        agg_df.to_csv(os.path.join(exp_folder, 'aggregate_ts.csv'), sep=',', index=False)
        
            
    @staticmethod
    def agg_per_sector(df, mapping, fun='sum'):
        tt = df.transpose().copy()
        tt['cat'] = tt.index.map(mapping)
        if fun=='sum':
            tt = tt.groupby('cat').sum().transpose()
        elif fun=='mean':
            tt = tt.groupby('cat').mean().transpose()
        else:
            raise ValueError('Fun should be sum or mean')
        return tt

        
        
