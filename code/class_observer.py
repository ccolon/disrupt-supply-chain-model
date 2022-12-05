import os
import json
import matplotlib.pyplot as plt
from plotly.offline import download_plotlyjs, iplot, plot
import plotly.graph_objs as go
import pandas as pd
import networkx as nx
import geopandas as gpd

from functions import rescale_values, generate_basemap


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
        self.av_safety_days = pd.DataFrame(index=range(0,Tfinal+1), 
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
    
    def collect_data2(self, firm_list, households, country_list, t):
        self.firms[t] = {firm.pid: {
            'production': firm.production,
            'profit': firm.profit,
            'transport_cost': firm.finance['costs']['transport'],
            'safety_days': firm.current_safety_days,
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
        
    def collect_data(self, firm_list, households, t):
        firms_production = [firm.production for firm in firm_list]
        firms_delta_price = [firm.delta_price_input for firm in firm_list]
        firms_profit = [firm.profit for firm in firm_list]
        firms_av_safety_days = [pd.Series(list(firm.current_safety_days.values())).mean() for firm in firm_list]
        self.production.loc[t] = firms_production + [sum(firms_production)]
        self.delta_price.loc[t] = firms_delta_price + [sum(firms_delta_price)/len(firms_delta_price)]
        self.profit.loc[t] = firms_profit + [sum(firms_profit)]
        self.consumption.loc[t] = list(households.consumption_per_sector.values()) + [households.consumption]
        self.spending.loc[t] = list(households.spending_per_sector.values()) + [households.spending]
        self.av_safety_days.loc[t] = firms_av_safety_days + [pd.Series(firms_av_safety_days).mean()]
        
        
    def collect_data_flows(self, transport_network, t, sectors=None):
        self.disrupted_nodes[t] = [node for node, val in nx.get_node_attributes(transport_network, "disruption_duration").items() if val > 0]
        self.disrupted_edges[t] = [nx.get_edge_attributes(transport_network, "link")[edge] for edge, val in nx.get_edge_attributes(transport_network, "disruption_duration").items() if val > 0]
        sectors = sectors or ['total']
        self.flows_snapshot[t] = {
            str(transport_network[edge[0]][edge[1]]['link']): {str(sector): transport_network[edge[0]][edge[1]]["flow_"+str(sector)] for sector in sectors}
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
        firm_avsafetydays_ts = pd.DataFrame({t: {firm_id: sum(val['safety_days'].values())/len(val['safety_days'].values()) for firm_id, val in self.firms[t].items()} for t in self.firms.keys()}).transpose()
        households_consumption_ts = pd.DataFrame({t: val['consumption'] for t, val in self.households.items()}).transpose()
        households_spending_ts = pd.DataFrame({t: val['spending'] for t, val in self.households.items()}).transpose()
        # Export
        firm_production_ts.to_csv(os.path.join(exp_folder, 'firm_production_ts.csv'), sep=',')
        firm_profit_ts.to_csv(os.path.join(exp_folder, 'firm_profit_ts.csv'), sep=',')
        firm_transportcost_ts.to_csv(os.path.join(exp_folder, 'firm_transportcost_ts.csv'), sep=',')
        firm_avsafetydays_ts.to_csv(os.path.join(exp_folder, 'firm_avsafetydays_ts.csv'), sep=',')
        households_consumption_ts.to_csv(os.path.join(exp_folder, 'households_consumption_ts.csv'), sep=',')
        households_spending_ts.to_csv(os.path.join(exp_folder, 'households_spending_ts.csv'), sep=',')
        # Get aggregate time series
        agg_df = pd.DataFrame({
            'firm_production': firm_production_ts.sum(axis=1),
            'firm_profit': firm_profit_ts.sum(axis=1),
            'firm_transportcost': firm_transportcost_ts.mean(axis=1),
            'firm_avsafetydays': firm_avsafetydays_ts.mean(axis=1),
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

    
    def plot_time_series(self, aggregation="global"):
        time_ts = list(self.households.keys())
        
        # Get whole time series
        firm_production_ts = pd.DataFrame({t: {firm_id: val['production'] for firm_id, val in self.firms[t].items()} for t in self.firms.keys()}).transpose()
        firm_profit_ts = pd.DataFrame({t: {firm_id: val['profit'] for firm_id, val in self.firms[t].items()} for t in self.firms.keys()}).transpose()
        firm_transportcost_ts = pd.DataFrame({t: {firm_id: val['transport_cost'] for firm_id, val in self.firms[t].items()} for t in self.firms.keys()}).transpose()
        firm_avsafetydays_ts = pd.DataFrame({t: {firm_id: sum(val['safety_days'].values())/len(val['safety_days'].values()) for firm_id, val in self.firms[t].items()} for t in self.firms.keys()}).transpose()
        households_consumption_ts = pd.DataFrame({t: val['consumption'] for t, val in self.households.items()}).transpose()
        households_spending_ts = pd.DataFrame({t: val['spending'] for t, val in self.households.items()}).transpose()

        # Global aggregation
        graphs_aggregate = [
            go.Scatter(
                x=time_ts,
                y=households_consumption_ts.sum(axis=1),
                name='Total household consumption',
                xaxis='x',
                yaxis='y'
            ),
            go.Scatter(
                x=time_ts,
                y=households_spending_ts.sum(axis=1),
                name='Total household spending',
                xaxis='x',
                yaxis='y'
            ),
            go.Scatter(
                x=time_ts,
                y=firm_production_ts.sum(axis=1),
                name='Total production',
                xaxis='x2',
                yaxis='y2'
            ),
            go.Scatter(
                x=time_ts,
                y=firm_profit_ts.sum(axis=1),
                name='Total profit',
                xaxis='x2',
                yaxis='y2'
            ),
            go.Scatter(
                x=time_ts,
                y=firm_avsafetydays_ts.mean(axis=1),
                name='Average safety days',
                xaxis='x3',
                yaxis='y3'
            ),
        ]
        
        # Aggregation
        if aggregation=="sector":
            mapping = self.firm_to_sector
            firm_production_ts = self.agg_per_sector(firm_production_ts, mapping, fun='sum')
            firm_profit_ts = self.agg_per_sector(firm_profit_ts, mapping, fun='sum')
            firm_transportcost_ts = self.agg_per_sector(firm_transportcost_ts, mapping, fun='mean')
            firm_avsafetydays_ts = self.agg_per_sector(firm_avsafetydays_ts, mapping, fun='mean')
            households_spending_ts = self.agg_per_sector(households_spending_ts, mapping, fun='sum')
            households_consumption_ts = self.agg_per_sector(households_consumption_ts, mapping, fun='sum')
            
        elif aggregation=="firm":
            None
            
        graphs_households = [
            go.Scatter(
                x=time_ts,
                y=households_spending_ts[col],
                name=col,
                xaxis='x4',
                yaxis='y4'
            ) for col in households_spending_ts.columns
        ] + [
            go.Scatter(
                x=time_ts,
                y=households_consumption_ts[col],
                name=col,
                xaxis='x7',
                yaxis='y7'
            ) for col in households_consumption_ts.columns
        ]
            
        graphs_firms = [
            go.Scatter(
                x=time_ts,
                y=firm_production_ts[col],
                name=col,
                xaxis='x5',
                yaxis='y5'
            ) for col in firm_production_ts.columns
        ] + [
            go.Scatter(
                x=time_ts,
                y=firm_profit_ts[col],
                name=col,
                xaxis='x8',
                yaxis='y8'
            ) for col in firm_profit_ts.columns
        ]
            
        graphs_inventories = [
            go.Scatter(
                x=time_ts,
                y=firm_avsafetydays_ts[col],
                name=col,
                xaxis='x6',
                yaxis='y6'
            ) for col in firm_avsafetydays_ts.columns
        ]
        
        dms = [[0, 0.3], [0.35, 0.65], [0.7, 1]]

        axis_titles = {"x":" ", "prod":"Production", "prof":"Profit", "cons":"Consumption", "spend":"Spending", "invent":"Inventories"}
        main_title = 'Households extra spending {:.4f}'.format(self.households_extra_spending)+\
                     '<br>Households consumption loss {:.4f}'.format(self.households_consumption_loss)
        layout = {
            "xaxis":{"domain":dms[0], "title":axis_titles['x']},
            "yaxis":{"domain":dms[2], 'anchor':'x', "title":axis_titles['cons']+' & '+axis_titles['spend'],
                     "range":[0, 1.05*max([households_spending_ts.sum().max(), households_consumption_ts.sum().max()])]},
            
            "xaxis2":{"domain":dms[1], 'anchor':'y2', "title":axis_titles['x'], 'zeroline':False,'showline':False},
            "yaxis2":{"domain":dms[2], 'anchor':'x2', "title":axis_titles['prod']+' & '+axis_titles['prof'], 'zeroline':False,'showline':False,
                     "range":[0, 1.05*max([firm_production_ts.sum().max(), firm_profit_ts.sum().max()])]},
            
            "xaxis3":{"domain":dms[2], 'anchor':'y3', "title":axis_titles['x'], 'zeroline':False,'showline':False},
            "yaxis3":{"domain":dms[2], 'anchor':'x3', "title":axis_titles['invent'], 'zeroline':False,'showline':False, 
                      "range":[0, 1.05*firm_avsafetydays_ts.max().max()]},
            
            "xaxis4":{"domain":dms[0], 'anchor':'y4', "title":axis_titles['x'], 'zeroline':False,'showline':False},
            "yaxis4":{"domain":dms[1], 'anchor':'x4', "title":axis_titles['spend'], 'zeroline':False,'showline':False, 
                      "range":[0, 1.05*households_spending_ts.max().max()]},
            
            "xaxis5":{"domain":dms[1], 'anchor':'y5', "title":axis_titles['x'], 'zeroline':False,'showline':False},
            "yaxis5":{"domain":dms[1], 'anchor':'x5', "title":axis_titles['prod'], 'zeroline':False,'showline':False, 
                      "range":[0, 1.05*firm_production_ts.max().max()]},
                        
            "xaxis6":{"domain":dms[2], 'anchor':'y6', "title":axis_titles['x'], 'zeroline':False,'showline':False},
            "yaxis6":{"domain":dms[1], 'anchor':'x6', "title":axis_titles['invent'], 'zeroline':False,'showline':False, 
                      "range":[0, 1.05*firm_avsafetydays_ts.max().max()]},
                        
            "xaxis7":{"domain":dms[0], 'anchor':'y7', "title":axis_titles['x'], 'zeroline':False,'showline':False},
            "yaxis7":{"domain":dms[0], 'anchor':'x7', "title":axis_titles['cons'], 'zeroline':False,'showline':False, 
                      "range":[0, 1.05*households_consumption_ts.max().max()]},
             
            "xaxis8":{"domain":dms[1], 'anchor':'y8', "title":axis_titles['x'], 'zeroline':False,'showline':False},
            "yaxis8":{"domain":dms[0], 'anchor':'x8', "title":axis_titles['prof'], 'zeroline':False,'showline':False, 
                      "range":[0, 1.05*firm_profit_ts.max().max()]},
            
            "showlegend":False,
            "title":main_title,
            "titlefont":{'color':'rgb(67,67,67)', 'size': 20}
        }
        
        graphs = graphs_aggregate+graphs_firms+graphs_households+graphs_inventories
        plot(go.Figure(data=graphs, layout=layout), filename=os.path.join(self.exp_folder, 'time_series.html'))
        
        
    def plot_aggregates(self):
        graphs = [
            go.Scatter(
                x=self.production.index,
                y=self.production['total'],
                name='Production'
            ),
            go.Scatter(
                x=self.consumption.index,
                y=self.consumption,
                name='Consumption'
            ),
            go.Scatter(
                x=self.spending.index,
                y=self.spending,
                name='Spending'
            ),
            go.Scatter(
                x=self.profit.index,
                y=self.profit['total'],
                name='Profit'
            )
        ]
        graphs = graphs + [
            go.Scatter(
                x=self.consumption.index,
                y=self.consumption,
                name='Consumption',
                xaxis='x2',
                yaxis='y2'
            ),
            go.Scatter(
                x=self.spending.index,
                y=self.spending,
                name='Spending',
                xaxis='x2',
                yaxis='y2'
            )
        ]
        graphs = graphs + [
            go.Scatter(
                x=self.av_safety_days.index,
                y=self.av_safety_days['average'],
                name='Av # safety days',
                xaxis='x3',
                yaxis='y3'
            )
        ]
        
        dms = [[0, 0.45], [0.55, 1]]
        
        axis_titles = ["Time", "Aggregate values"]
        main_title = 'Households extra spending {:.4f}'.format(self.households_extra_spending)+\
                     '<br>Households consumption loss {:.4f}'.format(self.households_consumption_loss)
        layout = {
            "xaxis":{"domain":dms[0], "title":axis_titles[0], 
                     'zeroline':False,'showline':False},
            "yaxis":{"domain":dms[1], 'anchor':'x', "title":axis_titles[1], 
                     "range":[0, 1.05*max([self.production['total'].max(), self.consumption.max(), self.profit['total'].max()])],
                     'zeroline':False,'showline':False},
            
            "xaxis2":{"domain":dms[0], "title":axis_titles[0], 
                     'zeroline':False,'showline':False},
            "yaxis2":{"domain":dms[0], 'anchor':'x', "title":axis_titles[1], 
                     "range":[0.995*min([self.spending.min()]), 1.005*max([self.spending.max()])],
                     'zeroline':False,'showline':False},
            
            "xaxis3":{"domain":dms[1], 'anchor':'y3', "title":axis_titles[0], 
                      'zeroline':False,'showline':False},
            "yaxis3":{"domain":dms[1], 'anchor':'x3', "title":axis_titles[1], 
                      "range":[0, 1.05*self.av_safety_days['average'].max()],
                      'zeroline':False,'showline':False},
            
            "showlegend":False,
            "title":main_title,
            "titlefont":{'color':'rgb(67,67,67)', 'size': 20}
        }
        
        plot(go.Figure(data=graphs, layout=layout), filename=os.path.join(self.exp_folder, 'time_series.html'))

        
        
    def create_snapshot(self, transport_network, t, sector='total'):
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.set_aspect('equal')
        dic_sector_to_color = {0: 'green', 1: 'blue', 2: 'purple', 'total': 'black'}
        
        # Plot transport network
        edge_to_plot = gpd.GeoDataFrame(data={
            'link': list(nx.get_edge_attributes(transport_network, "link").values()),
            'node_tuple': list(nx.get_edge_attributes(transport_network, "node_tuple").values()), 
            'geometry': list(nx.get_edge_attributes(transport_network, "geometry").values()), 
            'flow': list(nx.get_edge_attributes(transport_network, "flow_"+str(sector)).values())
        }).set_geometry('geometry')
        edge_to_plot.plot(ax=ax, color='grey', linewidth=1, zorder=0)
        edge_to_plot.plot(ax=ax, color=dic_sector_to_color[sector], linewidth=rescale_values(edge_to_plot["flow"], 0, 3), zorder=10)

        # Plot disrupted edges
        disrupted_edge_to_plot = edge_to_plot[edge_to_plot['link'].isin(self.disrupted_edges[t])]
        if len(disrupted_edge_to_plot) > 0:
            disrupted_edge_to_plot.plot(ax=ax, color='red', linewidth=3, zorder=100)

        # Plot disrupted nodes
        node_to_plot = gpd.GeoDataFrame(data={
            'nodenumber': list(nx.get_node_attributes(transport_network, "nodenumber").values()), 
            'geometry': list(nx.get_node_attributes(transport_network, "geometry").values())
        }).set_geometry('geometry')
        disrupted_node_to_plot = node_to_plot[node_to_plot['nodenumber'].isin(self.disrupted_nodes[t])]
        if len(disrupted_node_to_plot) > 0:
            disrupted_node_to_plot.plot(ax=ax, color='red', markersize=30, zorder=100)

        return fig
        
        
    def export_snapshot(self, transport_network, t, sector='total'):
        fig = self.create_snapshot(transport_network, t, sector)
        fig.savefig(os.path.join("plot", "snapshot_"+'{:03}'.format(t)+'_'+sector+'.png'))
        plt.close(fig)
        

    def add_flow_layer(self, fig, ax, basemap, dic_sector_to_color, t, sector='total', max_flow=None):
        flows = pd.Series(self.flows_snapshot[t][str(sector)], name='flow')
        edge_to_plot = pd.concat([basemap['transport_edges'], flows], axis=1)
        edge_to_plot.plot(ax=ax, color=dic_sector_to_color[sector], linewidth=rescale_values(edge_to_plot["flow"], 0, 25, max_flow, 0.5),
                          zorder=100, joinstyle='round', capstyle='round')
        return fig
    
    
    def add_disruption_layer(self, fig, ax, basemap, t):
        if len(self.disrupted_nodes[t]) > 0:
            disrupted_nodes_to_plot = basemap['transport_nodes'][basemap['transport_nodes']['nodenumber'].isin(self.disrupted_nodes[t])]
            disrupted_nodes_to_plot.plot(ax=ax, color='red', marker='*', markersize=100, zorder=1000)
            
        if len(self.disrupted_edges[t]) > 0:
            disrupted_edges_to_plot = basemap['transport_edges'][basemap['transport_edges']['link'].isin(self.disrupted_nodes[t])]
            disrupted_edges_to_plot.plot(ax=ax, color='red', marker='*', markersize=100, zorder=1000)
            
        return fig
    
    
    def generate_all_snapshots(self, Tfinal, sectors, transport_network, firm_list, plot_size, dic):
        exp_folder = os.path.join("app", "static", "data")
        #self.exp_folder
        
        # remove all files in folder
        for filename in os.listdir(exp_folder):
            os.remove(os.path.join(exp_folder, filename))

        # export info
        info = {'t0':1, 'tfinal':Tfinal, 'sectors':dic['sectorId_to_sectorName']}
        with open(os.path.join(exp_folder, 'info.json'), 'w') as file:
            json.dump(info, file)
        print("Observer: info exported")
            
        # generate snapshots
        max_flow = pd.Series(self.flows_snapshot[1]['total'], name='flow').max()
        
        for time_to_plot in range(1, Tfinal+1):
            for sector_to_plot in sectors:
                fig, ax, basemap = generate_basemap(transport_network, firm_list, plot_size, plot_firms=True)
                fig = self.add_disruption_layer(fig, ax, basemap, time_to_plot)
                fig = self.add_flow_layer(fig, ax, basemap, dic['sectorId_to_sectorColor'], time_to_plot, sector_to_plot, max_flow)
                plt.axis('off')
                fig.savefig(os.path.join(exp_folder, "snapshot_"+'{:03}'.format(time_to_plot)+'_'+str(sector_to_plot)+'.png'), bbox_inches='tight')
                plt.close(fig)
                print("Observer: exported snapshot for t="+str(time_to_plot)+" and sector "+str(sector_to_plot))
        
        
    def analyzeFlows(self, G, firm_list, exp_folder, dic):
        # Collect all flows
        io_flows = [[G[edge[0]][edge[1]]['object'].delivery, G[edge[0]][edge[1]]['object'].supplier_id, G[edge[0]][edge[1]]['object'].buyer_id] for edge in G.edges]
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
                import_b2b_flows_per_sector.set_index('to_sector').rename(columns={'quantity':'total'}).transpose()],
                axis=0),
            export_flows_per_sector.set_index('from_sector').rename(columns={'quantity':'total'})],
            axis=1).fillna(0)
        
        # Regional io matrix
        dic_location_to_region = dic['location_to_region']
        dic_firmid_to_region = {firm.pid: dic_location_to_region[firm.location] for firm in firm_list}

        domestic_b2b_flows['from_region'] = domestic_b2b_flows['supplier_id'].map(dic_firmid_to_region)
        domestic_b2b_flows['to_region'] = domestic_b2b_flows['buyer_id'].map(dic_firmid_to_region)

        domestic_b2b_flows_per_region = domestic_b2b_flows.groupby(['from_region', 'to_region'])['quantity'].sum().reset_index()

        regions = pd.Series(list(set(dic_location_to_region.values()))).sort_values().tolist()
        region_to_region_io_matrix = pd.DataFrame(index=regions, columns=regions, data=0)
        for i in range(domestic_b2b_flows_per_region.shape[0]):
            region_to_region_io_matrix.loc[domestic_b2b_flows_per_region['from_region'].iloc[i], domestic_b2b_flows_per_region['to_region'].iloc[i]] = domestic_b2b_flows_per_region['quantity'].iloc[i]


        # Export Report
        writer = pd.ExcelWriter(os.path.join(exp_folder, 'flow_report.xlsx'))
        final_consumption.to_excel(writer, 'final_consumption', index=False)
        observed_io_matrix.to_excel(writer, 'sector_io_matrix', index=True)
        region_to_region_io_matrix.to_excel(writer, 'region_to_region_io_matrix', index=True)
        country_to_country_transit_matrix.to_excel(writer, 'transit_matrix', index=True)
        import_b2b_flows_per_country.to_excel(writer, 'import_b2b_flows_per_country', index=True)
        export_flows_per_country.to_excel(writer, 'export_flows_per_country', index=True)
        writer.save()