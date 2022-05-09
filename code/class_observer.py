import os
import json
import pandas as pd
import networkx as nx
import geopandas as gpd
import logging

from functions import rescale_values

# Observer is an object that collects data while the simulation is running
# It does not export anything. Export functions are in export_functions.py
class Observer(object):
    
    def __init__(self, firm_list, Tfinal=0):
        self.firms = []
        self.households = []
        self.countries = []
        sector_list = list(set([firm.sector for firm in firm_list]))
        self.disruption_time = 1 #time of first disruption
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
        self.shipments_snapshot = {}
        self.disrupted_nodes = {}
        self.disrupted_edges = {}
        self.households_extra_spending = 0
        self.households_extra_local = 0
        self.households_extra_spending_local = {}
        self.households_extra_spending_per_firm = {}
        self.spending_recovered = True
        self.households_consumption_loss = 0
        self.households_consumption_loss_local = 0
        self.households_consumption_loss_per_firm = {}
        self.consumption_recovered = True
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
    


    def collect_agent_data(self, firm_list, household_list, country_list, time_step):
        self.firms += [
            {
                'time_step': time_step,
                'firm': firm.pid,
                'production': firm.production,
                'profit': firm.profit,
                'transport_cost': firm.finance['costs']['transport'],
                'input_cost': firm.finance['costs']['input'],
                'other_cost': firm.finance['costs']['other'],
                'inventory_duration': firm.current_inventory_duration,
                'generalized_transport_cost': firm.generalized_transport_cost,
                'usd_transported': firm.usd_transported,
                'tons_transported': firm.tons_transported,
                'tonkm_transported': firm.tonkm_transported
            } 
            for firm in firm_list
        ]

        self.countries += [
            {
                'time_step': time_step,
                'country': country.pid,
                'generalized_transport_cost': country.generalized_transport_cost,
                'usd_transported': country.usd_transported,
                'tons_transported': country.tons_transported,
                'tonkm_transported': country.tonkm_transported,
                'extra_spending': country.extra_spending,
                'consumption_loss': country.consumption_loss,
                'spending': sum(list(country.qty_purchased.values()))
            } 
            for country in country_list
        ]

        self.households += [
            {
                'time_step': time_step,
                'household': household.pid,
                'spending_per_retailer': household.spending_per_retailer,
                'consumption_per_retailer': household.consumption_per_retailer,
                'extra_spending': household.extra_spending,
                'consumption_loss': household.consumption_loss
            }
            for household in household_list
        ]
        


    def collect_agent_dataOLD(self, firm_list, household_list, country_list, time_step):
        self.firms[time_step] = {
            firm.pid: {
                'production': firm.production,
                'profit': firm.profit,
                'transport_cost': firm.finance['costs']['transport'],
                'input_cost': firm.finance['costs']['input'],
                'other_cost': firm.finance['costs']['other'],
                'inventory_duration': firm.current_inventory_duration,
                'generalized_transport_cost': firm.generalized_transport_cost,
                'usd_transported': firm.usd_transported,
                'tons_transported': firm.tons_transported,
                'tonkm_transported': firm.tonkm_transported
            } 
            for firm in firm_list
        }

        self.countries[time_step] = {
            country.pid: {
                'generalized_transport_cost': country.generalized_transport_cost,
                'usd_transported': country.usd_transported,
                'tons_transported': country.tons_transported,
                'tonkm_transported': country.tonkm_transported,
                'extra_spending': country.extra_spending,
                'consumption_loss': country.consumption_loss,
                'spending': sum(list(country.qty_purchased.values()))
            } 
            for country in country_list
        }

        self.households[time_step] = {
            household.pid: {
                'spending_per_retailer': household.spending_per_retailer,
                'consumption_per_retailer': household.consumption_per_retailer,
                'extra_spending': household.extra_spending,
                'consumption_loss': household.consumption_loss
            }
            for household in household_list
        }
        
        
    def collect_transport_flows(self, transport_network, time_step, flow_types=None,
        collect_shipments=False):
        """
        Store the transport flow at that time step.

        See TransportNetwork.compute_flow_per_segment() for details on the flow types.

        Parameters
        ----------
        transport_network : TransportNetwork
            Transport network
        time_step : int
            The time step to index these data
        flow_types : list of string
            See TransportNetwork.compute_flow_per_segment() for details
        collect_shipments : Boolean
            Whether or not to store all indivual shipments

        Returns
        -------
        Nothing
        """

        # Get disrupted nodes and edges
        self.disrupted_nodes[time_step] = [
            node 
            for node, val in nx.get_node_attributes(transport_network, "disruption_duration").items() 
            if val > 0
        ]
        self.disrupted_edges[time_step] = [
            nx.get_edge_attributes(transport_network, "id")[edge] 
            for edge, val in nx.get_edge_attributes(transport_network, "disruption_duration").items() 
            if val > 0
        ]

        # Store flow
        flow_types = flow_types or ['total']
        self.flows_snapshot[time_step] = {
            str(transport_network[edge[0]][edge[1]]['id']): {
                str(flow_type): transport_network[edge[0]][edge[1]]["flow_"+str(flow_type)] 
                for flow_type in flow_types
            }
            for edge in transport_network.edges
            if transport_network[edge[0]][edge[1]]['type'] != 'virtual' #deprecated, no virtual edges anymore
        }
        for edge in transport_network.edges:
            edge_id = transport_network[edge[0]][edge[1]]['id']
            self.flows_snapshot[time_step][str(edge_id)]['total_tons'] = \
                transport_network[edge[0]][edge[1]]["current_load"]

        # Store shipments
        if collect_shipments:
            self.shipments_snapshot[time_step] = {
                transport_network[edge[0]][edge[1]]['id']: transport_network[edge[0]][edge[1]]["shipments"]
                for edge in transport_network.edges
            }
        

    def collect_specific_flows(self, transport_network):
        self.specific_flows = {}
        multimodal_flows_to_collect = [
            'roads-maritime-shv',
            'roads-maritime-vnm',
            'railways-maritime',
            'waterways-maritime',
        ]
        special_flows_to_collect = [
            'poipet',
            'bavet'
        ]
        flow_types = ['import', 'export', 'transit']
        for edge in transport_network.edges:
            if transport_network[edge[0]][edge[1]]['multimodes'] in multimodal_flows_to_collect:
                self.specific_flows[transport_network[edge[0]][edge[1]]['multimodes']] = {
                    flow_type:  transport_network[edge[0]][edge[1]]["flow_"+str(flow_type)] 
                    for flow_type in flow_types
                }
            if transport_network[edge[0]][edge[1]]['special']:
                for special in special_flows_to_collect:
                    if special in transport_network[edge[0]][edge[1]]['special']:
                        self.specific_flows[special] = {
                            flow_type:  transport_network[edge[0]][edge[1]]["flow_"+str(flow_type)] 
                            for flow_type in flow_types
                        }

        # code to display key metric on multimodal internatinonal flow
        df = pd.DataFrame(self.specific_flows).transpose()
        # print(df)
        total_import = df.loc[multimodal_flows_to_collect, 'import'].sum()
        # print(total_import)
        total_export = df.loc[multimodal_flows_to_collect, 'export'].sum()
        total = total_import + total_export
        # print(total)
        res = {}
        for flow in multimodal_flows_to_collect:
            # print(df.loc[flow, "import"])
            # print(df.loc[flow, "import"] / total_import)
            res[flow] = {
                "import": df.loc[flow, "import"] / total_import,
                "export": df.loc[flow, "export"] / total_export,
                "total": df.loc[flow, ["import", "export"]].sum() / total
            }
        res["total"] = {
            "import": total_import,
            "export": total_export,
            "total": total
        }
        print(pd.DataFrame(res).transpose())

        
    def compute_sectoral_IO_table(self, graph):
        for edge in graph.edges:
            graph[edge[0]][edge[1]]['object'].delivery
        
    
    # def get_ts_feature_agg_all_agents(self, feature, agent_type):
    #     if agent_type=='firm':
    #         return pd.Series(index=list(self.firms.keys()), data=[sum([val[feature] for firm_id, val in self.firms[t].items()]) for t in self.firms.keys()])
    #     elif agent_type=='country':
    #         return pd.Series(index=list(self.countries.keys()), data=[sum([val[feature] for country_id, val in self.countries[t].items()]) for t in self.countries.keys()])
    #     elif agent_type=='firm+country':
    #         tot_firm = pd.Series(index=list(self.firms.keys()), data=[sum([val[feature] for firm_id, val in self.firms[t].items()]) for t in self.firms.keys()])
    #         tot_country = pd.Series(index=list(self.countries.keys()), data=[sum([val[feature] for country_id, val in self.countries[t].items()]) for t in self.countries.keys()])
    #         return tot_firm+tot_country
    #     else:
    #         raise ValueError("'agent_type' should be 'firm', 'country', or 'firm+country'")
    
        
    def evaluate_results(self, transport_network, household_list, 
        disrupted_nodes, epsilon_stop_condition, per_firm=False):

        # parameter (should be in input)
        initial_time_step = 0

        # Turn into dataframe
        household_df = pd.DataFrame(self.households)
        firm_df = pd.DataFrame(self.firms)
        country_df = pd.DataFrame(self.countries)

        # Evaluate household total extra spending
        ts = household_df.groupby('time_step')['extra_spending'].sum()
        self.households_extra_spending = ts.sum()
        msg = "Price impact on households: "+'{:.4f}'.format(self.households_extra_spending)
        self.spending_recovered = abs(ts.iloc[-1] - ts.loc[initial_time_step]) < epsilon_stop_condition
        if self.spending_recovered:
            msg += " (recoved)"
        else:
            msg += " (not recoved)"
        logging.info(msg)

        # Evaluate country total consumption loss
        ts = household_df.groupby('time_step')['consumption_loss'].sum()
        self.households_consumption_loss = household_df['consumption_loss'].sum()
        msg = "Shortage impact on households: "+'{:.4f}'.format(self.households_consumption_loss)
        self.consumption_recovered = abs(ts.iloc[-1] - ts.loc[initial_time_step]) < epsilon_stop_condition
        if self.consumption_recovered:
            msg += " (recoved)"
        else:
            msg += " (not recoved)"
        logging.info(msg)

        # Country extra spending
        ts = country_df.groupby('time_step')['extra_spending'].sum()
        self.countries_extra_spending = ts.sum()
        msg = "Price impact on countries: "+'{:.4f}'.format(self.countries_extra_spending)
        if abs(ts.iloc[-1] - ts.loc[initial_time_step]) < epsilon_stop_condition:
            msg += " (recoved)"
        else:
            msg += " (not recoved)"
        logging.info(msg)

        # Country extra spending
        ts = country_df.groupby('time_step')['consumption_loss'].sum()
        self.countries_consumption_loss = ts.sum()
        msg = "Shortage impact on countries: "+'{:.4f}'.format(self.countries_consumption_loss)
        if abs(ts.iloc[-1] - ts.loc[initial_time_step]) < epsilon_stop_condition:
            msg += " (recoved)"
        else:
            msg += " (not recoved)"
        logging.info(msg)


        # Compute other indicators
        ts = firm_df.groupby('time_step')['generalized_transport_cost'].sum() +\
             country_df.groupby('time_step')['generalized_transport_cost'].sum()
        self.generalized_cost_normal = ts.loc[initial_time_step]
        self.generalized_cost_disruption = ts.loc[self.disruption_time]
        
        ts = country_df.groupby('time_step')['generalized_transport_cost'].sum()
        self.generalized_cost_country_normal = ts.loc[initial_time_step]
        self.generalized_cost_country_disruption = ts.loc[self.disruption_time]
        
        ts = firm_df.groupby('time_step')['usd_transported'].sum() +\
             country_df.groupby('time_step')['usd_transported'].sum()
        self.usd_transported_normal = ts.loc[initial_time_step]
        self.usd_transported_disruption = ts.loc[self.disruption_time]

        ts = firm_df.groupby('time_step')['tons_transported'].sum() +\
             country_df.groupby('time_step')['tons_transported'].sum()
        self.tons_transported_normal = ts.loc[initial_time_step]
        self.tons_transported_disruption = ts.loc[self.disruption_time]
        
        ts = firm_df.groupby('time_step')['tonkm_transported'].sum() +\
             country_df.groupby('time_step')['tonkm_transported'].sum()
        self.tonkm_transported_normal = ts.loc[initial_time_step]
        self.tonkm_transported_disruption = ts.loc[self.disruption_time]

        
        
        # Measure impact per firm
        if per_firm:
            epsilon = 1e-6
            firm_id_in_disrupted_nodes = [
                # firm_id for disrupted_node in disruption['node'] 
                firm_id for disrupted_node in disrupted_nodes
                for firm_id in transport_network.node[disrupted_node]['firms_there']
            ]
            # spending
            spending_per_firm_per_t = pd.DataFrame([
                {
                    'time_step': record['time_step'],
                    'household': record['household'],
                    'firm_id': key,
                    'spending': value
                }
                for record in self.households
                for key, value in record['spending_per_retailer'].items()
            ])
            extra_spending_per_firm = spending_per_firm_per_t.groupby(['time_step', 'firm_id'])['spending'].sum().unstack('firm_id')
            extra_spending_per_firm = extra_spending_per_firm.sum() - extra_spending_per_firm.shape[0]*extra_spending_per_firm.iloc[0,:]
            self.households_extra_spending_local = extra_spending_per_firm[firm_id_in_disrupted_nodes].sum()
            
            # consumption
            consumption_per_firm_per_t = pd.DataFrame([
                {
                    'time_step': record['time_step'],
                    'household': record['household'],
                    'firm_id': key,
                    'consumption': value
                }
                for record in self.households
                for key, value in record['consumption_per_retailer'].items()
            ])
            consumption_loss_per_firm = consumption_per_firm_per_t.groupby(['time_step', 'firm_id'])['consumption'].sum().unstack('firm_id')
            consumption_loss_per_firm = -(consumption_loss_per_firm.sum() - consumption_loss_per_firm.shape[0]*consumption_loss_per_firm.iloc[0,:])
            self.households_consumption_loss_local = consumption_loss_per_firm[firm_id_in_disrupted_nodes].sum()

            self.households_extra_spending_per_firm = extra_spending_per_firm
            self.households_extra_spending_per_firm[self.households_extra_spending_per_firm<epsilon] = 0
            self.households_consumption_loss_per_firm = consumption_loss_per_firm
            self.households_consumption_loss_per_firm[self.households_consumption_loss_per_firm<epsilon] = 0


 
        
            
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

        
        
