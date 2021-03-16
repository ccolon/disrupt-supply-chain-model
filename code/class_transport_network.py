
import networkx as nx
import pandas as pd
import geopandas as gpd
import logging

from functions import rescale_values, congestion_function, transformUSDtoTons


class TransportNetwork(nx.Graph):
    
    def add_transport_node(self, node_id, all_nodes_data): #used in add_transport_edge_with_nodes
        node_attributes = ["id", "geometry", "special", "name"]
        node_data = all_nodes_data.loc[node_id, node_attributes].to_dict()
        node_data['shipments'] = {}
        node_data['disruption_duration'] = 0
        node_data['firms_there'] = []
        node_data['households_there'] = None
        node_data['type'] = 'road'
        self.add_node(node_id, **node_data)
        

    def add_transport_edge_with_nodes(self, edge_id, all_edges_data, all_nodes_data):
        # Selecting data
        edge_attributes = ['id', "type", 'surface', "geometry", "class", "km", 'special', "name",
            "multimodes", "capacity",
            "cost_per_ton", "travel_time", "time_cost", 'cost_travel_time', 'cost_variability', 'agg_cost']
        edge_data = all_edges_data.loc[edge_id, edge_attributes].to_dict()
        end_ids = all_edges_data.loc[edge_id, ["end1", "end2"]].tolist()
        # Creating the start and end nodes
        if end_ids[0] not in self.nodes:
            self.add_transport_node(end_ids[0], all_nodes_data)
        if end_ids[1] not in self.nodes:
            self.add_transport_node(end_ids[1], all_nodes_data)
        # Creating the edge
        self.add_edge(end_ids[0], end_ids[1], **edge_data)
        # print("edge id:", edge_id, "| end1:", end_ids[0], "| end2:", end_ids[1], "| nb edges:", len(self.edges))
        # print(self.edges)
        self[end_ids[0]][end_ids[1]]['node_tuple'] = (end_ids[0], end_ids[1])
        self[end_ids[0]][end_ids[1]]['shipments'] = {}
        self[end_ids[0]][end_ids[1]]['disruption_duration'] = 0
        self[end_ids[0]][end_ids[1]]['current_load'] = 0
        
        
    # def connect_country(self, country):
    #     self.add_node(country.pid, **{'type':'virtual'})
    #     for entry_point in country.entry_points: #ATT so far works for road only
    #         self.add_edge(entry_point, country.pid, 
    #             **{'type':'virtual', 'time_cost':1000}
    #         ) # high time cost to avoid that algo goes through countries

            
    # def remove_countries(self, country_list):
    #     country_node_to_remove = list(set(self.nodes) & set([country.pid for country in country_list]))
    #     for country in country_node_to_remove:
    #         self.remove_node(country)
            
        
    # def giveRouteCost(self, route):
    #     time_cost = 1 #cost cannot be 0
    #     for segment in route:
    #         if len(segment) == 2: #only edges have costs 
    #             if self[segment[0]][segment[1]]['type'] != 'virtual':
    #                 time_cost += self[segment[0]][segment[1]]['time_cost']
    #     return time_cost
        
        
    # def giveRouteCostAndTransportUnitCost(self, route):
    #     time_cost = 1 #cost cannot be 0
    #     cost_per_ton = 0
    #     for segment in route:
    #         if len(segment) == 2: #only edges have costs 
    #             if self[segment[0]][segment[1]]['type'] != 'virtual':
    #                 time_cost += self[segment[0]][segment[1]]['time_cost']
    #                 cost_per_ton += (surface=='paved')*self.graph['unit_cost']['road']['paved']+\
    #                              (surface=='unpaved')*self.graph['unit_cost']['road']['unpaved']

    #     return time_cost, cost_per_ton
    
    
    def giveRouteCaracteristicsOld(self, route):
        distance = 0 # km
        time_cost = 1 #USD, cost cannot be 0
        cost_per_ton = 0 #USD/ton
        for segment in route:
            if len(segment) == 2: #only edges have costs 
                if self[segment[0]][segment[1]]['type'] != 'virtual':
                    distance += self[segment[0]][segment[1]]['km']
                    time_cost += self[segment[0]][segment[1]]['time_cost']
                    surface = self[segment[0]][segment[1]]['surface']
                    cost_per_ton += (surface=='paved')*self.graph['unit_cost']['roads']['paved']+\
                                 (surface=='unpaved')*self.graph['unit_cost']['roads']['unpaved']
                    
        return distance, time_cost, cost_per_ton    
    

    def giveRouteCaracteristics(self, route):
        distance = 0 # km
        time_cost = 1 #USD, cost cannot be 0
        cost_per_ton = 0 #USD/ton
        for segment in route:
            if len(segment) == 2: #only edges have costs 
                if self[segment[0]][segment[1]]['type'] != 'virtual':
                    distance += self[segment[0]][segment[1]]['km']
                    time_cost += self[segment[0]][segment[1]]['time_cost']
                    cost_per_ton += self[segment[0]][segment[1]]['cost_per_ton']
                    
        return distance, time_cost, cost_per_ton
        
    
    def giveRouteCostWithCongestion(self, route):
        time_cost = 1 #cost cannot be 0
        for segment in route:
            if len(segment) == 2: #only edges have costs 
                if self[segment[0]][segment[1]]['type'] != 'virtual':
                    time_cost += self[segment[0]][segment[1]]['cost_variability'] + self[segment[0]][segment[1]]['cost_travel_time'] * (1 + self[segment[0]][segment[1]]['congestion'])
        return time_cost
        
        
    def giveCongestionCostOfTime(self, route):
        congestion_time_cost = 0
        for segment in route:
            if len(segment) == 2: #only edges have costs 
                if self[segment[0]][segment[1]]['type'] != 'virtual':
                    congestion_time_cost += self[segment[0]][segment[1]]['cost_travel_time'] * self[segment[0]][segment[1]]['congestion']
        return congestion_time_cost
        
        
    def defineWeights(self, route_optimization_weight):
        '''Define the edge weights used by firms and countries to decide routes

        There are 3 types of weights:
            - weight: the indicator 'route_optimization_weight' parametrized
            - capacity_weight: same as weight, but we add capacity_burden when load
            are over threshold
            - mode_weight: we generate different weights for different mode of transportation 
            defined in the commercial links. So far, we defined:
                - dom_road_weight: domestic roads (used between national firms)
                - intl_road_shv_weight: internatinal route using primarily roads via shv (maritime + roads)
                - intl_road_vnm_weight: internatinal route using primarily roads via vnm (maritime + roads)
                - intl_rail_weight: internatinal route using primarily ails (maritime + rails + roads)
                - intl_river_weight: internatinal route using primarily waterways (maritime + river + roads)

        The idea is to weight more or less different part of the network 
        to "force" agent to choose one mode or the other.
        Since road needs always to be taken, we define a smaller burden.

        We start with the "route_optimization_weight" chosen as parameter (e.g., cost_per_ton, travel_time)
        Then, we add a hugen burden if we want agents to avoid certain edges
        Or we set the weight to 0 if we want to favor it
        '''
        road_burden = 1e6
        other_mode_burden = 1e10
        self.mode_weights = ['road_weight', 'intl_road_shv_weight', 
            'intl_road_vnm_weight', 'intl_rail_weight', 'intl_river_weight']
        for edge in self.edges:
            self[edge[0]][edge[1]]['weight'] = self[edge[0]][edge[1]][route_optimization_weight]
            self[edge[0]][edge[1]]['capacity_weight'] = self[edge[0]][edge[1]][route_optimization_weight]
            self[edge[0]][edge[1]]['road_weight'] = self[edge[0]][edge[1]][route_optimization_weight]
            self[edge[0]][edge[1]]['intl_road_shv_weight'] = self[edge[0]][edge[1]][route_optimization_weight]
            self[edge[0]][edge[1]]['intl_road_vnm_weight'] = self[edge[0]][edge[1]][route_optimization_weight]
            self[edge[0]][edge[1]]['intl_rail_weight'] = self[edge[0]][edge[1]][route_optimization_weight]
            self[edge[0]][edge[1]]['intl_river_weight'] = self[edge[0]][edge[1]][route_optimization_weight]

            # road_weight => burden any multimodal links
            multimodal_links_to_exclude_dic = {
                "road_weight": [
                    'railways-maritime', 
                    'waterways-maritime', 
                    'roads-railways', 
                    'roads-waterways',
                    'roads-maritime-shv',
                    'roads-maritime-vnm'
                ],
                "intl_road_shv_weight": [
                    'railways-maritime', 
                    'waterways-maritime', 
                    'roads-railways', 
                    'roads-waterways',
                    'roads-maritime-vnm'
                ],
                "intl_road_vnm_weight": [
                    'railways-maritime', 
                    'waterways-maritime', 
                    'roads-railways', 
                    'roads-waterways',
                    'roads-maritime-shv'
                ],
                "intl_rail_weight": [
                    'waterways-maritime',
                    'roads-maritime-shv',
                    'roads-maritime-vnm',
                    'roads-waterways'
                ],
                "intl_river_weight": [
                    'railways-maritime',
                    'roads-maritime-shv',
                    'roads-maritime-vnm',
                    'roads-railways'
                ],                
            }
            for mode, multimodal_links_to_exclude in multimodal_links_to_exclude_dic.items():
                if self[edge[0]][edge[1]]['multimodes'] in multimodal_links_to_exclude:
                    self[edge[0]][edge[1]][mode] = other_mode_burden


    
    def locate_firms_on_nodes(self, firm_list, transport_nodes):
        transport_nodes['firms_there'] = ""
        for node_id in self.nodes:
            self.node[node_id]['firms_there'] = []
            # self.node[node_id]['households_there'] = []
        for firm in firm_list:
            self.node[firm.odpoint]['firms_there'].append(firm.pid)
            transport_nodes.loc[transport_nodes['id']==firm.odpoint, "firms_there"] += (','+str(firm.pid))
        # for household in household_list:
        #     self.node[household.odpoint]['households_there'].append(household.pid)
            # if firm.odpoint != -1:
            #     try:
            #         self.node[firm.odpoint]['firms_there'].append(firm.pid)
            #     except KeyError:
            #         logging.error('Transport network has no node numbered: '+str(firm.odpoint))
    

    def locate_households_on_nodes(self, household_list, transport_nodes):
        transport_nodes['household_there'] = None
        for household in household_list:
            # self.node[household.odpoint]['household_id'] = household.pid
            self.node[household.odpoint]['household_there'] = household.pid
            transport_nodes.loc[transport_nodes['id']==household.odpoint, "household_there"] = household.pid

    
    def provide_shortest_route(self, origin_node, destination_node, route_weight):
        '''
        nx.shortest_path returns path as list of nodes
        we transform it into a route, which contains nodes and edges:
        [(1,), (1,5), (5,), (5,8), (8,)]
        '''
        if (origin_node not in self.nodes):
            logging.debug("Origin node "+str(origin_node)+" not in the available transport network")
            return None

        elif (destination_node not in self.nodes):
            logging.debug("Destination node "+str(destination_node)+" not in the available transport network")
            return None

        elif nx.has_path(self, origin_node, destination_node):
            sp = nx.shortest_path(self, origin_node, destination_node, weight=route_weight)
            route = [[(sp[0],)]] + [[(sp[i], sp[i+1]), (sp[i+1],)] for i in range(0,len(sp)-1)]
            route = [item for item_tuple in route for item in item_tuple]
            return route

        else:
            logging.debug("There is no path between "+str(origin_node)+" and "+str(destination_node))
            return None


    def sum_indicator_on_route(self, route, indicator, detail_type=False):
        total_indicator = 0
        all_edges = [item for item in route if len(item) == 2]
        # if indicator == "intl_rail_weight":
        #     print("self[2586][2579]['intl_rail_weight']",self[2586][2579]["intl_rail_weight"])
        for edge in all_edges:
            # if indicator == "intl_rail_weight":
            #     print(edge, self[edge[0]][edge[1]][indicator])
            total_indicator += self[edge[0]][edge[1]][indicator]

        # If detail_type == True, we print the indicator per edge categories
        details = []
        if detail_type:
            for edge in all_edges:
                new_edge = {}
                new_edge['id'] = self[edge[0]][edge[1]]['id']
                new_edge['type'] = self[edge[0]][edge[1]]['type']
                new_edge['multimodes'] = self[edge[0]][edge[1]]['multimodes']
                new_edge['special'] = self[edge[0]][edge[1]]['special']
                new_edge[indicator] = self[edge[0]][edge[1]][indicator]
                details += [new_edge]
            details = pd.DataFrame(details).fillna('N/A')
            # print(details)
            detail_per_cat = details.groupby(['type', 'multimodes', 'special'])[indicator].sum()
            print(detail_per_cat)
        return total_indicator



    def get_undisrupted_network(self):
        available_nodes = [node for node in self.nodes if self.node[node]['disruption_duration']==0]
        available_subgraph = self.subgraph(available_nodes)
        available_edges = [edge for edge in self.edges if self[edge[0]][edge[1]]['disruption_duration']==0]
        available_subgraph = available_subgraph.edge_subgraph(available_edges)
        return TransportNetwork(available_subgraph)
        
        
    # def get_available_network(self):
    #     available_edges = [
    #         edge 
    #         for edge in self.edges 
    #         if self[edge[0]][edge[1]]['current_load'] < self[edge[0]][edge[1]]['capacity']
    #     ]
    #     available_subgraph = self.edge_subgraph(available_edges)
    #     return TransportNetwork(available_subgraph)


    def disrupt_roads(self, disruption):
        # Disrupting nodes
        for node_id in disruption['node']:
            logging.info('Road node '+str(node_id)+
                ' gets disrupted for '+str(disruption['duration'])+ ' time steps')
            self.node[node_id]['disruption_duration'] = disruption['duration']
        # Disrupting edges
        for edge in self.edges:
            if self[edge[0]][edge[1]]['type'] == 'virtual':
                continue
            else:
                if self[edge[0]][edge[1]]['id'] in disruption['edge']:
                    logging.info('Road edge '+str(self[edge[0]][edge[1]]['id'])+
                        ' gets disrupted for '+str(disruption['duration'])+ ' time steps')                                            
                    self[edge[0]][edge[1]]['disruption_duration'] = disruption['duration']
            
            
    def update_road_state(self):
        for node in self.nodes:
            if self.node[node]['disruption_duration'] > 0:
                self.node[node]['disruption_duration'] -= 1
        for edge in self.edges:
            if self[edge[0]][edge[1]]['disruption_duration'] > 0:
                self[edge[0]][edge[1]]['disruption_duration'] -= 1
        #return subset of self
            
            
    def transport_shipment(self, commercial_link):
        # Select the route to transport the shimpment: main or alternative
        if commercial_link.current_route == 'main':
            route_to_take = commercial_link.route
        elif commercial_link.current_route == 'alternative':
            route_to_take = commercial_link.alternative_route
        else:
            route_to_take = []
        
        # Propagate the shipments
        for route_segment in route_to_take:
            if len(route_segment) == 2: #pass shipments to edges
                self[route_segment[0]][route_segment[1]]['shipments'][commercial_link.pid] = {
                    "from": commercial_link.supplier_id,
                    "to": commercial_link.buyer_id,
                    "quantity": commercial_link.delivery,
                    "tons": commercial_link.delivery_in_tons,
                    "product_type": commercial_link.product,
                    "flow_category": commercial_link.category,
                    "price": commercial_link.price
                }
            elif len(route_segment) == 1: #pass shipments to nodes
                self.node[route_segment[0]]['shipments'][commercial_link.pid] = {
                    "from": commercial_link.supplier_id,
                    "to": commercial_link.buyer_id,
                    "quantity": commercial_link.delivery,
                    "tons": commercial_link.delivery_in_tons,
                    "product_type": commercial_link.product,
                    "flow_category": commercial_link.category,
                    "price": commercial_link.price
                }

        # Propagate the load
        self.update_load_on_route(route_to_take, commercial_link.delivery_in_tons)


    def update_load_on_route(self, route, load):

        '''Affect a load to a route

        The current_load attribute of each edge in the route will be incread by the new load.
        A load is typically expressed in tons.

        If the current_load exceeds the capacity, then capacity_burden is added to both the 
        mode_weight and the capacity_weight.
        This will prevent firms from choosing this route
        '''
        # logging.info("Edge (2610, 2589): current_load "+str(self[2610][2589]['current_load']))
        capacity_burden = 1e5
        all_edges = [item for item in route if len(item) == 2]
        # if 'railways' in self.give_route_mode(route):
        #     print("self[2586][2579]['current_load']", self[2586][2579]['current_load'])
        for edge in all_edges:
            # if (edge[0] == 2610) & (edge[1] == 2589):
            #     logging.info('Edge '+str(edge)+": current_load "+str(self[edge[0]][edge[1]]['current_load']))
            # check that the edge to be loaded is not already over capacity
            # if (self[edge[0]][edge[1]]['current_load'] > self[edge[0]][edge[1]]['capacity']):
            #     logging.info('Edge '+str(edge)+" ("+self[edge[0]][edge[1]]['type']\
            #         +") is already over capacity and will be loaded more!")
            # Add the load
            self[edge[0]][edge[1]]['current_load'] += load
            # If it exceeds capacity, add the capacity_burden to both the mode_weight and the capacity_weight
            if (self[edge[0]][edge[1]]['current_load'] > self[edge[0]][edge[1]]['capacity']):
                logging.info('Edge '+str(edge)+" ("+self[edge[0]][edge[1]]['type']\
                    +") has exceeded its capacity ("+str(self[edge[0]][edge[1]]['capacity'])+")")
                self[edge[0]][edge[1]]["capacity_weight"] += capacity_burden
                for mode_weight in self.mode_weights:
                    self[edge[0]][edge[1]][mode_weight] += capacity_burden
                # print("self[edge[0]][edge[1]][current_load]", self[edge[0]][edge[1]]['current_load'])


    def reset_current_loads(self, route_optimization_weight):
        '''Reset current_load to 0

        If an edge was burdened due to capacity exceedance, we remove the burden
        '''
        for edge in self.edges:
            self[edge[0]][edge[1]]['current_load'] = 0

        self.defineWeights(route_optimization_weight)


    def give_route_mode(self, route):
        '''Which mode is used on the route?

        Return the list of transport mode used along the route
        '''
        modes = []
        all_edges = [item for item in route if len(item) == 2]
        for edge in all_edges:
            modes += [self[edge[0]][edge[1]]['type']]
        return list(dict.fromkeys(modes))


    def check_edge_in_route(self, route, searched_edge):
        all_edges = [item for item in route if len(item) == 2]
        for edge in all_edges:
            if (searched_edge[0] == edge[0]) and (searched_edge[1] == edge[1]):
                return True
        return False


    def remove_shipment(self, commercial_link):
        """Look for the shipment corresponding to the commercial link
        in any edges and nodes of the main and alternative route,
        and remove it
        """
        route_to_take = commercial_link.route + commercial_link.alternative_route
        for route_segment in route_to_take:
            if len(route_segment) == 2: #segment is an edge
                if commercial_link.pid in self[route_segment[0]][route_segment[1]]['shipments'].keys():
                    del self[route_segment[0]][route_segment[1]]['shipments'][commercial_link.pid]
            elif len(route_segment) == 1: #segment is a node
                if commercial_link.pid in self.node[route_segment[0]]['shipments'].keys():
                    del self.node[route_segment[0]]['shipments'][commercial_link.pid]

    
    def compute_flow_per_segment(self, flow_types=['total']):
        """
        Sum all flow of each 'flow_type' per transport edge

        The flow type are given as a list in the flow_types argument.
        It can corresponds to:
        - "total": sum of all flows
        - one of the CommercialLink.category, i.e., 'domestic_B2B', 
        'domestic_B2C', 'import', 'export'
        - one of the CommerialLink.product, i.e., the sectors

        Parameters
        ----------
        flow_types : list of string
            Flow type to evaluate

        Returns
        -------
        Nothing
        """
        for edge in self.edges():
            if self[edge[0]][edge[1]]['type'] != 'virtual':
                for flow_type in flow_types:
                    # either total
                    if flow_type == 'total':
                        self[edge[0]][edge[1]]['flow_'+flow_type] = sum([
                            shipment['quantity'] 
                            for shipment in self[edge[0]][edge[1]]["shipments"].values()
                        ])
                    # or flow category
                    elif flow_type in ['domestic_B2C', 'domestic_B2B', 'import', 'export', 'transit']:
                        self[edge[0]][edge[1]]['flow_'+flow_type] = sum([
                            shipment['quantity'] 
                            for shipment in self[edge[0]][edge[1]]["shipments"].values() 
                            if shipment['flow_category'] == flow_type
                        ])
                    # or product type
                    else: 
                        self[edge[0]][edge[1]]['flow_'+flow_type] = sum([
                            shipment['quantity'] 
                            for shipment in self[edge[0]][edge[1]]["shipments"].values() 
                            if shipment['product_type'] == flow_type
                        ])

  
    def evaluate_normal_traffic(self, sectorId_to_volumeCoef=None):
        self.evaluate_traffic(sectorId_to_volumeCoef)
        self.congestionned_edges = []
        for edge in self.edges():
            if self[edge[0]][edge[1]]['type'] == 'virtual':
                continue
            self[edge[0]][edge[1]]['traffic_normal'] = self[edge[0]][edge[1]]['traffic_current']
            self[edge[0]][edge[1]]['congestion'] = 0


    def evaluate_congestion(self, sectorId_to_volumeCoef=None):
        self.evaluate_traffic(sectorId_to_volumeCoef)
        self.congestionned_edges = []
        for edge in self.edges():
            if self[edge[0]][edge[1]]['type'] == 'virtual':
                continue
            self[edge[0]][edge[1]]['congestion'] = congestion_function(
                self[edge[0]][edge[1]]['traffic_current'], 
                self[edge[0]][edge[1]]['traffic_normal']
            )
            if self[edge[0]][edge[1]]['congestion'] > 1e-6:
                self.congestionned_edges += [edge]


    def evaluate_traffic(self, sectorId_to_volumeCoef=None):
        # If we have a correspondance of sector moneraty flow to volume,
        # we identify the sector that generate volume
        if sectorId_to_volumeCoef is not None:
            sectors_causing_congestion = [
                sector 
                for sector, coefficient in sectorId_to_volumeCoef.items() 
                if coefficient > 0
            ]

        for edge in self.edges():
            if self[edge[0]][edge[1]]['type'] == 'virtual':
                continue
            # If we have a correspondance of sector moneraty flow to volume,
            # we use volume
            if sectorId_to_volumeCoef is not None:
                volume = 0
                for sector_id in sectors_causing_congestion:
                    list_montetary_flows = [
                        shipment['quantity'] 
                        for shipment in self[edge[0]][edge[1]]["shipments"].values() 
                        if shipment['product_type'] == sector_id
                    ]
                    volume += sectorId_to_volumeCoef[sector_id] * sum(list_montetary_flows)
                self[edge[0]][edge[1]]['traffic_current'] = volume
            # Otherwise we use montery flow directly
            else:
                monetary_value_of_flows = sum([
                    shipment['quantity'] 
                    for shipment in self[edge[0]][edge[1]]["shipments"].values()
                ])
                self[edge[0]][edge[1]]['traffic_current'] = monetary_value_of_flows


    def reinitialize_flows_and_disruptions(self):
        for node in self.nodes:
            self.nodes[node]['disruption_duration'] = 0
            self.nodes[node]['shipments'] = {}
        for edge in self.edges:
            self[edge[0]][edge[1]]['disruption_duration'] = 0
            self[edge[0]][edge[1]]['shipments'] = {}
            self[edge[0]][edge[1]]['congestion'] = 0
            self[edge[0]][edge[1]]['current_load'] = 0
