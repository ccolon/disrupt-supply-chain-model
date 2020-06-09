import networkx as nx
import geopandas as gpd
import logging

from functions import rescale_values, congestion_function


class TransportNetwork(nx.Graph):
    
    def add_transport_node(self, node_id, all_nodes_data): #used in add_transport_edge_with_nodes
        node_attributes = ["id", "geometry", "special"]
        node_data = all_nodes_data.loc[node_id, node_attributes].to_dict()
        node_data['shipments'] = {}
        node_data['disruption_duration'] = 0
        node_data['firms_there'] = []
        node_data['type'] = 'road'
        self.add_node(node_id, **node_data)
        

    def add_transport_edge_with_nodes(self, edge_id, all_edges_data, all_nodes_data):
        # Selecting data
        edge_attributes = ['id', "type", 'surface', "geometry", "class", "km", 'special',
            "cost_per_ton", "travel_time", "time_cost", 'cost_travel_time', 'cost_variability']
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
        
        
    # def giveRouteDistance(self, route):
    #     distance = 0
    #     for segment in route:
    #         if len(segment) == 2: #only edges have distances 
    #             distance += self[segment[0]][segment[1]]['km']
    #     return distance
    
    
    def locate_firms_on_nodes(self, firm_list):
        for node_id in self.nodes:
            self.node[node_id]['firms_there'] = []
        for firm in firm_list:
            if firm.odpoint != -1:
                try:
                    self.node[firm.odpoint]['firms_there'].append(firm.pid)
                except KeyError:
                    logging.error('Transport network has no node numbered: '+str(firm.odpoint))
    
    
    def provide_shortest_route(self, origin_node, destination_node):
        if (origin_node not in self.nodes):
            logging.warning("Origin node "+str(origin_node)+" not in the available transport network")
            return None

        elif (destination_node not in self.nodes):
            logging.warning("Destination node "+str(destination_node)+" not in the available transport network")
            return None

        elif nx.has_path(self, origin_node, destination_node):
            sp = nx.shortest_path(self, origin_node, destination_node, weight="cost_per_ton")#time_cost
            route = [[(sp[0],)]] + [[(sp[i], sp[i+1]), (sp[i+1],)] for i in range(0,len(sp)-1)]
            route = [item for item_tuple in route for item in item_tuple]
            return route

        else:
            logging.warning("There is no path between "+str(origin_node)+" and "+str(destination_node))
            return None

        
    def available_subgraph(self):
        available_nodes = [node for node in self.nodes if self.node[node]['disruption_duration']==0]
        available_subgraph = self.subgraph(available_nodes)
        available_edges = [edge for edge in self.edges if self[edge[0]][edge[1]]['disruption_duration']==0]
        available_subgraph = available_subgraph.edge_subgraph(available_edges)
        return TransportNetwork(available_subgraph)
        
        
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
        if commercial_link.current_route == 'main':
            route_to_take = commercial_link.route
        elif commercial_link.current_route == 'alternative':
            route_to_take = commercial_link.alternative_route
        else:
            route_to_take = []
            
        for route_segment in route_to_take:
            if len(route_segment) == 2: #pass shipments to edges
                self[route_segment[0]][route_segment[1]]['shipments'][commercial_link.pid] = {
                    "from": commercial_link.supplier_id,
                    "to": commercial_link.buyer_id,
                    "quantity": commercial_link.delivery,
                    "product_type": commercial_link.product,
                    "flow_category": commercial_link.category,
                    "price": commercial_link.price
                }
            elif len(route_segment) == 1: #pass shipments to nodes
                self.node[route_segment[0]]['shipments'][commercial_link.pid] = {
                    "from": commercial_link.supplier_id,
                    "to": commercial_link.buyer_id,
                    "quantity": commercial_link.delivery,
                    "product_type": commercial_link.product,
                    "flow_category": commercial_link.category,
                    "price": commercial_link.price
                }


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
                    if flow_type == 'total':
                        self[edge[0]][edge[1]]['flow_'+flow_type] = sum([
                            shipment['quantity'] 
                            for shipment in self[edge[0]][edge[1]]["shipments"].values()
                        ])
                    elif flow_type in ['domestic_B2B', 'import', 'export', 'transit']:
                        self[edge[0]][edge[1]]['flow_'+flow_type] = sum([
                            shipment['quantity'] 
                            for shipment in self[edge[0]][edge[1]]["shipments"].values() 
                            if shipment['flow_category'] == flow_type
                        ])
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
