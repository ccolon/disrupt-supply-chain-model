import networkx as nx
import geopandas as gpd
import logging

from functions import rescale_values, congestion_function


class TransportNetwork(nx.Graph):
    
    def add_transport_node(self, node_id, all_nodes_data): #used in add_transport_edge_with_nodes
        node_attributes = ["nodename", "region", "geometry", "nodenumber"]
        node_data = all_nodes_data.loc[all_nodes_data['nodenumber']==node_id, node_attributes].iloc[0].to_dict()
        node_data['shipments'] = {}
        node_data['disruption_duration'] = 0
        node_data['firms_there'] = []
        node_data['type'] = 'road'
        self.add_node(node_id, **node_data)
               
            
    def add_transport_edge(self, edge_id, all_edges_data): #not used, use add_transport_edge_with_nodes instead
        edge_attributes = ['roadlabel', 'roadclass', 'kmpaved', 'kmunpaved', 'cor_name', "geometry"]
        edge_data = all_edges_data.loc[all_edges_data['link']==edge_id, edge_attributes].iloc[0].to_dict()
        edge_data['type'] = 'road'
        start_and_end_id = all_edges_data.loc[all_edges_data['link']==edge_id, ["startumber", "endnoumber"]].iloc[0].tolist()
        self.add_edge(start_and_end_id[0], start_and_end_id[1], **edge_data)
        
        
    def add_transport_edge_with_nodes(self, edge_id, all_edges_data, all_nodes_data): # used
        edge_attributes = ['link', 'roadlabel', 'roadclass', 'kmpaved', 'kmunpaved', 'cor_name', "geometry", "time_cost", 'cost_travel_time', 'cost_variability']
        edge_data = all_edges_data.loc[all_edges_data['link']==edge_id, edge_attributes].iloc[0].to_dict()
        edge_data['type'] = 'road'
        start_and_end_id = all_edges_data.loc[all_edges_data['link']==edge_id, ["startumber", "endnoumber"]].iloc[0].tolist()
        # Creating the start and end nodes
        self.add_transport_node(start_and_end_id[0], all_nodes_data)
        self.add_transport_node(start_and_end_id[1], all_nodes_data)
        # Creating the edge
        self.add_edge(start_and_end_id[0], start_and_end_id[1], **edge_data)
        self[start_and_end_id[0]][start_and_end_id[1]]['node_tuple'] = (start_and_end_id[0], start_and_end_id[1])
        self[start_and_end_id[0]][start_and_end_id[1]]['shipments'] = {}
        self[start_and_end_id[0]][start_and_end_id[1]]['disruption_duration'] = 0
        
        
    def connect_country(self, country):
        self.add_node(country.pid, **{'type':'virtual'})
        for transit_point in country.transit_points: #ATT so far works for road only
            self.add_edge(transit_point, country.pid, **{'type':'virtual', 'time_cost':1000}) # high time cost to avoid that algo goes through countries

            
    def remove_countries(self, country_list):
        country_node_to_remove = list(set(self.nodes) & set([country.pid for country in country_list]))
        for country in country_node_to_remove:
            self.remove_node(country)
            
        
    def giveRouteCost(self, route):
        time_cost = 1 #cost cannot be 0
        for segment in route:
            if len(segment) == 2: #only edges have costs 
                if self[segment[0]][segment[1]]['type'] != 'virtual':
                    time_cost += self[segment[0]][segment[1]]['time_cost']
        return time_cost
        
        
    def giveRouteCostAndTransportUnitCost(self, route):
        time_cost = 1 #cost cannot be 0
        cost_per_ton = 0
        for segment in route:
            if len(segment) == 2: #only edges have costs 
                if self[segment[0]][segment[1]]['type'] != 'virtual':
                    time_cost += self[segment[0]][segment[1]]['time_cost']
                    cost_per_ton += self.graph['unit_cost']['road']['paved'] * self[segment[0]][segment[1]]['kmpaved'] + self.graph['unit_cost']['road']['unpaved'] * self[segment[0]][segment[1]]['kmunpaved']
                    
        return time_cost, cost_per_ton
    
    
    def giveRouteCaracteristics(self, route):
        distance = 0 # km
        time_cost = 1 #USD, cost cannot be 0
        cost_per_ton = 0 #USD/ton
        for segment in route:
            if len(segment) == 2: #only edges have costs 
                if self[segment[0]][segment[1]]['type'] != 'virtual':
                    distance += self[segment[0]][segment[1]]['kmpaved'] + self[segment[0]][segment[1]]['kmunpaved']
                    time_cost += self[segment[0]][segment[1]]['time_cost']
                    cost_per_ton += self.graph['unit_cost']['road']['paved'] * self[segment[0]][segment[1]]['kmpaved'] + self.graph['unit_cost']['road']['unpaved'] * self[segment[0]][segment[1]]['kmunpaved']
                    
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
        
        
    def giveRouteDistance(self, route):
        distance = 0
        for segment in route:
            if len(segment) == 2: #only edges have distances 
                distance += self[segment[0]][segment[1]]['kmpaved'] + self[segment[0]][segment[1]]['kmunpaved']
        return distance
    
    
    def locate_firms_on_nodes(self, firm_list):
        for node_id in self.nodes:
            self.node[node_id]['firms_there'] = []
        for firm in firm_list:
            if firm.location != -1:
                try:
                    self.node[firm.location]['firms_there'].append(firm.pid)
                except KeyError:
                    logging.error('Transport network has no node numbered: '+str(firm.location))
    
    
    def provide_shortest_route(self, origin_node, destination_node):
        if (origin_node not in self.nodes) or (destination_node not in self.nodes):
            return None
        elif nx.has_path(self, origin_node, destination_node):
            sp = nx.shortest_path(self, origin_node, destination_node, weight="time_cost")
            route = [[(sp[0],)]] + [[(sp[i], sp[i+1]), (sp[i+1],)] for i in range(0,len(sp)-1)]
            route = [item for item_tuple in route for item in item_tuple]
            return route
        else:
            return None

        
    def available_subgraph(self):
        available_nodes = [node for node in self.nodes if self.node[node]['disruption_duration']==0]
        available_subgraph = self.subgraph(available_nodes)
        available_edges = [edge for edge in self.edges if self[edge[0]][edge[1]]['disruption_duration']==0]
        available_subgraph = available_subgraph.edge_subgraph(available_edges)
        return TransportNetwork(available_subgraph)
        
        
    def disrupt_roads(self, disrupted_roads, duration=1):
        logging.debug('Disrupting roads')
        for road_node_nb in disrupted_roads['node_nb']:
            logging.info('Road node '+str(road_node_nb)+' gets disrupted for '+str(duration)+ ' time steps')
            self.node[road_node_nb]['disruption_duration'] = duration
        for edge in self.edges:
            if self[edge[0]][edge[1]]['type'] == 'virtual':
                continue
            else:
                if self[edge[0]][edge[1]]['link'] in disrupted_roads['edge_link']:
                    logging.info('Road edge '+str(self[edge[0]][edge[1]]['link'])+' gets disrupted for '+str(duration)+ ' time steps')                                            
                    self[edge[0]][edge[1]]['disruption_duration'] = duration
            
            
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
                    "price": commercial_link.price
                }
            elif len(route_segment) == 1: #pass shipments to nodes
                self.node[route_segment[0]]['shipments'][commercial_link.pid] = {
                    "from": commercial_link.supplier_id,
                    "to": commercial_link.buyer_id,
                    "quantity": commercial_link.delivery,
                    "product_type": commercial_link.product,
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

    
    def compute_flow_per_segment(self, sectors=['total']):
        for edge in self.edges():
            if self[edge[0]][edge[1]]['type'] != 'virtual':
                for sector in sectors:
                    if str(sector) == 'total':
                        self[edge[0]][edge[1]]['flow_'+str(sector)] = sum([shipment['quantity'] for shipment in self[edge[0]][edge[1]]["shipments"].values()])
                    elif str(sector) == 'domestic':
                        self[edge[0]][edge[1]]['flow_domestic'] = sum([shipment['quantity'] for shipment in self[edge[0]][edge[1]]["shipments"].values() if len(str(shipment['product_type']))<=2])
                    else: 
                        self[edge[0]][edge[1]]['flow_'+str(sector)] = sum([shipment['quantity'] for shipment in self[edge[0]][edge[1]]["shipments"].values() if shipment['product_type'] == sector])

  
    def evaluate_normal_traffic(self, sectorId_to_volumeCoef=None):
        self.congestionned_edges = []
        if sectorId_to_volumeCoef is not None:
            sectors_causing_congestion = [sector for sector, coefficient in sectorId_to_volumeCoef.items() if coefficient > 0]
        for edge in self.edges():
            if self[edge[0]][edge[1]]['type'] != 'virtual':
                if sectorId_to_volumeCoef is not None:
                    volume = 0
                    for sector_id in sectors_causing_congestion:
                        list_montetary_flows = [shipment['quantity'] for shipment in self[edge[0]][edge[1]]["shipments"].values() if shipment['product_type'] == sector_id]
                        volume += sectorId_to_volumeCoef[sector_id] * sum(list_montetary_flows)
                    self[edge[0]][edge[1]]['traffic_normal'] = volume
                else:
                    monetary_value_of_flows = sum([shipment['quantity'] for shipment in self[edge[0]][edge[1]]["shipments"].values()])
                    self[edge[0]][edge[1]]['traffic_normal'] = monetary_value_of_flows
                self[edge[0]][edge[1]]['traffic_current'] = self[edge[0]][edge[1]]['traffic_normal']
                self[edge[0]][edge[1]]['congestion'] = 0


    def evaluate_congestion(self, sectorId_to_volumeCoef=None):
        self.congestionned_edges = []
        if sectorId_to_volumeCoef is not None:
            sectors_causing_congestion = [sector for sector, coefficient in sectorId_to_volumeCoef.items() if coefficient > 0]
        for edge in self.edges():
            if self[edge[0]][edge[1]]['type'] != 'virtual':
                if sectorId_to_volumeCoef is not None:
                    volume = 0
                    for sector_id in sectors_causing_congestion:
                        list_montetary_flows = [shipment['quantity'] for shipment in self[edge[0]][edge[1]]["shipments"].values() if shipment['product_type'] == sector_id]
                        volume += sectorId_to_volumeCoef[sector_id] * sum(list_montetary_flows)
                    self[edge[0]][edge[1]]['traffic_current'] = volume
                else:
                    monetary_value_of_flows = sum([shipment['quantity'] for shipment in self[edge[0]][edge[1]]["shipments"].values()])
                    self[edge[0]][edge[1]]['traffic_current'] = monetary_value_of_flows
                self[edge[0]][edge[1]]['congestion'] = congestion_function(self[edge[0]][edge[1]]['traffic_current'], self[edge[0]][edge[1]]['traffic_normal'])
                if self[edge[0]][edge[1]]['congestion'] > 1e-6:
                    self.congestionned_edges += [edge]


    def reinitialize_flows_and_disruptions(self):
        for node in self.nodes:
            self.nodes[node]['disruption_duration'] = 0
            self.nodes[node]['shipments'] = {}
        for edge in self.edges:
            self[edge[0]][edge[1]]['disruption_duration'] = 0
            self[edge[0]][edge[1]]['shipments'] = {}
            self[edge[0]][edge[1]]['congestion'] = 0
