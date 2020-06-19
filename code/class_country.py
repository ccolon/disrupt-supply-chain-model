import random
import pandas as pd
import math
import logging

from class_commerciallink import CommercialLink
from functions import rescale_values, \
    generate_weights_from_list, \
    determine_suppliers_and_weights,\
    identify_firms_in_each_sector,\
    identify_special_transport_nodes

class Country(object):

    def __init__(self, pid=None, qty_sold=None, qty_purchased=None, odpoint=None, 
        purchase_plan=None, transit_from=None, transit_to=None, supply_importance=None, 
        usd_per_ton=None):
        # Instrinsic parameters
        self.pid = pid
        self.usd_per_ton = usd_per_ton
        self.odpoint = odpoint
        
        # Parameter based on data
        # self.entry_points = entry_points or []
        self.transit_from = transit_from or {}
        self.transit_to = transit_to or {}
        self.supply_importance = supply_importance

        # Parameters depending on supplier-buyer network
        self.clients = {}
        self.purchase_plan = purchase_plan or {}
        self.qty_sold = qty_sold or {}
        self.qty_purchased = qty_purchased or {}
        self.qty_purchased_perfirm = {}

        # Variable
        self.generalized_transport_cost = 0
        self.usd_transported = 0
        self.tons_transported = 0
        self.tonkm_transported = 0
        self.extra_spending = 0
        self.consumption_loss = 0

    def reset_variables(self):
        self.generalized_transport_cost = 0
        self.usd_transported = 0
        self.tons_transported = 0
        self.tonkm_transported = 0
        self.extra_spending = 0
        self.consumption_loss = 0
        
    
    def create_transit_links(self, graph, country_list):
        for selling_country_pid, quantity in self.transit_from.items():
            selling_country_object = [country for country in country_list if country.pid==selling_country_pid][0]
            graph.add_edge(selling_country_object, self,
                       object=CommercialLink(
                           pid=str(selling_country_pid)+'to'+str(self.pid),
                           product='transit',
                           product_type="transit", #suppose that transit type are non service, material stuff
                           category="transit",
                           supplier_id=selling_country_pid,
                           buyer_id=self.pid))
            graph[selling_country_object][self]['weight'] = 1
            self.purchase_plan[selling_country_pid] = quantity
            selling_country_object.clients[self.pid] = {'sector':self.pid, 'share':0}


    def select_suppliers(self, graph, firm_list, country_list, sector_table, transport_nodes):
        # Select other country as supplier: transit flows
        self.create_transit_links(graph, country_list)
            
        # Select Tanzanian suppliers
        ## Identify firms from each sectors
        dic_sector_to_firmid = identify_firms_in_each_sector(firm_list)
        share_exporting_firms = sector_table.set_index('sector')['share_exporting_firms'].to_dict()
        ## Identify odpoints which exports
        export_odpoints = identify_special_transport_nodes(transport_nodes, "export")
        ## Identify sectors to buy from
        present_sectors = list(set(list(dic_sector_to_firmid.keys())))
        sectors_to_buy_from = list(self.qty_purchased.keys())
        present_sectors_to_buy_from = list(set(present_sectors) & set(sectors_to_buy_from))
        ## For each one of these sectors, select suppliers
        supplier_selection_mode = {
            "importance_export": {
                "export_odpoints": export_odpoints,
                "bonus": 10
            }
        }
        for sector in present_sectors_to_buy_from: #only select suppliers from sectors that are present
            # Identify potential suppliers
            potential_supplier_pid = dic_sector_to_firmid[sector]
            # Evaluate how much to select
            nb_selected_suppliers = math.ceil(
                len(dic_sector_to_firmid[sector])*share_exporting_firms[sector]
            )
            # Select supplier and weights
            selected_supplier_ids, supplier_weights = determine_suppliers_and_weights(
                potential_supplier_pid,
                nb_selected_suppliers,
                firm_list,
                mode=supplier_selection_mode)
               # Materialize the link
            for supplier_id in selected_supplier_ids:
                # For each supplier, create an edge in the economic network
                graph.add_edge(firm_list[supplier_id], self,
                           object=CommercialLink(
                               pid=str(supplier_id)+'to'+str(self.pid),
                               product=sector,
                               product_type=firm_list[supplier_id].sector_type,
                               category="export",
                               supplier_id=supplier_id,
                               buyer_id=self.pid))
                # Associate a weight
                weight = supplier_weights.pop(0)
                graph[firm_list[supplier_id]][self]['weight'] = weight
                # Households save the name of the retailer, its sector, its weight, and adds it to its purchase plan
                self.qty_purchased_perfirm[supplier_id] = {
                    'sector': sector, 
                    'weight': weight, 
                    'amount': self.qty_purchased[sector] * weight
                }
                self.purchase_plan[supplier_id] = self.qty_purchased[sector] * weight
                # The supplier saves the fact that it exports to this country. 
                # The share of sales cannot be calculated now, we put 0 for the moment
                firm_list[supplier_id].clients[self.pid] = {'sector':self.pid, 'share':0}
            
            
    def send_purchase_orders(self, graph):
        for edge in graph.in_edges(self):
            try:
                quantity_to_buy = self.purchase_plan[edge[0].pid]
            except KeyError:
                print("Country "+self.pid+": No purchase plan for supplier", edge[0].pid)
                quantity_to_buy = 0
            graph[edge[0]][self]['object'].order = quantity_to_buy

            
    def decide_routes(self, graph, transport_network, route_optimization_weight):
        self.usual_transport_cost = 0
        for edge in graph.out_edges(self):
            if edge[1].pid == -1: # we do not create route for households
                continue
            elif edge[1].odpoint == -1: # we do not create route for service firms 
                continue
            else:
                #Find rounte
                origin_node = self.odpoint
                destination_node = edge[1].odpoint
                route = transport_network.provide_shortest_route(
                    origin_node, destination_node,
                    route_weight=route_optimization_weight)
                #Store it into commercial link object
                graph[self][edge[1]]['object'].route = route
                distance, route_time_cost, cost_per_ton = transport_network.giveRouteCaracteristics(route)
                graph[self][edge[1]]['object'].route_length = distance
                graph[self][edge[1]]['object'].route_time_cost = route_time_cost
                graph[self][edge[1]]['object'].route_cost_per_ton = cost_per_ton

            
    def deliver_products(self, graph, transport_network, route_optimization_weight):
        """ The quantity to be delivered is the quantity that was ordered (no rationning takes place)
        """
        self.generalized_transport_cost = 0
        self.usd_transported = 0
        self.tons_transported = 0
        self.tonkm_transported = 0
        self.qty_sold = 0
        for edge in graph.out_edges(self):
            graph[self][edge[1]]['object'].delivery = graph[self][edge[1]]['object'].order

            explicit_service_firm = True
            if explicit_service_firm:
                # If send services, no use of transport network
                if graph[self][edge[1]]['object'].product_type in ['utility', 'transport', 'services']:
                    graph[self][edge[1]]['object'].price = graph[self][edge[1]]['object'].eq_price
                    self.qty_sold += graph[self][edge[1]]['object'].delivery
                # Otherwise, send shipment through transportation network     
                else:
                    self.send_shipment(
                        graph[self][edge[1]]['object'], 
                        transport_network,
                        route_optimization_weight
                    )
            else:
                if (edge[1].odpoint != -1): # to non service firms, send shipment through transportation network                   
                    self.send_shipment(
                        graph[self][edge[1]]['object'], 
                        transport_network,
                        route_optimization_weight
                    )
                else: # if it sends to service firms, nothing to do. price is equilibrium price
                    graph[self][edge[1]]['object'].price = graph[self][edge[1]]['object'].eq_price
                    self.qty_sold += graph[self][edge[1]]['object'].delivery


    def send_shipment(self, commercial_link, transport_network, route_optimization_weight):
        """Only apply to B2B flows 
        """
        if len(commercial_link.route)==0:
            raise ValueError("Country "+str(self.pid)+
                ": commercial link "+str(commercial_link.pid)+
                " is not associated to any route, I cannot send any shipment to client "+
                str(commercial_link.pid))
    
        if self.check_route_avaibility(commercial_link, transport_network, 'main') == 'available':
            # If the normal route is available, we can send the shipment as usual and pay the usual price
            commercial_link.current_route = 'main'
            commercial_link.price = commercial_link.eq_price
            transport_network.transport_shipment(commercial_link)
            
            self.generalized_transport_cost += commercial_link.route_time_cost + commercial_link.delivery / (self.usd_per_ton*1e-6) * commercial_link.route_cost_per_ton
            self.usd_transported += commercial_link.delivery
            self.tons_transported += commercial_link.delivery / (self.usd_per_ton*1e-6)
            self.tonkm_transported += commercial_link.delivery / (self.usd_per_ton*1e-6) *commercial_link.route_length
            self.qty_sold += commercial_link.delivery
            return 0

        # If there is a disruption, we try the alternative route, if there is any
        if (len(commercial_link.alternative_route)>0) & (self.check_route_avaibility(commercial_link, transport_network, 'alternative') == 'available'):
            commercial_link.current_route = 'alternative'
            route = commercial_link.alternative_route
        # Otherwise we have to find a new one
        else:
            origin_node = self.odpoint
            destination_node = commercial_link.route[-1][0]
            route = transport_network.available_subgraph()\
                                     .provide_shortest_route(
                                        origin_node, destination_node,
                                        route_weight=route_optimization_weight)
            # We evaluate the cost of this new route
            if route is not None:
                commercial_link.alternative_route = route
                distance, route_time_cost, cost_per_ton = transport_network.giveRouteCaracteristics(route)
                commercial_link.alternative_route_length = distance
                commercial_link.alternative_route_time_cost = route_time_cost
                commercial_link.alternative_route_cost_per_ton = cost_per_ton
        
        if route is not None:
            commercial_link.current_route = 'alternative'
            self.generalized_transport_cost += commercial_link.alternative_route_time_cost + commercial_link.delivery / (self.usd_per_ton*1e-6) * commercial_link.alternative_route_cost_per_ton
            self.usd_transported += commercial_link.delivery
            self.tons_transported += commercial_link.delivery / (self.usd_per_ton*1e-6)
            self.tonkm_transported += commercial_link.delivery / (self.usd_per_ton*1e-6) * commercial_link.alternative_route_length
            self.qty_sold += commercial_link.delivery

            if False: #relative cost change with actual bill
                new_transport_bill = commercial_link.delivery / (self.usd_per_ton*1e-6) * commercial_link.alternative_route_cost_per_ton
                normal_transport_bill = commercial_link.delivery / (self.usd_per_ton*1e-6) * commercial_link.route_cost_per_ton
                added_transport_bill = max(new_transport_bill - normal_transport_bill, 0)
                relative_cost_change = added_transport_bill/normal_transport_bill
                relative_price_change_transport = 0.2 * relative_cost_change
                total_relative_price_change = relative_price_change_transport
                commercial_link.price = commercial_link.eq_price * (1 + total_relative_price_change)

            elif True: #actual repercussion de la bill
                added_costUSD_per_ton = max(commercial_link.alternative_route_cost_per_ton - commercial_link.route_cost_per_ton, 0)
                added_costUSD_per_mUSD = added_costUSD_per_ton / (self.usd_per_ton*1e-6)
                added_costmUSD_per_mUSD = added_costUSD_per_mUSD*1e-6
                commercial_link.price = commercial_link.eq_price + added_costmUSD_per_mUSD
                relative_price_change_transport = commercial_link.price / commercial_link.eq_price - 1
                
            else:
                # We translate this real cost into transport cost
                relative_cost_change = (commercial_link.alternative_route_time_cost - commercial_link.route_time_cost)/commercial_link.route_time_cost
                relative_price_change_transport = 0.2 * relative_cost_change
                # With that, we deliver the shipment
                total_relative_price_change = relative_price_change_transport
                commercial_link.price = commercial_link.eq_price * (1 + total_relative_price_change)
                commercial_link.current_route = 'alternative'

            transport_network.transport_shipment(commercial_link)
            # Print information
            logging.debug("Country "+str(self.pid)+": found an alternative route to client "+
                str(commercial_link.buyer_id)+", it is costlier by "+
                '{:.0f}'.format(100*relative_price_change_transport)+"%, price is "+
                '{:.4f}'.format(commercial_link.price)+" instead of "+
                '{:.4f}'.format(commercial_link.eq_price))
        
        else:
            logging.debug("Country "+str(self.pid)+": because of disruption, there is"+
                "no route between me and client "+str(commercial_link.buyer_id))
            # We do not write how the input price would have changed
            commercial_link.price = commercial_link.eq_price
            # We do not pay the transporter, so we don't increment the transport cost

                    
    def check_route_avaibility(self, commercial_link, transport_network, which_route='main'):
        
        if which_route=='main':
            route_to_check = commercial_link.route
        elif which_route=='alternative':
            route_to_check = commercial_link.alternative_route
        else:
            KeyError('Wrong value for parameter which_route, admissible values are main and alternative')
        
        res = 'available'
        for route_segment in route_to_check:
            if len(route_segment) == 2:
                if transport_network[route_segment[0]][route_segment[1]]['disruption_duration'] > 0:
                    res = 'disrupted'
                    break
            if len(route_segment) == 1:
                if transport_network.node[route_segment[0]]['disruption_duration'] > 0:
                    res = 'disrupted'
                    break
        return res


                    
    def receive_products_and_pay(self, graph, transport_network):
        self.extra_spending = 0
        self.consumption_loss = 0
        for edge in graph.in_edges(self):
            if (edge[0].odpoint == -1): # if buys service, get directly from commercial link
                self.receive_service_and_pay(graph[edge[0]][self]['object'])
            else: # else collect through transport network
                self.receive_shipment_and_pay(graph[edge[0]][self]['object'], transport_network)


    def receive_service_and_pay(self, commercial_link):
        quantity_delivered = commercial_link.delivery
        commercial_link.payment = quantity_delivered * commercial_link.price
        self.extra_spending += quantity_delivered * (commercial_link.price - commercial_link.eq_price)
        

    def receive_shipment_and_pay(self, commercial_link, transport_network):
        """Firm look for shipments in the transport nodes it is locatedd
        It takes those which correspond to the commercial link 
        It receives them, thereby removing them from the transport network
        Then it pays the corresponding supplier along the commecial link
        """
        #quantity_intransit = commercial_link.delivery
        quantity_delivered = 0
        price = 1
        if commercial_link.pid in transport_network.node[self.odpoint]['shipments'].keys():
            quantity_delivered += transport_network.node[self.odpoint]['shipments'][commercial_link.pid]['quantity']
            price = transport_network.node[self.odpoint]['shipments'][commercial_link.pid]['price']
            transport_network.remove_shipment(commercial_link)
        # Increment extra spending
        self.extra_spending += quantity_delivered * (price - commercial_link.eq_price)
        # Increment consumption loss
        self.consumption_loss += commercial_link.delivery - quantity_delivered
        # Log if quantity received does not match order
        if abs(commercial_link.delivery - quantity_delivered) > 1e-6:
            logging.debug("Agent "+str(self.pid)+": quantity delivered by "+
                str(commercial_link.supplier_id)+" is "+str(quantity_delivered)+
                ". It was supposed to be "+str(commercial_link.delivery)+".")
        # Make payment
        commercial_link.payment = quantity_delivered * price
        
        
        

    def evaluate_commercial_balance(self, graph):
        exports = sum([graph[self][edge[1]]['object'].payment for edge in graph.out_edges(self)])
        imports = sum([graph[edge[0]][self]['object'].payment for edge in graph.in_edges(self)])
        print("Country "+self.pid+": imports "+str(imports)+" from Tanzania and export "+str(exports)+" to Tanzania")
        
        
    def add_congestion_malus2(self, graph, transport_network): 
        """Congestion cost are perceived costs, felt by firms, but they do not influence prices paid to transporter, hence do not change price
        """
        if len(transport_network.congestionned_edges) > 0:
            # for each client
            for edge in graph.out_edges(self):
                if graph[self][edge[1]]['object'].current_route == 'main':
                    route_to_check = graph[self][edge[1]]['object'].route
                elif graph[self][edge[1]]['object'].current_route == 'alternative':
                    route_to_check = graph[self][edge[1]]['object'].alternative_route
                else:
                    continue
                # check if the route currently used is congestionned
                if len(set(route_to_check) & set(transport_network.congestionned_edges)) > 0:
                    # if it is, we add its cost to the generalized cost model
                    self.generalized_transport_cost += transport_network.giveCongestionCostOfTime(route_to_check)