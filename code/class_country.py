import random
import pandas as pd
import math
import logging

import class_commerciallink
from functions import rescale_values, \
    generate_weights_from_list, \
    determine_suppliers_and_weights,\
    identify_firms_in_each_sector,\
    identify_special_transport_nodes,\
    transformUSDtoTons, agent_decide_initial_routes,\
    agent_receive_products_and_pay


class Country(object):

    def __init__(self, pid=None, qty_sold=None, qty_purchased=None, odpoint=None, 
        purchase_plan=None, transit_from=None, transit_to=None, supply_importance=None, 
        usd_per_ton=None):
        # Instrinsic parameters
        self.agent_type = "country"
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
                       object=class_commerciallink.CommercialLink(
                           pid=str(selling_country_pid)+'->'+str(self.pid),
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
                           object=class_commerciallink.CommercialLink(
                               pid=str(supplier_id)+'->'+str(self.pid),
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

           
    def choose_route(self, transport_network, 
        origin_node, destination_node, 
        possible_transport_modes):
        
        # If possible_transport_modes is "roads", then simply pick the shortest road route
        if possible_transport_modes == "roads":
            route = transport_network.provide_shortest_route(origin_node,
                destination_node, route_weight="road_weight")
            return route, "roads"
        
        # If possible_transport_modes is "intl_multimodes",
        capacity_burden = 1e5
        if possible_transport_modes == "intl_multimodes":
            # pick routes for each modes
            modes = ['intl_road_shv', 'intl_road_vnm', 'intl_rail', 'intl_river']
            routes = { 
                mode: transport_network.provide_shortest_route(origin_node,
                    destination_node, route_weight=mode+"_weight")
                for mode in modes
            }
            # compute associated weight and capacity_weight
            modes_weight = { 
                mode: {
                    mode+"_weight": transport_network.sum_indicator_on_route(route, mode+"_weight"),
                    "weight": transport_network.sum_indicator_on_route(route, "weight", detail_type=False),
                    "capacity_weight": transport_network.sum_indicator_on_route(route, "capacity_weight")
                }
                for mode, route in routes.items()
            }
            # print(self.pid, modes_weight)
            # remove any mode which is over capacity (where capacity_weight > capacity_burden)
            for mode, route in routes.items():
                if mode != "intl_rail":
                    if transport_network.check_edge_in_route(route, (2610, 2589)):
                        print("(2610, 2589) in", mode)
                # if weight_dic['capacity_weight'] >= capacity_burden:
                #     print(mode, "will be eliminated")
            # if modes_weight['intl_rail']['capacity_weight'] >= capacity_burden:
            #     print("intl_rail", "will be eliminated")
            # else:
            #     print("intl_rail", "will not be eliminated")

            modes_weight = { 
                mode: weight_dic['weight']
                for mode, weight_dic in modes_weight.items()
                if weight_dic['capacity_weight'] < capacity_burden
            }
            if len(modes_weight) == 0:
                logging.warning("All transport modes are over capacity, no route selected!")
                return None
            # and select one route choosing random weighted choice
            selection_weights = rescale_values(list(modes_weight.values()), minimum=0, maximum=0.5)
            selection_weights = [1-w for w in selection_weights]
            selected_mode = random.choices(
                list(modes_weight.keys()), 
                weights=selection_weights, 
                k=1
            )[0]
            route = routes[selected_mode]
            # print("Country "+str(self.pid)+" chooses "+selected_mode+
            #     " to serve a client located "+str(destination_node))
            # print(transport_network.give_route_mode(route))
            return route, selected_mode

        raise ValueError("The transport_mode attributes of the commerical link\
                          does not belong to ('roads', 'intl_multimodes')")


    def decide_initial_routes(self, graph, transport_network, transport_modes,
        account_capacity, monetary_unit_flow):

        agent_decide_initial_routes(self, graph, transport_network, transport_modes,
        account_capacity, monetary_unit_flow)
        '''for edge in graph.out_edges(self):
            if edge[1].pid == -1: # we do not create route for households
                continue
            elif edge[1].odpoint == -1: # we do not create route for service firms if explicit_service_firms = False
                continue
            else:
                # Get the id of the orign and destination node
                origin_node = self.odpoint
                destination_node = edge[1].odpoint
                # Define the type of transport mode to use
                cond_from = (transport_modes['from'] == self.pid) #self is a country
                if isinstance(edge[1], Firm): #see what is the other end
                    cond_to = (transport_modes['to'] == "domestic")
                else:
                    cond_to = (transport_modes['to'] == edge[1].pid)
                    # we have not implemented the "sector" condition
                transport_mode = transport_modes.loc[cond_from & cond_to, "transport_mode"].iloc[0]
                graph[self][edge[1]]['object'].transport_mode = transport_mode
                route, selected_mode = self.choose_route(
                    transport_network=transport_network, 
                    origin_node=origin_node, 
                    destination_node=destination_node, 
                    possible_transport_modes=transport_mode
                )
                # Store it into commercial link object
                graph[self][edge[1]]['object'].storeRouteInformation(
                    route=route,
                    transport_mode=selected_mode,
                    main_or_alternative="main",
                    transport_network=transport_network
                )
                # Update the "current load" on the transport network
                # if current_load exceed burden, then add burden to the weight
                if account_capacity:
                    new_load_in_usd = graph[self][edge[1]]['object'].order
                    new_load_in_tons = transformUSDtoTons(new_load_in_usd, monetary_unit_flow, self.usd_per_ton)
                    transport_network.update_load_on_route(route, new_load_in_tons)
'''
            
    def deliver_products(self, graph, transport_network,
                        monetary_unit_transport_cost, monetary_unit_flow, 
                        cost_repercussion_mode, explicit_service_firm):
        """ The quantity to be delivered is the quantity that was ordered (no rationning takes place)
        """
        self.generalized_transport_cost = 0
        self.usd_transported = 0
        self.tons_transported = 0
        self.tonkm_transported = 0
        self.qty_sold = 0
        for edge in graph.out_edges(self):
            if graph[self][edge[1]]['object'].order == 0:
                logging.debug("Agent "+str(self.pid)+": "+
                    str(graph[self][edge[1]]['object'].buyer_id)+" is my client but did not order")
                continue
            graph[self][edge[1]]['object'].delivery = graph[self][edge[1]]['object'].order
            graph[self][edge[1]]['object'].delivery_in_tons = \
                transformUSDtoTons(graph[self][edge[1]]['object'].order, monetary_unit_flow, self.usd_per_ton)

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
                        monetary_unit_transport_cost,
                        monetary_unit_flow,
                        cost_repercussion_mode
                    )
            else:
                if (edge[1].odpoint != -1): # to non service firms, send shipment through transportation network                   
                    self.send_shipment(
                        graph[self][edge[1]]['object'], 
                        transport_network,
                        monetary_unit_transport_cost,
                        monetary_unit_flow,
                        cost_repercussion_mode
                    )
                else: # if it sends to service firms, nothing to do. price is equilibrium price
                    graph[self][edge[1]]['object'].price = graph[self][edge[1]]['object'].eq_price
                    self.qty_sold += graph[self][edge[1]]['object'].delivery


    def send_shipment(self, commercial_link, transport_network,
        monetary_unit_transport_cost, monetary_unit_flow, cost_repercussion_mode):

        if commercial_link.delivery_in_tons == 0:
            print("delivery", commercial_link.delivery)
            print("supplier_id", commercial_link.supplier_id)
            print("buyer_id", commercial_link.buyer_id)

        monetary_unit_factor = {
            "mUSD": 1e6,
            "kUSD": 1e3,
            "USD": 1
        }
        factor = monetary_unit_factor[monetary_unit_flow]
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
            
            self.generalized_transport_cost += commercial_link.route_time_cost \
                + commercial_link.delivery_in_tons * commercial_link.route_cost_per_ton
            self.usd_transported += commercial_link.delivery
            self.tons_transported += commercial_link.delivery_in_tons
            self.tonkm_transported += commercial_link.delivery_in_tons *commercial_link.route_length
            self.qty_sold += commercial_link.delivery
            return 0

        # If there is a disruption, we try the alternative route, if there is any
        if (len(commercial_link.alternative_route)>0) & \
           (self.check_route_avaibility(commercial_link, transport_network, 'alternative') == 'available'):
            commercial_link.current_route = 'alternative'
            route = commercial_link.alternative_route
        # Otherwise we have to find a new one
        else:
            origin_node = self.odpoint
            destination_node = commercial_link.route[-1][0]
            route, selected_mode = self.choose_route(
                transport_network=transport_network.get_undisrupted_network(), 
                origin_node=origin_node,
                destination_node=destination_node, 
                possible_transport_modes=commercial_link.possible_transport_modes
            )
            # We evaluate the cost of this new route
            if route is not None:
                commercial_link.storeRouteInformation(
                    route=route,
                    transport_mode=selected_mode,
                    main_or_alternative="alternative",
                    transport_network=transport_network
                )
        
        # If the alternative route is available, or if we discovered one, we proceed
        if route is not None:
            commercial_link.current_route = 'alternative'
            # Calculate contribution to generalized transport cost, to usd/tons/tonkms transported
            self.generalized_transport_cost += commercial_link.alternative_route_time_cost \
                + commercial_link.delivery_in_tons * commercial_link.alternative_route_cost_per_ton
            self.usd_transported += commercial_link.delivery
            self.tons_transported += commercial_link.delivery_in_tons
            self.tonkm_transported += commercial_link.delivery_in_tons * commercial_link.alternative_route_length
            self.qty_sold += commercial_link.delivery

            if cost_repercussion_mode == "type1": #relative cost change with actual bill
                # Calculate relative increase in routing cost
                new_transport_bill = commercial_link.delivery_in_tons * commercial_link.alternative_route_cost_per_ton
                normal_transport_bill = commercial_link.delivery_in_tons * commercial_link.route_cost_per_ton
                relative_cost_change = max(new_transport_bill - normal_transport_bill, 0)/normal_transport_bill
                # Translate that into an increase in transport costs in the balance sheet
                relative_price_change_transport = 0.2 * relative_cost_change
                total_relative_price_change = relative_price_change_transport
                commercial_link.price = commercial_link.eq_price * (1 + total_relative_price_change)

            elif cost_repercussion_mode == "type2": #actual repercussion de la bill
                added_costUSD_per_ton = max(commercial_link.alternative_route_cost_per_ton - commercial_link.route_cost_per_ton, 0)
                added_costUSD_per_mUSD = added_costUSD_per_ton / (self.usd_per_ton/factor)
                added_costmUSD_per_mUSD = added_costUSD_per_mUSD/factor
                commercial_link.price = commercial_link.eq_price + added_costmUSD_per_mUSD
                relative_price_change_transport = commercial_link.price / commercial_link.eq_price - 1
                
            elif cost_repercussion_mode == "type3":
                # We translate this real cost into transport cost
                relative_cost_change = (commercial_link.alternative_route_time_cost - commercial_link.route_time_cost)/commercial_link.route_time_cost
                relative_price_change_transport = 0.2 * relative_cost_change
                # With that, we deliver the shipment
                total_relative_price_change = relative_price_change_transport
                commercial_link.price = commercial_link.eq_price * (1 + total_relative_price_change)

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
        agent_receive_products_and_pay(self, graph, transport_network)

    #     self.extra_spending = 0
    #     self.consumption_loss = 0
    #     for edge in graph.in_edges(self):
    #         if (edge[0].odpoint == -1): # if buys service, get directly from commercial link
    #             self.receive_service_and_pay(graph[edge[0]][self]['object'])
    #         else: # else collect through transport network
    #             self.receive_shipment_and_pay(graph[edge[0]][self]['object'], transport_network)


    # def receive_service_and_pay(self, commercial_link):
    #     quantity_delivered = commercial_link.delivery
    #     commercial_link.payment = quantity_delivered * commercial_link.price
    #     self.extra_spending += quantity_delivered * (commercial_link.price - commercial_link.eq_price)
        
    # def receive_shipment_and_pay(self, commercial_link, transport_network):
    #     """Firm look for shipments in the transport nodes it is locatedd
    #     It takes those which correspond to the commercial link 
    #     It receives them, thereby removing them from the transport network
    #     Then it pays the corresponding supplier along the commecial link
    #     """
    #     #quantity_intransit = commercial_link.delivery
    #     quantity_delivered = 0
    #     price = 1
    #     if commercial_link.pid in transport_network.node[self.odpoint]['shipments'].keys():
    #         quantity_delivered += transport_network.node[self.odpoint]['shipments'][commercial_link.pid]['quantity']
    #         price = transport_network.node[self.odpoint]['shipments'][commercial_link.pid]['price']
    #         transport_network.remove_shipment(commercial_link)
    #     # Increment extra spending
    #     self.extra_spending += quantity_delivered * (price - commercial_link.eq_price)
    #     # Increment consumption loss
    #     self.consumption_loss += commercial_link.delivery - quantity_delivered
    #     # Log if quantity received does not match order
    #     if abs(commercial_link.delivery - quantity_delivered) > 1e-6:
    #         logging.debug("Agent "+str(self.pid)+": quantity delivered by "+
    #             str(commercial_link.supplier_id)+" is "+str(quantity_delivered)+
    #             ". It was supposed to be "+str(commercial_link.delivery)+".")
    #     # Make payment
    #     commercial_link.payment = quantity_delivered * price      
        
        

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