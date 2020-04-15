from functions import purchase_planning_function, production_function, evaluate_safety_days, generate_weights, compute_distance_from_arcmin, rescale_values
from class_commerciallink import CommercialLink
import random
import networkx as nx
import shapely
import logging
import numpy as np

class Firm(object):
    
    def __init__(self, pid, location=0, sector=0, input_mix=None, target_margin=0.2, utilization_rate=0.8,
                 importance=1, long=None, lat=None, geometry=None,
                 suppliers=None, clients=None, production=0, safety_days=1, reactivity_rate=1, usd_per_ton=2864):
        # Parameters depending on data
        self.pid = pid
        self.location = location
        self.long = long
        self.lat = lat
        self.geometry = geometry
        self.importance = importance
        self.sector = sector
        self.input_mix = input_mix or {}
        self.usd_per_ton = usd_per_ton

        # Free parameters
        if input_mix is None:
            self.safety_days = safety_days
        else:
            self.safety_days = {key: safety_days for key in input_mix.keys()}
        self.reactivity_rate = reactivity_rate
        self.eq_production_capacity = production / utilization_rate
        self.utilization_rate = utilization_rate
        self.target_margin = target_margin

        # Parameters depending on supplier-buyer network
        self.suppliers = suppliers or {}
        self.clients = clients or {}
        
        # Parameters sets at initialization
        self.eq_finance = {"sales":0, 'costs':{"input":0, "transport":0, "other":0}}
        self.eq_profit = 0
        self.eq_price = 1
        self.eq_total_order = 0

        # Variables, all initialized
        self.production = production
        self.production_target = production
        self.production_capacity = production / utilization_rate
        self.purchase_plan = {}
        self.order_book = {}
        self.total_order = 0
        self.input_needs = {}
        self.rationing = 1
        self.eq_needs = {}
        self.current_safety_days = {}
        self.inventory = {}
        self.product_stock = 0
        self.profit = 0
        self.finance = {"sales":0, 'costs':{"input":0, "transport":0, "other":0}}
        self.delta_price_input = 0
        self.generalized_transport_cost = 0
        self.usd_transported = 0
        self.tons_transported = 0
        self.tonkm_transported = 0

    def reset_variables(self):
        self.eq_finance = {"sales":0, 'costs':{"input":0, "transport":0, "other":0}}
        self.eq_profit = 0
        self.eq_price = 1
        self.production = 0
        self.production_target = 0
        self.production_capacity = self.eq_production_capacity
        self.purchase_plan = {}
        self.order_book = {}
        self.total_order = 0
        self.input_needs = {}
        self.rationing = 1
        self.eq_needs = {}
        self.current_safety_days = {}
        self.inventory = {}
        self.product_stock = 0
        self.profit = 0
        self.finance = {"sales":0, 'costs':{"input":0, "transport":0, "other":0}}
        self.delta_price_input = 0
        self.generalized_transport_cost = 0
        self.usd_transported = 0
        self.tons_transported = 0
        self.tonkm_transported = 0

        
        
    def add_noise_to_geometry(self, noise_level=1e-5):
        self.geometry = shapely.geometry.point.Point(self.long+noise_level*random.uniform(0,1), self.lat+noise_level*random.uniform(0,1))
        
    
    def distance_to_other(self, other_firm):
        if (self.location == -1) or (other_firm.location == -1): #if virtual firms
            return 1
        else:
            return compute_distance_from_arcmin(self.long, self.lat, other_firm.long, other_firm.lat)
    
    
    def select_suppliers(self, graph, firm_list, country_list, nb_suppliers_per_sector=1, weight_localization=1):
        for sector_id, sector_weight in self.input_mix.items():

            # Select international suppliers
            if isinstance(sector_id, str):
                # Inspect potential suppliers outside the country
                importance_threshold = 1e-6
                potential_supplier_pid = [country.pid for country in country_list if country.supply_importance>importance_threshold]  # Identify countries as suppliers if the corresponding sector does export
                importance_of_each = [country.supply_importance for country in country_list if country.supply_importance>importance_threshold]
                weight_choice = np.array(importance_of_each)
            
            # Select domestic suppliers
            else:
                potential_supplier_pid = [firm.pid for firm in firm_list if firm.sector == sector_id] # Identify the id of potential suppliers among the other firms
                if sector_id == self.sector:
                    potential_supplier_pid.remove(self.pid) # remove oneself
                distance_to_each = rescale_values([self.distance_to_other(firm_list[firm_pid]) for firm_pid in potential_supplier_pid]) # Compute distance to each of them (vol d oiseau)
                importance_of_each = rescale_values([firm_list[firm_pid].importance for firm_pid in potential_supplier_pid]) # Get importance for each of them
                weight_choice = np.array(importance_of_each) / (np.array(distance_to_each)**weight_localization)
            # Select supplier
            
            if random.uniform(0,1) < nb_suppliers_per_sector-1:
                nb_suppliers_to_choose = 2
                if nb_suppliers_to_choose > len(potential_supplier_pid):
                    nb_suppliers_to_choose = 1
            else:
                nb_suppliers_to_choose = 1

            weight_choice /= weight_choice.sum()
            selected_supplier_id = np.random.choice(potential_supplier_pid, p=weight_choice, size=nb_suppliers_to_choose, replace=False).tolist()
            supplier_weights = generate_weights(nb_suppliers_to_choose) # Generate one random weight per number of supplier, sum to 1
            for supplier_id in selected_supplier_id:
                # Retrieve the supplier object from the id
                if type(supplier_id) == str:
                    supplier_object = [country for country in country_list if country.pid==supplier_id][0]
                else:
                    supplier_object = firm_list[supplier_id]
                # Create an edge in the graph
                graph.add_edge(supplier_object, self,
                               object=CommercialLink(
                                   pid=str(supplier_id)+"to"+str(self.pid),
                                   product=sector_id,
                                   supplier_id=supplier_id,
                                   buyer_id=self.pid)
                              )
                # Associate a weight, which includes the I/O technical coefficient
                supplier_weight = supplier_weights.pop()
                graph[supplier_object][self]['weight'] = sector_weight * supplier_weight
                # The firm saves the name of the supplier, its sector, its weight (without I/O technical coefficient)
                self.suppliers[supplier_id] = {'sector':sector_id, 'weight':supplier_weight}
                # The supplier saves the name of the client, its sector. The share of sales cannot be calculated now
                supplier_object.clients[self.pid] = {'sector':self.sector, 'share':0, 'share_transport':0}
        
    
    def decide_routes(self, graph, transport_network):
        for edge in graph.out_edges(self):
            if edge[1].pid == -1: # we do not create route for households
                continue
            elif edge[1].location == -1: # we do not create route for service firms 
                continue
            else:
                origin_node = self.location
                destination_node = edge[1].location
                route = transport_network.provide_shortest_route(origin_node, destination_node)
                if route is not None:
                    graph[self][edge[1]]['object'].route = route
                    distance, route_time_cost, cost_per_ton = transport_network.giveRouteCaracteristics(route)
                    graph[self][edge[1]]['object'].route_length = distance
                    graph[self][edge[1]]['object'].route_time_cost = route_time_cost
                    graph[self][edge[1]]['object'].route_cost_per_ton = cost_per_ton
                else:
                    logging.error('Firm '+str(self.pid)+': I did not find any route from me to firm '+str(edge[1].pid))
                    raise Exception("\t\tFirm "+str(self.pid)+": there is no route between me and firm "+str(edge[1].pid))
    
    
    def calculate_client_share_in_sales(self):
        # Only works if the order book was computed
        self.total_order = sum([order for client_pid, order in self.order_book.items()])
        self.total_B2B_order = sum([order for client_pid, order in self.order_book.items() if client_pid != -1])
        for client_pid, info in self.clients.items():
            if self.total_order == 0:
                info['share'] = 0
                info['share_transport'] = 0
            else:
                info['share'] = self.order_book[client_pid] / self.total_order
            if self.total_B2B_order == 0:
                info['share_transport'] = 0
            else:
                if client_pid == -1:
                    info['share_transport'] = 0
                else:
                    info['share_transport'] = self.order_book[client_pid] / self.total_B2B_order
        
    
    def aggregate_orders(self):
        self.total_order = sum([order for client_pid, order in self.order_book.items()])

    def decide_production_plan(self):
        self.production_target = self.total_order - self.product_stock

        
    def calculate_price(self, graph, firm_list):
        """
        Evaluate the relative increase in price due to changes in input price
        In addition, upon delivery, price will be adjusted for each client to reflect potential rerouting
        """ 
        if self.check_if_supplier_changed_price(graph, firm_list):
            if False:
                self.delta_price_input = self.calculate_input_induced_price_change(graph)
                logging.debug('Firm '+str(self.pid)+': Input prices have changed, I set my price to '+'{:.4f}'.format(self.eq_price*(1+self.delta_price_input))+" instead of "+str(self.eq_price))
                
            elif True:
                eq_theoretical_input_cost = 0
                current_theoretical_input_cost = 0
                for edge in graph.in_edges(self):
                    eq_theoretical_input_cost += graph[edge[0]][self]['object'].eq_price * graph[edge[0]][self]['weight']
                    current_theoretical_input_cost += graph[edge[0]][self]['object'].price * graph[edge[0]][self]['weight']
                added_input_cost = (current_theoretical_input_cost - eq_theoretical_input_cost) * self.total_order
                self.delta_price_input = added_input_cost / self.total_order
                logging.debug('Firm '+str(self.pid)+': Input prices have changed, I set my price to '+'{:.4f}'.format(self.eq_price*(1+self.delta_price_input/self.total_order))+" instead of "+str(self.eq_price))
        else:
            self.delta_price_input = 0

        
    def evaluate_input_needs(self):
        self.input_needs = {
            input_pid: self.input_mix[input_pid]*self.production_target
            for input_pid, mix in self.input_mix.items()
        }
        
        
    def decide_purchase_plan(self, mode="equilibrium"):
        """
        If mode="equilibrium", it aims to come back to equilibrium inventories
        If mode="reactive", it uses current orders to evaluate the target inventories
        """

        if mode=="reactive":
            ref_input_needs = self.input_needs
            
        elif mode=="equilibrium":
            ref_input_needs = self.eq_needs
            
        # Evaluate the current safety days
        self.current_safety_days = {
            input_id: (evaluate_safety_days(ref_input_needs[input_id], stock) if input_id in ref_input_needs.keys() else 0)
            for input_id, stock in self.inventory.items()
        }

        # Alert if there is less than a day of an input
        if True:
            for input_id, safety_days in self.current_safety_days.items():
                if safety_days is not None:
                    if safety_days < 1 - 1e-6:
                        if -1 in self.clients.keys():
                            sales_to_hh = self.clients[-1]['share'] * self.production_target
                        else:
                            sales_to_hh = 0
                        logging.debug('Firm '+str(self.pid)+" of sector "+str(self.sector)+" selling to households "+str(sales_to_hh)+" less than 1 day of inventory for input type "+str(input_id))
            
        # Evaluate purchase plan for each sector
        purchase_plan_per_sector = {
            input_id: purchase_planning_function(need, self.inventory[input_id], self.safety_days[input_id], self.reactivity_rate)
            #input_id: purchase_planning_function(need, self.inventory[input_id], self.safety_days_old, self.reactivity_rate)
            for input_id, need in ref_input_needs.items()
        }
        # Deduce the purchase plan for each supplier
        self.purchase_plan = {
            supplier_id: purchase_plan_per_sector[info['sector']] * info['weight']
            for supplier_id, info in self.suppliers.items()
        }


    def send_purchase_orders(self, graph):
        for edge in graph.in_edges(self):
            if edge[0].pid in self.purchase_plan.keys():
                quantity_to_buy = self.purchase_plan[edge[0].pid]
                if quantity_to_buy == 0:
                    logging.debug("Firm "+str(self.pid)+": I am not planning to buy anything from supplier "+str(edge[0].pid))
            else:
                logging.error("Firm "+str(self.pid)+": supplier "+str(edge[0].pid)+" is not in my purchase plan")
                quantity_to_buy = 0
            graph[edge[0]][self]['object'].order = quantity_to_buy

                
    def retrieve_orders(self, graph):
        for edge in graph.out_edges(self):
            quantity_ordered = graph[self][edge[1]]['object'].order
            self.order_book[edge[1].pid] = quantity_ordered

    
    def produce(self, mode="Leontief"):
        max_production = production_function(self.inventory, self.input_mix, mode)
        self.production = min([max_production, self.production_target, self.production_capacity])
        self.product_stock += self.production
        if mode=="Leontief":
            input_used = {input_id: self.production * mix for input_id, mix in self.input_mix.items()}
            self.inventory = {input_id: quantity - input_used[input_id] for input_id, quantity in self.inventory.items()}
        else:
            raise ValueError("Wrong mode chosen")

    
    def calculate_input_induced_price_change(self, graph):
        """The firm evaluates the input costs of producting one unit of output if it had to buy the inputs at current price
        It is a theoretical cost, because in simulations it may use inventory
        """
        eq_theoretical_input_cost = 0
        current_theoretical_input_cost = 0
        for edge in graph.in_edges(self):
            eq_theoretical_input_cost += graph[edge[0]][self]['object'].eq_price * graph[edge[0]][self]['weight']
            current_theoretical_input_cost += graph[edge[0]][self]['object'].price * graph[edge[0]][self]['weight']
        input_cost_share = eq_theoretical_input_cost / 1
        relative_change = (current_theoretical_input_cost - eq_theoretical_input_cost) / eq_theoretical_input_cost
        return relative_change * input_cost_share / (1 - self.target_margin)
        
    
    def check_if_supplier_changed_price(self, graph, firm_list):# firms could record the last price they paid their input
        for edge in graph.in_edges(self):
            if abs(graph[edge[0]][self]['object'].price - graph[edge[0]][self]['object'].eq_price) > 1e-6:
                if True:
                    if str(graph[edge[0]][self]['object'].supplier_id)[0] == "C":
                        sector_of_supplier = "C"
                        same_place = 0
                        distance_of_supplier = 0
                    else:
                        sector_of_supplier = firm_list[graph[edge[0]][self]['object'].supplier_id].sector
                        distance_of_supplier = self.distance_to_other(firm_list[graph[edge[0]][self]['object'].supplier_id])
                        if self.location == firm_list[graph[edge[0]][self]['object'].supplier_id].location:
                            same_place = 1
                        else:
                            same_place = 0
                    if -1 in self.clients.keys():
                        sales_to_hh = self.clients[-1]['share'] * self.production_target
                    else:
                        sales_to_hh = 0
                    
                    logging.debug("Firm "+str(+self.pid)+" of sector "+str(self.sector)+" who sells "+str(sales_to_hh)+" to households"+\
                    " has supplier "+str(graph[edge[0]][self]['object'].supplier_id)+" of sector "+str(sector_of_supplier)+" who is located at "+str(distance_of_supplier)+" increased price")
                return True
        return False
    
    
    def deliver_without_infrastructure(self, commercial_link):
        """ The firm deliver its products without using transportation infrastructure
        This case applies to service firm, and to nonservice firms selling to service firms (they are not localized) and to households 
        Note that we still account for transport cost, proportionnaly to the share of the clients
        Price can be higher than 1, if there are changes in price inputs
        """
        commercial_link.price = commercial_link.eq_price * (1 + self.delta_price_input)
        self.product_stock -= commercial_link.delivery
        self.finance['costs']['transport'] += self.clients[commercial_link.buyer_id]['share'] * self.eq_finance['costs']['transport']

    
    def deliver_products(self, graph, transport_network=None, rationing_mode="equal"):
        # Compute rationing factor
        if self.total_order == 0:
            logging.info('Firm '+str(self.pid)+': no one ordered to me')
            
        else:
            self.rationing = self.product_stock / self.total_order
            if self.rationing > 1 + 1e6:
                logging.debug('Firm '+str(self.pid)+': I have produced too much')
                self.rationing = 1
            
            if self.rationing >= 1-1e-6:
                quantity_to_deliver = {buyer_id: order for buyer_id, order in self.order_book.items()}
                
            else:
                logging.debug('Firm '+str(self.pid)+': I have to ration my clients by '+'{:.2f}'.format((1-self.rationing)*100)+'%')
                # Evaluate the quantity to deliver to each buyer
                if rationing_mode=="equal":
                    quantity_to_deliver = {buyer_id: order * self.rationing for buyer_id, order in self.order_book.items()}
                    
                elif rationing_mode=="household_first":
                    if -1 not in self.order_book.keys(): #no household orders to this firm
                        quantity_to_deliver = {buyer_id: order * self.rationing for buyer_id, order in self.order_book.items()}
                    elif len(self.order_book.keys())==1: #only households order to this firm
                        quantity_to_deliver = {-1: self.total_order}
                    else:
                        order_households = self.order_book[-1]
                        if order_households < self.product_stock:
                            remaining_product_stock = self.product_stock - order_households
                            if (self.total_order-order_households) <= 0:
                                logging.warning("Firm "+str(self.pid)+': '+str(self.total_order-order_households))
                            rationing_for_business = remaining_product_stock / (self.total_order-order_households)
                            quantity_to_deliver = {buyer_id: order * rationing_for_business for buyer_id, order in self.order_book.items() if buyer_id != -1}
                            quantity_to_deliver[-1] = order_households
                        else:
                            quantity_to_deliver = {buyer_id: 0 for buyer_id, order in self.order_book.items() if buyer_id != -1}
                            quantity_to_deliver[-1] = self.product_stock
                else:
                    raise ValueError('Wrong rationing_mode chosen')
            # We initialize transport costs, it will be updated for each shipment
            self.finance['costs']['transport'] = 0
            self.generalized_transport_cost = 0
            self.usd_transported = 0
            self.tons_transported = 0
            self.tonkm_transported = 0
            
            # For each client, we define the quantity to deliver then send the shipment 
            for edge in graph.out_edges(self):
                graph[self][edge[1]]['object'].delivery = quantity_to_deliver[edge[1].pid]
                
                # If it's B2B and no service client, we send to the transport network, price will be adjusted according to transport conditions
                if (self.location != -1) and (edge[1].location != -1) and (edge[1].pid != -1):
                    self.send_shipment(graph[self][edge[1]]['object'], transport_network)
                
                # If it's B2C, or B2B with service client, we send directly, and adjust price with input costs. There is still transport costs.
                elif (self.location == -1) or (edge[1].location == -1) or (edge[1].pid == -1):
                    self.deliver_without_infrastructure(graph[self][edge[1]]['object'])
                
                # If it's B2C, we send directly, and adjust price with input costs. There is still transport costs.
                else:
                    logging.error('There should not be this other case.')
                    
                
    def send_shipment(self, commercial_link, transport_network):
        """Only apply to B2B flows 
        """
        if len(commercial_link.route)==0:
            logging.error("Firm "+str(self.pid)+": commercial link "+str(commercial_link.pid)+" is not associated to any route, I cannot send any shipment to client "+str(commercial_link.pid))
            
        else:
            if self.check_route_avaibility(commercial_link, transport_network, 'main') == 'available':
                # If the normal route is available, we can send the shipment as usual and pay the usual price
                commercial_link.price = commercial_link.eq_price * (1 + self.delta_price_input)
                commercial_link.current_route = 'main'
                transport_network.transport_shipment(commercial_link)
                self.product_stock -= commercial_link.delivery
                self.generalized_transport_cost += commercial_link.route_time_cost + commercial_link.delivery / (self.usd_per_ton*1e-6) * commercial_link.route_cost_per_ton
                self.usd_transported += commercial_link.delivery
                self.tons_transported += commercial_link.delivery / (self.usd_per_ton*1e-6)
                self.tonkm_transported += commercial_link.delivery / (self.usd_per_ton*1e-6) *commercial_link.route_length
                
                self.finance['costs']['transport'] += self.clients[commercial_link.buyer_id]['share'] * self.eq_finance['costs']['transport']
                
            else:
                # If there is a disruption, we try the alternative route, if there is any
                if (len(commercial_link.alternative_route)>0) & (self.check_route_avaibility(commercial_link, transport_network, 'alternative') == 'available'):
                    route = commercial_link.alternative_route
                else:
                # Otherwise we have to find a new one
                    origin_node = self.location
                    destination_node = commercial_link.route[-1][0]
                    route = transport_network.available_subgraph().provide_shortest_route(origin_node, destination_node)
                    if route is not None: # if we find a new route, we save it as the alternative one
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
                
                    # We translate this real cost into transport cost
                    if False: #relative cost change with actual bill
                        new_transport_bill = commercial_link.delivery / (self.usd_per_ton*1e-6) * commercial_link.alternative_route_cost_per_ton
                        normal_transport_bill = commercial_link.delivery / (self.usd_per_ton*1e-6) * commercial_link.route_cost_per_ton
                        added_transport_bill = max(new_transport_bill - normal_transport_bill, 0)
                        relative_cost_change = added_transport_bill/normal_transport_bill
                        self.finance['costs']['transport'] += self.eq_finance['costs']['transport'] * self.clients[commercial_link.buyer_id]['share'] * (1 + relative_cost_change)
                        relative_price_change_transport = self.eq_finance['costs']['transport'] * relative_cost_change / ((1-self.target_margin) * self.eq_finance['sales'])
                        total_relative_price_change = self.delta_price_input + relative_price_change_transport
                        commercial_link.price = commercial_link.eq_price * (1 + total_relative_price_change)

                    elif True: #actual repercussion de la bill
                        added_costUSD_per_ton = max(commercial_link.alternative_route_cost_per_ton - commercial_link.route_cost_per_ton, 0)
                        added_costUSD_per_mUSD = added_costUSD_per_ton / (self.usd_per_ton*1e-6)
                        added_costmUSD_per_mUSD = added_costUSD_per_mUSD*1e-6
                        added_transport_bill = added_costmUSD_per_mUSD * commercial_link.delivery
                        self.finance['costs']['transport'] += self.eq_finance['costs']['transport'] + added_transport_bill
                        commercial_link.price = commercial_link.eq_price + self.delta_price_input + added_costmUSD_per_mUSD
                        relative_price_change_transport = commercial_link.price / (commercial_link.eq_price + self.delta_price_input) - 1
                        
                        logging.debug('Firm '+str(self.pid)+": qty "+str(commercial_link.delivery / (self.usd_per_ton*1e-6)) +
                        " increase in route cost per ton "+ str((commercial_link.alternative_route_cost_per_ton-commercial_link.route_cost_per_ton)/commercial_link.route_cost_per_ton)+
                        " increased bill mUSD "+str(added_costmUSD_per_mUSD*commercial_link.delivery))
                        
                    else:
                        relative_cost_change = (commercial_link.alternative_route_time_cost - commercial_link.route_time_cost)/commercial_link.route_time_cost
                        self.finance['costs']['transport'] += self.eq_finance['costs']['transport'] * self.clients[commercial_link.buyer_id]['share'] * (1 + relative_cost_change)
                        relative_price_change_transport = self.eq_finance['costs']['transport'] * relative_cost_change / ((1-self.target_margin) * self.eq_finance['sales'])
                        # With that, we deliver the shipment
                        total_relative_price_change = self.delta_price_input + relative_price_change_transport
                        commercial_link.price = commercial_link.eq_price * (1 + total_relative_price_change)
                    transport_network.transport_shipment(commercial_link)
                    self.product_stock -= commercial_link.delivery
                    # Print information
                    logging.debug("Firm "+str(self.pid)+": found an alternative route to "+str(commercial_link.buyer_id)+", it is costlier by "+'{:.0f}'.format(100*relative_price_change_transport)+'%'+", price is "+'{:.4f}'.format(commercial_link.price)+" instead of "+'{:.4f}'.format(commercial_link.eq_price*(1+self.delta_price_input)))
                
                else:
                    logging.debug('Firm '+str(self.pid)+": because of disruption, there is no route between me and firm "+str(commercial_link.buyer_id))
                    # We do not write how the input price would have changed
                    commercial_link.price = commercial_link.eq_price
                    commercial_link.current_route = 'none'
                    # We do not pay the transporter, so we don't increment the transport cost
                    # We set delivery to 0
                    commercial_link.delivery = 0
    
  
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
    
    
    def add_congestion_malus(self, graph, transport_network):
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
                    # if it is, compare actual cost with normal cost
                    actual_route_time_cost = transport_network.giveRouteCostWithCongestion(route_to_check)
                    
                    # If it is on the main route, then there was no previous price increase due to transport
                    if graph[self][edge[1]]['object'].current_route == 'main':
                        relative_cost_change_no_congestion = 0
                        relative_cost_change_with_congestion = (actual_route_time_cost - graph[self][edge[1]]['object'].route_time_cost)/graph[self][edge[1]]['object'].route_time_cost
                        self.finance['costs']['transport'] += self.eq_finance['costs']['transport'] * self.clients[edge[1].pid]['share'] * (1 + relative_cost_change_with_congestion)
                    
                    # Otherwise, we need to incremen the added price increase due to transport
                    elif graph[self][edge[1]]['object'].current_route == 'alternative':
                        relative_cost_change_no_congestion = (graph[self][edge[1]]['object'].alternative_route_time_cost - graph[self][edge[1]]['object'].route_time_cost)/graph[self][edge[1]]['object'].route_time_cost
                        relative_cost_change_with_congestion = (actual_route_time_cost - graph[self][edge[1]]['object'].route_time_cost)/graph[self][edge[1]]['object'].route_time_cost

                    # We increment financial costs
                    self.finance['costs']['transport'] += self.eq_finance['costs']['transport'] * self.clients[edge[1].pid]['share'] * (1 + relative_cost_change_with_congestion - relative_cost_change_no_congestion)

                    # We compute the new price increase due to transport and congestion
                    relative_price_change_transport_no_congestion = self.eq_finance['costs']['transport'] * relative_cost_change_no_congestion / ((1-self.target_margin) * self.eq_finance['sales'])
                    relative_price_change_transport_with_congestion = self.eq_finance['costs']['transport'] * relative_cost_change_with_congestion / ((1-self.target_margin) * self.eq_finance['sales'])
                    total_relative_price_change = self.delta_price_input + relative_price_change_transport_with_congestion
                    new_price = graph[self][edge[1]]['object'].eq_price * (1 + total_relative_price_change)
                    logging.debug('Firm '+str(self.pid)+": price of transport to "+str(edge[1].pid)+"is impacted by congestion. New price: "+str(new_price)+" vs. "+str(graph[self][edge[1]]['object'].price)+". Route cost with congestion: "+str(actual_route_time_cost)+" vs. normal: "+str(graph[self][edge[1]]['object'].route_time_cost)+", delta input: "+str(self.delta_price_input)+",  delta transport no congestion: "+str(relative_price_change_transport_no_congestion)+", delta congestion:"+str(relative_price_change_transport_with_congestion - relative_price_change_transport_no_congestion))
                    graph[self][edge[1]]['object'].price = new_price
                        
                    # Retransfer shipment
                    transport_network.transport_shipment(graph[self][edge[1]]['object'])


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
        for edge in graph.in_edges(self): 
            if (edge[0].location == -1) or (self.location == -1): # if service, directly
                self.receive_service_and_pay(graph[edge[0]][self]['object'])
            else: # else collect through transport network
                self.receive_shipment_and_pay(graph[edge[0]][self]['object'], transport_network)

                
    def receive_service_and_pay(self, commercial_link):
        quantity_delivered = commercial_link.delivery
        self.inventory[commercial_link.product] += quantity_delivered
        commercial_link.payment = quantity_delivered * commercial_link.price

        
    def receive_shipment_and_pay(self, commercial_link, transport_network):
        """Firm look for shipments in the transport nodes it is located
        It takes those which correspond to the commercial link 
        It receives them, thereby removing them from the transport network
        Then it pays the corresponding supplier along the commecial link
        """
        quantity_intransit = commercial_link.delivery
        quantity_delivered = 0
        price = 1
        if commercial_link.pid in transport_network.node[self.location]['shipments'].keys():
            quantity_delivered += transport_network.node[self.location]['shipments'][commercial_link.pid]['quantity']
            price = transport_network.node[self.location]['shipments'][commercial_link.pid]['price']
            transport_network.remove_shipment(commercial_link)
        self.inventory[commercial_link.product] += quantity_delivered
        if abs(quantity_intransit - quantity_delivered) > 1e-6:
            logging.debug("Firm "+str(self.pid)+": quantity delivered by firm"+str(commercial_link.supplier_id)+"("+str(quantity_delivered)+") differs from what was supposed to be delivered ("+str(commercial_link.delivery)+")")
        commercial_link.payment = quantity_delivered * price
        
        
    def evaluate_profit(self, graph):
        self.finance['sales'] = sum([graph[self][edge[1]]['object'].payment for edge in graph.out_edges(self)])
        self.finance['costs']['input'] = sum([graph[edge[0]][self]['object'].payment for edge in graph.in_edges(self)]) 
        self.profit = self.finance['sales'] - self.finance['costs']['input'] - self.finance['costs']['other'] - self.finance['costs']['transport']
        if self.finance['sales'] > 0:
            if abs(self.profit/self.finance['sales'] - self.target_margin) > 1e-3:
                logging.debug('Firm '+str(self.pid)+': my margin differs from the target one: '+'{:.3f}'.format(self.profit/self.finance['sales'])+' instead of '+str(self.target_margin))
        
           
           
    def print_info(self):
        print("\nFirm "+str(self.pid)+" from sector "+str(self.sector)+":")
        print("suppliers:", self.suppliers)
        print("clients:", self.clients)
        print("input_mix:", self.input_mix)
        print("order_book:", self.order_book, "; total_order:", self.total_order)
        print("input_needs:", self.input_needs)
        print("purchase_plan:", self.purchase_plan)
        print("inventory:", self.inventory)
        print("production:", self.production, "; production target:", self.production_target, "; product stock:", self.product_stock)
        print("profit:", self.profit, ";", self.finance)