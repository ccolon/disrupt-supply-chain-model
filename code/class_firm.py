from functions import purchase_planning_function, production_function, \
                      evaluate_inventory_duration, generate_weights, \
                      compute_distance_from_arcmin, rescale_values, \
                      transformUSDtoTons
from class_commerciallink import CommercialLink
import random
import networkx as nx
import shapely
import logging
import numpy as np

class Firm(object):
    
    def __init__(self, pid, odpoint=0, sector=0, sector_type=None, input_mix=None, target_margin=0.2, utilization_rate=0.8,
                 importance=1, long=None, lat=None, geometry=None,
                 suppliers=None, clients=None, production=0, inventory_duration_target=1, reactivity_rate=1, usd_per_ton=2864):
        # Parameters depending on data
        self.pid = pid
        self.odpoint = odpoint
        self.long = long
        self.lat = lat
        self.geometry = geometry
        self.importance = importance
        self.sector = sector
        self.sector_type = sector_type
        self.input_mix = input_mix or {}
        self.usd_per_ton = usd_per_ton

        # Free parameters
        if input_mix is None:
            self.inventory_duration_target = inventory_duration_target
        else:
            self.inventory_duration_target = {key: inventory_duration_target for key in input_mix.keys()}
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
        self.current_inventory_duration = {}
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
        self.current_inventory_duration = {}
        self.inventory = {}
        self.product_stock = 0
        self.profit = 0
        self.finance = {"sales":0, 'costs':{"input":0, "transport":0, "other":0}}
        self.delta_price_input = 0
        self.generalized_transport_cost = 0
        self.usd_transported = 0
        self.tons_transported = 0
        self.tonkm_transported = 0

        
    def initialize_ope_var_using_eq_production(self, eq_production):
        self.production_target = eq_production
        self.production = self.production_target
        self.eq_production_capacity = self.production_target / self.utilization_rate
        self.production_capacity = self.eq_production_capacity
        self.evaluate_input_needs()
        self.eq_needs = self.input_needs
        self.inventory = {
            input_id: need * (1+self.inventory_duration_target[input_id]) 
            for input_id, need in self.input_needs.items()
        }
        self.decide_purchase_plan()

        
    def initialize_fin_var_using_eq_cost(self, eq_production, eq_input_cost,
        eq_transport_cost, eq_other_cost):
        self.eq_finance['sales'] = eq_production
        self.eq_finance['costs']['input'] = eq_input_cost
        self.eq_finance['costs']['transport'] = eq_transport_cost
        self.eq_finance['costs']['other'] = eq_other_cost
        self.eq_profit = self.eq_finance['sales'] - sum(self.eq_finance['costs'].values())
        self.finance['sales'] = self.eq_finance['sales']
        self.finance['costs']['input'] = self.eq_finance['costs']['input']
        self.finance['costs']['transport'] = self.eq_finance['costs']['transport']
        self.finance['costs']['other'] = self.eq_finance['costs']['other']
        self.profit = self.eq_profit
        self.delta_price_input = 0


    def add_noise_to_geometry(self, noise_level=1e-5):
        self.geometry = shapely.geometry.point.Point(self.long+noise_level*random.uniform(0,1), self.lat+noise_level*random.uniform(0,1))
        
    
    def distance_to_other(self, other_firm):
        if (self.odpoint == -1) or (other_firm.odpoint == -1): #if virtual firms
            return 1
        else:
            return compute_distance_from_arcmin(self.long, self.lat, other_firm.long, other_firm.lat)
    
    
    def select_suppliers(self, graph, firm_list, country_list, 
        nb_suppliers_per_input=1, weight_localization=1, import_code='IMP'):
        """
        The firm selects its suppliers.

        The firm checks its input mix to identify which type of inputs are needed.
        For each type of input, it selects the appropriate number of suppliers.
        Choice of suppliers is random, based on distance to eligible suppliers and 
        their importance.

        If imports are needed, the firms select a country as supplier. Choice is 
        random, based on the country's importance.

        Parameters
        ----------
        graph : networkx.DiGraph
            Supply chain graph
        firm_list : list of Firms
            Generated by createFirms function
        country_list : list of Countries
            Generated by createCountriesfunction
        nb_suppliers_per_input : float between 1 and 2
            Nb of suppliers per type of inputs. If it is a decimal between 1 and 2,
            some firms will have 1 supplier, other 2 suppliers, such that the
            average matches the specified value.
        weight_localization : float
            Give weight to distance when choosing supplier. The larger, the closer
            the suppliers will be selected.
        import_code : string
            Code that identify imports in the input mix.

        Returns
        -------
        int
            0

        """
        for sector_id, sector_weight in self.input_mix.items():

            # If it is imports, identify international suppliers and calculate
            # their probability to be chosen, which is based on importance.
            if sector_id == import_code:
                # Identify countries as suppliers if the corresponding sector does export
                importance_threshold = 1e-6
                potential_supplier_pid = [
                    country.pid 
                    for country in country_list 
                    if country.supply_importance>importance_threshold
                ]  
                importance_of_each = [
                    country.supply_importance 
                    for country in country_list 
                    if country.supply_importance>importance_threshold
                ]
                prob_to_be_selected = np.array(importance_of_each)
                prob_to_be_selected /= prob_to_be_selected.sum()
            
            # For the other types of inputs, identify the domestic suppliers, and
            # calculate their probability to be chosen, based on distance and importance
            else:
                potential_supplier_pid = [firm.pid for firm in firm_list if firm.sector == sector_id] # Identify the id of potential suppliers among the other firms
                if sector_id == self.sector:
                    potential_supplier_pid.remove(self.pid) # remove oneself
                if len(potential_supplier_pid)==0:
                    raise ValueError("Firm "+str(self.pid)+
                        ": no potential supplier for input "+str(sector_id))
                # print("\n", self.pid, ":", str(len(potential_supplier_pid)), "for sector", sector_id)
                # print([
                #     self.distance_to_other(firm_list[firm_pid]) 
                #     for firm_pid in potential_supplier_pid
                # ])
                distance_to_each = rescale_values([
                    self.distance_to_other(firm_list[firm_pid]) 
                    for firm_pid in potential_supplier_pid
                ]) # Compute distance to each of them (vol d oiseau)
                # print(distance_to_each)
                importance_of_each = rescale_values([firm_list[firm_pid].importance for firm_pid in potential_supplier_pid]) # Get importance for each of them
                prob_to_be_selected = np.array(importance_of_each) / (np.array(distance_to_each)**weight_localization)
                prob_to_be_selected /= prob_to_be_selected.sum()
            
            # Determine the numbre of supplier(s) to select. 1 or 2.
            if random.uniform(0,1) < nb_suppliers_per_input-1:
                nb_suppliers_to_choose = 2
                if nb_suppliers_to_choose > len(potential_supplier_pid):
                    nb_suppliers_to_choose = 1
            else:
                nb_suppliers_to_choose = 1

            # Select the supplier(s). It there is 2 suppliers, then we generate 
            # random weight. It determines how much is bought from each supplier.
            selected_supplier_id = np.random.choice(potential_supplier_pid, 
                p=prob_to_be_selected, size=nb_suppliers_to_choose, replace=False).tolist()
            supplier_weights = generate_weights(nb_suppliers_to_choose) # Generate one random weight per number of supplier, sum to 1
            
            # For each new supplier, create a new CommercialLink in the supply chain network.
            for supplier_id in selected_supplier_id:
                # Retrieve the appropriate supplier object from the id
                # If it is a country we get it from the country list
                # It it is a firm we get it from the firm list
                if sector_id == import_code:
                    supplier_object = [country for country in country_list if country.pid==supplier_id][0]
                    link_category = 'import'
                    product_type = "imports"
                else:
                    supplier_object = firm_list[supplier_id]
                    link_category = 'domestic_B2B'
                    product_type = firm_list[supplier_id].sector_type
                # Create an edge in the graph
                graph.add_edge(supplier_object, self,
                               object=CommercialLink(
                                   pid=str(supplier_id)+"to"+str(self.pid),
                                   product=sector_id,
                                   product_type=product_type,
                                   category=link_category, 
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
        
    
    def choose_route(self, transport_network, 
        origin_node, destination_node, 
        possible_transport_modes):
        '''There are 3 types of weight to select routes
        - 'weight': the weight itself (e.g., cost per ton)
        - 'weight_with_capacity': same as weight, but burden added on edges on which 
        load > capacity. Burden is arbitrary, made to force agents to avoid it
        - 'road_weight', 'intl_road_weight', 'intl_rail_weight', 'intl_river_weight':
        same as weight, but burden added on specific edges to force firm choose a
        specific transport mode
        '''

        # If possible_transport_modes is "roads", then simply pick the shortest road route
        if possible_transport_modes == "roads":
            route = transport_network.provide_shortest_route(origin_node,
                destination_node, route_weight="road_weight")
            return route, "roads"
        
        # If possible_transport_modes is "intl_multimodes",
        capacity_burden = 1e5
        if possible_transport_modes == "intl_multimodes":
            # pick routes for each modes
            modes = ['intl_road', 'intl_rail', 'intl_river']
            routes = { 
                mode: transport_network.provide_shortest_route(origin_node,
                    destination_node, route_weight=mode+"_weight")
                for mode in modes
            }
            # compute associated weight and capacity_weight
            modes_weight = { 
                mode: {
                    mode+"_weight": transport_network.sum_indicator_on_route(route, mode+"_weight"),
                    "weight": transport_network.sum_indicator_on_route(route, "weight"),
                    "capacity_weight": transport_network.sum_indicator_on_route(route, "capacity_weight")
                }
                for mode, route in routes.items()
            }
            # remove any mode which is over capacity (where capacity_weight > capacity_burden)
            modes_weight = { 
                mode: weight_dic['weight']
                for mode, weight_dic in modes_weight.items()
                if weight_dic['capacity_weight'] < capacity_burden
            }
            if len(modes_weight) == 0:
                logging.warning("All transport modes are over capacity, no route selected!")
                return None
            # and select one route choosing random weighted choice
            selected_mode = random.choices(
                list(modes_weight.keys()), 
                weights=list(modes_weight.values()), 
                k=1
            )[0]
            # print("Firm "+str(self.pid)+" chooses "+selected_mode+
            #     " to serve a client located "+str(destination_node))
            route = routes[selected_mode]
            return route, selected_mode

        raise ValueError("The transport_mode attributes of the commerical link\
                          does not belong to ('roads', 'intl_multimodes')")


    def decide_initial_routes(self, graph, transport_network, transport_modes,
        account_capacity, monetary_unit_flow):
        for edge in graph.out_edges(self):
            if edge[1].pid == -1: # we do not create route for households
                continue
            elif edge[1].odpoint == -1: # we do not create route for service firms if explicit_service_firms = False
                continue
            else:
                # Get the id of the orign and destination node
                origin_node = self.odpoint
                destination_node = edge[1].odpoint
                # Define the type of transport mode to use
                cond_from = (transport_modes['from'] == "domestic") #self is a firm
                if isinstance(edge[1], Firm): #see what is the other end
                    cond_to = (transport_modes['to'] == "domestic")
                else:
                    cond_to = (transport_modes['to'] == edge[1].pid)
                    # we have not implemented the "sector" condition
                transport_mode = transport_modes.loc[cond_from & cond_to, "transport_mode"].iloc[0]
                graph[self][edge[1]]['object'].transport_mode = transport_mode
                # Choose the route and the corresponding mode
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
            # One way to compute it is commented.
            #     self.delta_price_input = self.calculate_input_induced_price_change(graph)
            #     logging.debug('Firm '+str(self.pid)+': Input prices have changed, I set '+
            #     "my price to "+'{:.4f}'.format(self.eq_price*(1+self.delta_price_input))+
            #     " instead of "+str(self.eq_price))

            # I compute how much would be my input cost to produce one unit of output
            # if I had to buy the input at this price
            eq_unitary_input_cost = 0
            est_unitary_input_cost_at_current_price = 0
            for edge in graph.in_edges(self):
                eq_unitary_input_cost += graph[edge[0]][self]['object'].eq_price * graph[edge[0]][self]['weight']
                est_unitary_input_cost_at_current_price += graph[edge[0]][self]['object'].price * graph[edge[0]][self]['weight']
            # I scale this added cost to my total orders
            self.delta_price_input = est_unitary_input_cost_at_current_price - eq_unitary_input_cost
            if self.delta_price_input is np.nan:
                print(self.delta_price_input)
                print(est_unitary_input_cost_at_current_price)
                print(eq_unitary_input_cost)
            # added_input_cost = (est_unitary_input_cost_at_current_price - eq_unitary_input_cost) * self.total_order
            # self.delta_price_input = added_input_cost / self.total_order
            logging.debug('Firm '+str(self.pid)+': Input prices have changed, I set my price to '+
                '{:.4f}'.format(self.eq_price*(1+self.delta_price_input))+
                " instead of "+str(self.eq_price))
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
        self.current_inventory_duration = {
            input_id: (evaluate_inventory_duration(ref_input_needs[input_id], stock) if input_id in ref_input_needs.keys() else 0)
            for input_id, stock in self.inventory.items()
        }

        # Alert if there is less than a day of an input
        if True:
            for input_id, inventory_duration in self.current_inventory_duration.items():
                if inventory_duration is not None:
                    if inventory_duration < 1 - 1e-6:
                        if -1 in self.clients.keys():
                            sales_to_hh = self.clients[-1]['share'] * self.production_target
                        else:
                            sales_to_hh = 0
                        logging.debug('Firm '+str(self.pid)+" of sector "+str(self.sector)+" selling to households "+str(sales_to_hh)+" less than 1 day of inventory for input type "+str(input_id))
            
        # Evaluate purchase plan for each sector
        purchase_plan_per_sector = {
            input_id: purchase_planning_function(need, self.inventory[input_id], self.inventory_duration_target[input_id], self.reactivity_rate)
            #input_id: purchase_planning_function(need, self.inventory[input_id], self.inventory_duration_old, self.reactivity_rate)
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
                # if True:
                #     if isinstance(graph[edge[0]][self]['object'].supplier_id, str):
                #         sector_of_supplier = "IMP"
                #         same_place = 0
                #         distance_of_supplier = 0
                #     else:
                #         sector_of_supplier = firm_list[graph[edge[0]][self]['object'].supplier_id].sector
                #         distance_of_supplier = self.distance_to_other(firm_list[graph[edge[0]][self]['object'].supplier_id])
                #         if self.odpoint == firm_list[graph[edge[0]][self]['object'].supplier_id].odpoint:
                #             same_place = 1
                #         else:
                #             same_place = 0
                #     if -1 in self.clients.keys():
                #         sales_to_hh = self.clients[-1]['share'] * self.production_target
                #     else:
                #         sales_to_hh = 0
                    
                #     logging.debug("Firm "+str(+self.pid)+" of sector "+str(self.sector)+" who sells "+str(sales_to_hh)+" to households"+\
                #     " has supplier "+str(graph[edge[0]][self]['object'].supplier_id)+" of sector "+str(sector_of_supplier)+" who is located at "+str(distance_of_supplier)+" increased price")
                return True
        return False
    
    

    def deliver_products(self, graph, transport_network=None, 
        rationing_mode="equal",
        monetary_unit_transport_cost="USD", monetary_unit_flow="mUSD",
        cost_repercussion_mode="type1", explicit_service_firm=True):
        # print("deliver_products", 0 in transport_network.nodes)
        # Do nothing if no orders
        if self.total_order == 0:
            logging.warning('Firm '+str(self.pid)+': no one ordered to me')
            return 0
        
        # Otherwise compute rationing factor
        self.rationing = self.product_stock / self.total_order
        if self.rationing > 1 + 1e-6:
            logging.debug('Firm '+str(self.pid)+': I have produced too much')
            self.rationing = 1
        
        elif self.rationing >= 1 - 1e-6:
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
            
            if explicit_service_firm:
                # If the client is B2C
                if edge[1].pid == -1:
                    self.deliver_without_infrastructure(graph[self][edge[1]]['object'])
                # If this is service flow, deliver without infrastructure
                elif self.sector_type in ['utility', 'transport', 'services']:
                    self.deliver_without_infrastructure(graph[self][edge[1]]['object'])
                # otherwise use infrastructure
                else:
                    self.send_shipment(
                        graph[self][edge[1]]['object'], 
                        transport_network,
                        monetary_unit_transport_cost,
                        monetary_unit_flow,
                        cost_repercussion_mode
                    )

            else:
                # If it's B2B and no service client, we send to the transport network, 
                # price will be adjusted according to transport conditions
                if (self.odpoint != -1) and (edge[1].odpoint != -1) and (edge[1].pid != -1):
                    self.send_shipment(
                        graph[self][edge[1]]['object'], 
                        transport_network,
                        monetary_unit_transport_cost,
                        monetary_unit_flow,
                        cost_repercussion_mode
                    )
                
                # If it's B2C, or B2B with service client, we send directly, 
                # and adjust price with input costs. There is still transport costs.
                elif (self.odpoint == -1) or (edge[1].odpoint == -1) or (edge[1].pid == -1):
                    self.deliver_without_infrastructure(graph[self][edge[1]]['object'])

                else:
                    logging.error('There should not be this other case.')


         
    def deliver_without_infrastructure(self, commercial_link):
        """ The firm deliver its products without using transportation infrastructure
        This case applies to service firm and to households 
        Note that we still account for transport cost, proportionnaly to the share of the clients
        Price can be higher than 1, if there are changes in price inputs
        """
        commercial_link.price = commercial_link.eq_price * (1 + self.delta_price_input)
        self.product_stock -= commercial_link.delivery
        self.finance['costs']['transport'] += self.clients[commercial_link.buyer_id]['share'] * self.eq_finance['costs']['transport']


                
    def send_shipment(self, commercial_link, transport_network,
        monetary_unit_transport_cost, monetary_unit_flow, cost_repercussion_mode):

        monetary_unit_factor = {
            "mUSD": 1e6,
            "kUSD": 1e3,
            "USD": 1
        }
        factor = monetary_unit_factor[monetary_unit_flow]
        # print("send_shipment", 0 in transport_network.nodes)
        """Only apply to B2B flows 
        """
        if len(commercial_link.route) == 0:
            raise ValueError("Firm "+str(self.pid)+" "+str(self.sector)+
                ": commercial link "+str(commercial_link.pid)+" (qty "+str(commercial_link.order)+")"
                " is not associated to any route, I cannot send any shipment to client "+
                str(commercial_link.buyer_id))

        if self.check_route_avaibility(commercial_link, transport_network, 'main') == 'available':
            # If the normal route is available, we can send the shipment as usual 
            # and pay the usual price
            commercial_link.price = commercial_link.eq_price * (1 + self.delta_price_input)
            commercial_link.current_route = 'main'
            transport_network.transport_shipment(commercial_link, monetary_unit_flow, self.usd_per_ton)
            self.product_stock -= commercial_link.delivery
            self.generalized_transport_cost += \
                commercial_link.route_time_cost \
                + transformUSDtoTons(commercial_link.delivery, monetary_unit_flow, self.usd_per_ton) \
                * commercial_link.route_cost_per_ton
            self.usd_transported += commercial_link.delivery
            self.tons_transported += transformUSDtoTons(commercial_link.delivery, monetary_unit_flow, self.usd_per_ton)
            self.tonkm_transported += transformUSDtoTons(commercial_link.delivery, monetary_unit_flow, self.usd_per_ton) \
                                    * commercial_link.route_length
            self.finance['costs']['transport'] += \
                self.clients[commercial_link.buyer_id]['share'] \
                * self.eq_finance['costs']['transport']
            return 0
            
        # If there is an alternative route already discovered, 
        # and if this alternative route is available, then we use it
        if (len(commercial_link.alternative_route)>0) \
            & (self.check_route_avaibility(commercial_link, transport_network, 'alternative') \
                == 'available'):
            route = commercial_link.alternative_route
        # Otherwise we need to discover a new one
        else:
            origin_node = self.odpoint
            destination_node = commercial_link.route[-1][0]
            route, selected_mode = self.choose_route(
                transport_network=transport_network.get_undisrupted_network(), 
                origin_node=origin_node,
                destination_node=destination_node, 
                possible_transport_modes=commercial_link.possible_transport_modes
            )
            # If we find a new route, we save it as the alternative one
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
                + transformUSDtoTons(commercial_link.delivery, monetary_unit_flow, self.usd_per_ton) \
                * commercial_link.alternative_route_cost_per_ton
            self.usd_transported += commercial_link.delivery
            self.tons_transported += transformUSDtoTons(commercial_link.delivery, monetary_unit_flow, self.usd_per_ton)
            self.tonkm_transported += transformUSDtoTons(commercial_link.delivery, monetary_unit_flow, self.usd_per_ton) \
                * commercial_link.alternative_route_length
        
            # We translate this real cost into transport cost
            if cost_repercussion_mode == "type1": #relative cost change with actual bill
                # Calculate relative increase in routing cost
                new_transport_bill = transformUSDtoTons(commercial_link.delivery, monetary_unit_flow, self.usd_per_ton) \
                    * commercial_link.alternative_route_cost_per_ton
                normal_transport_bill = transformUSDtoTons(commercial_link.delivery, monetary_unit_flow, self.usd_per_ton) \
                    * commercial_link.route_cost_per_ton
                relative_cost_change = max(new_transport_bill - normal_transport_bill, 0)/normal_transport_bill
                # Translate that into an increase in transport costs in the balance sheet
                self.finance['costs']['transport'] += \
                    self.eq_finance['costs']['transport'] \
                    * self.clients[commercial_link.buyer_id]['share'] \
                    * (1 + relative_cost_change)
                relative_price_change_transport = \
                    self.eq_finance['costs']['transport'] \
                    * relative_cost_change \
                    / ((1-self.target_margin) * self.eq_finance['sales'])
                # Calculate the relative price change, including any increase due to the prices of inputs
                total_relative_price_change = self.delta_price_input + relative_price_change_transport
                commercial_link.price = commercial_link.eq_price * (1 + total_relative_price_change)

            elif cost_repercussion_mode == "type2": #actual repercussion de la bill
                added_costUSD_per_ton = max(commercial_link.alternative_route_cost_per_ton - \
                    commercial_link.route_cost_per_ton, 0)
                added_costUSD_per_mUSD = added_costUSD_per_ton / (self.usd_per_ton/factor)
                added_costmUSD_per_mUSD = added_costUSD_per_mUSD/factor
                added_transport_bill = added_costmUSD_per_mUSD * commercial_link.delivery
                self.finance['costs']['transport'] += \
                    self.eq_finance['costs']['transport'] + added_transport_bill
                commercial_link.price = commercial_link.eq_price \
                    + self.delta_price_input \
                    + added_costmUSD_per_mUSD
                relative_price_change_transport = \
                    commercial_link.price / (commercial_link.eq_price + self.delta_price_input) - 1
                if (commercial_link.price is None) or (commercial_link.price is np.nan):
                    raise ValueError("Price should be a float, it is "+str(commercial_link.price))
                
                logging.debug('Firm '+str(self.pid)+
                    ": qty "+transformUSDtoTons(commercial_link.delivery, monetary_unit_flow, self.usd_per_ton)+'tons'+
                    " increase in route cost per ton "+ str((commercial_link.alternative_route_cost_per_ton-commercial_link.route_cost_per_ton)/commercial_link.route_cost_per_ton)+
                    " increased bill mUSD "+str(added_costmUSD_per_mUSD*commercial_link.delivery)
                )
                
            elif cost_repercussion_mode == "type3":
                relative_cost_change = \
                    (commercial_link.alternative_route_time_cost \
                    - commercial_link.route_time_cost)\
                    /commercial_link.route_time_cost
                self.finance['costs']['transport'] += \
                    self.eq_finance['costs']['transport'] \
                    * self.clients[commercial_link.buyer_id]['share'] \
                    * (1 + relative_cost_change)
                relative_price_change_transport = \
                    self.eq_finance['costs']['transport'] \
                    * relative_cost_change \
                    / ((1-self.target_margin) * self.eq_finance['sales'])
                
                total_relative_price_change = self.delta_price_input + relative_price_change_transport
                commercial_link.price = commercial_link.eq_price * (1 + total_relative_price_change)

            # With that, we deliver the shipment
            transport_network.transport_shipment(commercial_link, monetary_unit_flow, self.usd_per_ton)
            self.product_stock -= commercial_link.delivery
            # Print information
            logging.debug("Firm "+str(self.pid)+": found an alternative route to "+
                str(commercial_link.buyer_id)+", it is costlier by "+
                '{:.2f}'.format(100*relative_price_change_transport)+"%, price is "+
                '{:.4f}'.format(commercial_link.price)+" instead of "+
                '{:.4f}'.format(commercial_link.eq_price*(1+self.delta_price_input)))
        
        # If we do not find a route, then we do not deliver
        else:
            logging.debug('Firm '+str(self.pid)+": because of disruption, "+
                "there is no route between me and firm "+str(commercial_link.buyer_id))
            # We do not write how the input price would have changed
            commercial_link.price = commercial_link.eq_price
            commercial_link.current_route = 'none'
            # We do not pay the transporter, so we don't increment the transport cost
            # We set delivery to 0
            commercial_link.delivery = 0

  
    def add_congestion_malus2(self, graph, transport_network): 
        """Congestion cost are perceived costs, felt by firms, but they do not influence 
        prices paid to transporter, hence do not change price
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
                    transport_network.transport_shipment(graph[self][edge[1]]['object'], monetary_unit_flow, self.usd_per_ton)


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
        explicit_service_firm = True
        for edge in graph.in_edges(self): 
            if explicit_service_firm == True:
                if graph[edge[0]][self]['object'].product_type in ['services', 'utility', 'transport']:
                    self.receive_service_and_pay(graph[edge[0]][self]['object'])
                else:
                    self.receive_shipment_and_pay(graph[edge[0]][self]['object'], transport_network)
            else:
                if (edge[0].odpoint == -1) or (self.odpoint == -1): # if service, directly
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
        # quantity_intransit = commercial_link.delivery
        # Get delivery and update price
        quantity_delivered = 0
        price = 1
        if commercial_link.pid in transport_network.node[self.odpoint]['shipments'].keys():
            quantity_delivered += transport_network.node[self.odpoint]['shipments'][commercial_link.pid]['quantity']
            price = transport_network.node[self.odpoint]['shipments'][commercial_link.pid]['price']
            transport_network.remove_shipment(commercial_link)
        # Add to inventory
        self.inventory[commercial_link.product] += quantity_delivered
        # Log if quantity received differs from what it was supposed to be
        if abs(commercial_link.delivery - quantity_delivered) > 1e-6:
            logging.debug("Agent "+str(self.pid)+": quantity delivered by "+
                str(commercial_link.supplier_id)+" is "+str(quantity_delivered)+
                ". It was supposed to be "+str(commercial_link.delivery)+".")
        # Make payment
        commercial_link.payment = quantity_delivered * price
        
        
    def evaluate_profit(self, graph):
        self.finance['sales'] = sum([
            graph[self][edge[1]]['object'].payment 
            for edge in graph.out_edges(self)
        ])
        self.finance['costs']['input'] = sum([
            graph[edge[0]][self]['object'].payment 
            for edge in graph.in_edges(self)
        ]) 
        self.profit = self.finance['sales'] \
            - self.finance['costs']['input'] \
            - self.finance['costs']['other'] \
            - self.finance['costs']['transport']

        expected_gross_margin_no_transport = 1 - sum(list(self.input_mix.values()))
        realized_gross_margin_no_transport = \
            (self.finance['sales'] - self.finance['costs']['input']) \
            / self.finance['sales']
        realized_margin = self.profit / self.finance['sales']

        # Log discrepancies
        if abs(realized_gross_margin_no_transport - expected_gross_margin_no_transport) > 1e-6:
            logging.debug('Firm '+str(self.pid)+': realized gross margin without transport is '+
                '{:.3f}'.format(realized_gross_margin_no_transport)+" instead of "+
                '{:.3f}'.format(expected_gross_margin_no_transport))

        if abs(realized_margin - self.target_margin) > 1e-6:
            logging.debug('Firm '+str(self.pid)+': my margin differs from the target one: '+
                '{:.3f}'.format(realized_margin)+' instead of '+str(self.target_margin))


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