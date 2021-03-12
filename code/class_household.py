from functions import determine_nb_suppliers, select_supplier_from_list,\
                agent_receive_products_and_pay, calculate_distance_between_agents

import random
import pandas as pd
import logging

from class_commerciallink import CommercialLink


class Household(object):

    def __init__(self, pid, odpoint, long, lat, sector_consumption):
        # Intrinsic parameters
        self.agent_type = "household"
        self.pid = pid
        self.odpoint = odpoint
        self.long = long
        self.lat = lat
        # Parameters depending on data
        self.sector_consumption = sector_consumption
        # Parameters depending on network
        self.purchase_plan = {}
        self.retailers = {}
        # Variables reset and updated at each time step
        self.consumption_per_retailer = {}
        self.tot_consumption = 0
        self.consumption_per_sector = {}
        self.spending_per_retailer = {}
        self.tot_spending = 0
        self.spending_per_sector = {}
        # Cumulated variables reset at beginning and updated at each time step
        self.consumption_loss = 0
        self.extra_spending = 0
        
    
    def reset_variables(self):
        self.consumption_per_retailer = {}
        self.tot_consumption = 0
        self.spending_per_retailer = {}
        self.tot_spending = 0
        self.extra_spending = 0
        self.consumption_loss = 0
        self.extra_spending_per_sector = {}
        self.consumption_loss_per_sector = {}
    
    
    def initialize_var_on_purchase_plan(self):
        if len(self.purchase_plan) == 0:
            logging.warn("Households initialize variables based on purchase plan, "
                +"but it is empty.")

        self.consumption_per_retailer = self.purchase_plan
        self.tot_consumption = sum(list(self.purchase_plan.values()))
        self.consumption_loss_per_sector = {sector: 0 for sector in self.purchase_plan.keys()}
        self.spending_per_retailer = self.consumption_per_retailer
        self.tot_spending = self.tot_consumption
        self.extra_spending_per_sector = {sector: 0 for sector in self.purchase_plan.keys()}


    def select_suppliers(self, graph, firm_list, firm_table, nb_retailers, weight_localization):
        for sector, amount in self.sector_consumption.items():
            # Establish list of potential firms
            potential_firms = firm_table.loc[firm_table['sector']==sector, 'id'].tolist()
            if len(potential_firms) == 0:
                raise ValueError('No firm to select')

            # Determine number of suppliers to choose from
            nb_suppliers_to_choose = determine_nb_suppliers(
                nb_suppliers_per_input=nb_retailers, 
                max_nb_of_suppliers=len(potential_firms)
            )

            # Select based on size and distance
            retailers, retailer_weights = select_supplier_from_list(
                self, firm_list, 
                nb_suppliers_to_choose, potential_firms, 
                distance=True, importance=False, 
                weight_localization=weight_localization, force_same_odpoint=True
            )

            # For each of them, create commercial link
            for retailer_id in retailers:
                # For each retailer, create an edge in the economic network
                graph.add_edge(firm_list[retailer_id], self,
                            object=CommercialLink(
                                pid=str(retailer_id)+'->'+str(self.pid),
                                product=sector,
                                product_type=firm_list[retailer_id].sector_type,
                                category="domestic_B2C",
                                supplier_id=retailer_id,
                                buyer_id=self.pid)
                            )
                # Associate a weight in the commercial link, the household's purchase plan & retailer list, in the retailer's client list
                weight = retailer_weights.pop()
                graph[firm_list[retailer_id]][self]['weight'] = weight
                self.purchase_plan[retailer_id] = weight * self.sector_consumption[sector]
                self.retailers[retailer_id] = {'sector': sector, 'weight' : weight}
                distance = calculate_distance_between_agents(self, firm_list[retailer_id])
                firm_list[retailer_id].clients[self.pid] = {
                    'sector': "households", 'share':0, 'transport_share':0, "distance": distance
                } #The share of sales cannot be calculated now.


    def send_purchase_orders(self, graph):
        for edge in graph.in_edges(self):
            try:
                quantity_to_buy = self.purchase_plan[edge[0].pid]
            except KeyError:
                print("Households: No purchase plan for supplier", edge[0].pid)
                quantity_to_buy = 0
            graph[edge[0]][self]['object'].order = quantity_to_buy


    def receive_products_and_pay(self, graph, transport_network):
        agent_receive_products_and_pay(self, graph, transport_network)
        # Re-initialize values that reset at each time step 
        #self.reset_variables()
        # self.consumption = {}
        # self.tot_consumption = 0 #quantity
        # self.spending = {}
        # self.tot_spending = 0 #money

        '''
        for edge in graph.in_edges(self):
            # For each retailer, get the products
            quantity_delivered = graph[edge[0]][self]['object'].delivery
            self.consumption[edge[0].pid] = quantity_delivered
            self.tot_consumption += quantity_delivered
            # Update the price and pay
            price = graph[edge[0]][self]['object'].price
            graph[edge[0]][self]['object'].payment = quantity_delivered * price
            # Measure spending
            self.spending[edge[0].pid] = quantity_delivered * price
            self.tot_spending += quantity_delivered * price
            # Increment extra spending and consumption loss
            self.extra_spending += quantity_delivered * \
                                (price - graph[edge[0]][self]['object'].eq_price)
            # Increment consumption loss
            consum_loss = (self.purchase_plan[edge[0].pid] - quantity_delivered) * \
                        (graph[edge[0]][self]['object'].eq_price)
            self.consumption_loss += consum_loss
            # Log if there is undelivered goods
            if consum_loss >= 1e-6:
                logging.debug("Household: Firm "+str(edge[0].pid)+" supposed to deliver "+
                    str(self.purchase_plan[edge[0].pid])+ " but delivered "+
                    str(quantity_delivered))
        '''
        