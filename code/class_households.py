import random
import pandas as pd
import logging

from class_commerciallink import CommercialLink


class Households(object):

    def __init__(self, final_demand=1, purchase_plan=None, final_demand_per_sector=None):
        # Intrinsic parameters
        self.pid = -1
        self.odpoint = -1
        # Parameters depending on data and network
        self.final_demand = final_demand
        self.final_demand_per_sector = final_demand_per_sector or {}
        self.purchase_plan = purchase_plan or {}
        self.retailers = {}
        self.budget = 1
        # Variables reset and updated at each time step
        self.consumption = {}
        self.tot_consumption = 0
        self.consumption_per_sector = {}
        self.spending = {}
        self.tot_spending = 0
        self.spending_per_sector = {}
        # Cumulated variables reset at beginning and updated at each time step
        self.consumption_loss = 0
        self.extra_spending = 0
        
    
    def reset_variables(self):
        self.budget = 1
        self.consumption = {}
        self.tot_consumption = 0
        self.spending = {}
        self.tot_spending = 0
        self.extra_spending = 0
        self.consumption_loss = 0
        self.consumption_loss_per_sector = {key: 0 for key, val in self.final_demand_per_sector.items()}
    
    
    def select_suppliers(self, graph, firm_list, mode='inputed'):
        if mode=='equal':
            for firm in firm_list:
                graph.add_edge(firm, self,
                               object=CommercialLink(
                                   pid=str(firm.pid)+'to'+str(self.pid),
                                   product=firm.sector,
                                   category="domestic_B2C",
                                   supplier_id=firm.pid,
                                   buyer_id=self.pid)
                              )
                self.purchase_plan = {firm.pid: self.budget/len(firm_list) for firm in firm_list}
                
        elif mode=='selected_retailers':
            firm_id_each_sector = pd.DataFrame({
                'firm': [firm.pid for firm in firm_list],
                'sector': [firm.sector for firm in firm_list]})
            dic_sector_to_nbfirms = firm_id_each_sector.groupby('sector')['firm'].count().to_dict()
            sectors = firm_id_each_sector['sector'].unique()
            consumption_each_sector_quantity = {sector: self.budget/len(sectors) for sector in sectors}

            for sector in sectors:
                nb_retailers = random.randint(1,dic_sector_to_nbfirms[sector])
                retailers = random.sample(firm_id_each_sector.loc[firm_id_each_sector['sector']==sector, 'firm'].tolist(), nb_retailers)
                weight = 1 / nb_retailers
                for retailer_id in retailers:
                    # For each supplier, create an edge in the economic network
                    graph.add_edge(firm_list[retailer_id], self,
                               object=CommercialLink(
                                   pid=str(retailer_id)+'to'+str(self.pid),
                                   product=sector,
                                   supplier_id=retailer_id,
                                   buyer_id=self.pid)
                              )
                    # Associate a weight
                    graph[firm_list[retailer_id]][self]['weight'] = weight
                    # Households save the name of the retailer, its sector, its weight, and adds it to its purchase plan
                    self.retailers[retailer_id] = {'sector':self.pid, 'weight':weight}
                    self.purchase_plan[retailer_id] = consumption_each_sector_quantity[sector] * weight
                    # The retailer saves the fact that it supplies to households. The share of sales cannot be calculated now.
                    firm_list[retailer_id].clients[self.pid] = {'sector':self.pid, 'share':0}
            
            
        elif mode == 'inputed':
            if len(self.purchase_plan)==0:
                raise KeyError('Households: mode==inputed but no purchase plan')
            elif len(self.final_demand_per_sector)==0:
                raise KeyError('Households: mode==inputed but no final_demand_per_sector')
            else:
                for retailer_id, purchase_this_retailer in self.purchase_plan.items():
                    if purchase_this_retailer > 0:
                        sector = firm_list[retailer_id].sector
                        graph.add_edge(firm_list[retailer_id], self,
                               object=CommercialLink(
                                   pid=str(retailer_id)+'to'+str(self.pid),
                                   product=sector,
                                   supplier_id=retailer_id,
                                   buyer_id=self.pid)
                              )
                        weight = purchase_this_retailer / self.final_demand_per_sector[sector]
                        graph[firm_list[retailer_id]][self]['weight'] = weight
                        self.retailers[retailer_id] = {'sector':self.pid, 'weight':weight}
                        firm_list[retailer_id].clients[self.pid] = {'sector':self.pid, 'share':0}
                
        else:
            raise ValueError("Households: Wrong mode chosen")


    def send_purchase_orders(self, graph):
        for edge in graph.in_edges(self):
            try:
                quantity_to_buy = self.purchase_plan[edge[0].pid]
            except KeyError:
                print("Households: No purchase plan for supplier", edge[0].pid)
                quantity_to_buy = 0
            graph[edge[0]][self]['object'].order = quantity_to_buy


    def receive_products_and_pay(self, graph):
        # Re-initialize values that reset at each time step 
        self.consumption = {}
        self.tot_consumption = 0 #quantity
        self.spending = {}
        self.tot_spending = 0 #money

        for edge in graph.in_edges(self):
            # For each retailer, get the products
            quantity_delivered = graph[edge[0]][self]['object'].delivery
            self.consumption[edge[0].pid] = quantity_delivered
            self.tot_consumption += quantity_delivered
            # Get the price and pay
            price = graph[edge[0]][self]['object'].price
            graph[edge[0]][self]['object'].payment = quantity_delivered * price
            self.spending[edge[0].pid] = quantity_delivered * price
            self.tot_spending += quantity_delivered * price
            # Increment extra spending
            self.extra_spending += quantity_delivered * (price - graph[edge[0]][self]['object'].eq_price)
            # Increment consumption loss
            consum_loss = (self.purchase_plan[edge[0].pid] - quantity_delivered) * (graph[edge[0]][self]['object'].eq_price)
            if consum_loss >= 1e-6:
                if True:
                    logging.debug("Household: Firm "+str(edge[0].pid)+" supposed to deliver "+str(self.purchase_plan[edge[0].pid])+ " but delivered "+str(quantity_delivered))
                self.consumption_loss += consum_loss

        

    def print_info(self):
        print("\nHouseholds:")
        print("final_demand:", self.final_demand)
        print("purchase_plan:", self.purchase_plan)
        print("consumption:", self.consumption)

        
