class CommercialLink(object):

    def __init__(self, pid=None, supplier_id=None, buyer_id=None, product=None, 
        product_type=None, category=None, order=0, delivery=0, payment=0, 
        route=None):
        # Parameter
        self.pid = pid
        self.product = product #sector of producing firm
        self.product_type = product_type #service, manufacturing, etc. (=sector_type)
        self.category = category #import, export, domestic_B2B, transit
        self.route = route or [] # node_id path of the transport network, as
                                 # [(node1, ), (node1, node2), (node2, ), (node2, node3), (node3, )]
        self.route_length = 1
        self.route_time_cost = 0
        self.route_cost_per_ton = 0
        self.route_mode = "road"
        self.supplier_id = supplier_id
        self.buyer_id = buyer_id
        self.eq_price = 1
        self.possible_transport_modes = "roads"
        
        # Variable
        self.current_route = 'main'
        self.order = order # flows upstream
        self.delivery = delivery # flows downstream. What is supposed to be delivered (if no transport pb)
        self.realized_delivery = delivery
        self.payment = payment # flows upstream
        self.alternative_route = []
        self.alternative_route_length = 1
        self.alternative_route_time_cost = 0
        self.alternative_route_cost_per_ton = 0
        self.alternative_route_mode = None
        self.price = 1

    def print_info(self):
        # print("\nCommercial Link from "+str(self.supplier_id)+" to "+str(self.buyer_id)+":")
        # print("route:", self.route)
        # print("alternative route:", self.alternative_route)
        # print("product:", self.product)
        # print("order:", self.order)
        # print("delivery:", self.delivery)
        # print("payment:", self.payment)
        attribute_to_print = [a for a in dir(self) if not a.startswith('__') and not callable(getattr(self, a))]
        for attribute in attribute_to_print:
            print(attribute+": "+str(getattr(self, attribute)))
            print("\n")

        
        
    def reset_variables(self):
        # Variable
        self.current_route = 'main'
        self.order = 0 # flows upstream
        self.delivery = 0 # flows downstream
        self.payment = 0 # flows upstream
        self.alternative_route = []
        self.alternative_route_time_cost = 0
        self.alternative_route_cost_per_ton = 0
        self.price = 1
        
        
    def storeRouteInformation(self, route, transport_mode, 
        main_or_alternative, transport_network):

        distance, route_time_cost, cost_per_ton = transport_network.giveRouteCaracteristics(route)

        if main_or_alternative == "main":
            self.route = route
            self.route_mode = transport_mode
            self.route_length = distance
            self.route_time_cost = route_time_cost
            self.route_cost_per_ton = cost_per_ton
    
        elif main_or_alternative == "alternative":
            self.alternative_route = route
            self.alternative_route_mode = transport_mode
            self.alternative_route_length = distance
            self.alternative_route_time_cost = route_time_cost
            self.alternative_route_cost_per_ton = cost_per_ton

            switching_cost = 0.05
            if self.alternative_route_mode != self.route_mode:
                self.alternative_route_cost_per_ton * (1 + switching_cost)

        else:
            raise ValueError("'main_or_alternative' is not in ['main', 'alternative']")
