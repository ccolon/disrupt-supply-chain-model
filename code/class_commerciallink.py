class CommercialLink(object):

    def __init__(self, pid=None, supplier_id=None, buyer_id=None, product=None, category=None, order=0, delivery=0, payment=0, route=None):
        # Parameter
        self.pid = pid
        self.product = product
        self.category = category
        self.route = route or [] # node_id path of the transport network, as
                                 # [(node1, ), (node1, node2), (node2, ), (node2, node3), (node3, )]
        self.route_length = 1
        self.route_time_cost = 0
        self.route_cost_per_ton = 0
        self.supplier_id = supplier_id
        self.buyer_id = buyer_id
        self.eq_price = 1
        
        # Variable
        self.current_route = 'main'
        self.order = order # flows upstream
        self.delivery = delivery # flows downstream
        self.payment = payment # flows upstream
        self.alternative_route = []
        self.alternative_route_length = 1
        self.alternative_route_time_cost = 0
        self.alternative_route_cost_per_ton = 0
        self.price = 1

    def print_info(self):
        print("\nCommercial Link from "+str(self.supplier_id)+" to "+str(self.buyer_id)+":")
        print("route:", self.route)
        print("alternative route:", self.alternative_route)
        print("product:", self.product)
        print("order:", self.order)
        print("delivery:", self.delivery)
        print("payment:", self.payment)
        
        
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
        
        

    
