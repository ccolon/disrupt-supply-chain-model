from functions import transformUSDtoTons

def agent_decide_initial_routes(agent, graph, transport_network, transport_modes,
        account_capacity, monetary_unit_flow):
    for edge in graph.out_edges(agent):
        if edge[1].pid == -1: # we do not create route for households
            continue
        elif edge[1].odpoint == -1: # we do not create route for service firms if explicit_service_firms = False
            continue
        else:
            # Get the id of the orign and destination node
            origin_node = agent.odpoint
            destination_node = edge[1].odpoint
            # Define the type of transport mode to use by looking in the transport_mode table
            if agent.agent_type == 'firm':
                cond_from = (transport_modes['from'] == "domestic")
            elif agent.agent_type == 'country':
                cond_from = (transport_modes['from'] == self.pid)
            else:
                raise ValueError("'agent' must be a Firm or a Country")
            if agent.agent_type == 'firm': #see what is the other end
                cond_to = (transport_modes['to'] == "domestic")
            elif agent.agent_type == 'country':
                cond_to = (transport_modes['to'] == edge[1].pid)
            else:
                raise ValueError("'edge[1]' must be a Firm or a Country")
                # we have not implemented the "sector" condition
            transport_mode = transport_modes.loc[cond_from & cond_to, "transport_mode"].iloc[0]
            graph[agent][edge[1]]['object'].transport_mode = transport_mode
            # Choose the route and the corresponding mode
            route, selected_mode = agent.choose_route(
                transport_network=transport_network, 
                origin_node=origin_node, 
                destination_node=destination_node, 
                possible_transport_modes=transport_mode
            )
            # Store it into commercial link object
            graph[agent][edge[1]]['object'].storeRouteInformation(
                route=route,
                transport_mode=selected_mode,
                main_or_alternative="main",
                transport_network=transport_network
            )
            # Update the "current load" on the transport network
            # if current_load exceed burden, then add burden to the weight
            if account_capacity:
                new_load_in_usd = graph[agent][edge[1]]['object'].order
                new_load_in_tons = transformUSDtoTons(new_load_in_usd, monetary_unit_flow, agent.usd_per_ton)
                transport_network.update_load_on_route(route, new_load_in_tons)