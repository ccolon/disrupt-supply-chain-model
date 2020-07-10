# Data requirements for the DisruptSupplyChain model

## Transport

- Road network
    - The ideal structure is one shapefile for the links, another for the nodes.
    - For each link:
        - road class (primary, secondary, etc.)
        - road condition (e.g., paved or unpaved)
        - nb of lanes (optional)
        - whether it is a bridge or not (optional)
    - Transport cost and capacity:
        - average transport price per km and per ton or $-worth of goods
        - speed per type of road
        - capacity per type of road
        - some measure of road unreliability (variation of travel time)
    - If available, a selection of the "main" nodes, i.e., those with most traffic

- If applicable, same information for other transport network (e.g., railways, waterways)

- International transport nodes (ports, airports, border crossings)


## Firm data and inventories

- Business census / firm registry, and, for each firm:
    - its sector, coded in a standard classification (e.g., ISIC),
    - its location (gps coordinates or administrative unit),
    - its size (nb of employees or turnover)
    If not available, then total production per sector and per administrative unit (the more disaggregated, the better supply chain flows can be mapped onto the transport network)
- Amount of inventories per sector (e.g., from a survey)
- Agricultural / fishing / logging production per km2 or administrative units. These economic activities are often not registered in any firm census, in particular fishing, logging, agriculture.


## Trade & national account

- Input-output table (e.g., GTAP, produced by the country)
- International trade date (e.g., GTAP, UNComtrade)
