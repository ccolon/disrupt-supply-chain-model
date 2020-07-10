# Data requirements for the DisruptSupplyChain model

## Transport

- Road network
    - The ideal form is one shapefile for the links, another for the nodes.
    - Road condition for each link (e.g., paved or unpaved)
    - Transport cost and use:
        - average transport price per km and per $-worth of goods (or per ton)
        - speed per road condition
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
