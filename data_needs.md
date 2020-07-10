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


#Supply & demand
•   Input-output table
•   Business census / firm registry, and, for each firm:
o   its sector, coded in a standard classification (e.g., ISIC),
o   its location (gps coordinates or administrative unit),
o   its size (nb of employees or turnover),
•   
•   Geographically-explicit population data (gridded and/or using administrative boundaries)
•   International trade data and input output table (we used GTAP, Cambodia is included)

•   If we want to include all sectors, then we need firm-level data, i.e., a list of the registered companies of the country, and, for each of them: 
o   its sector, coded in a standard classification (e.g., ISIC),
o   its location (gps coordinates or administrative unit),
o   its size (nb of employees or turnover),
o   ideally, the name of its main suppliers and customers, but that is often not available.
o   If there is any significant industry for the country (e.g., mining), any additional information on the volume and location of production can be integrated to increase accuracy.
•   We need some indication of the inventories held by firms of the various sector
o   For Tanzania, we asked specific questions on a firm survey that was carried out last year in Tanzania. About 800 firms were surveyed.
o   We can use some approximate values to begin with.
•   Other data that you probably already have are:

