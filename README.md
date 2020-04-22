# DisruptSupplyChain Model

## Inputs

Create an `input` directory in the root of the project

Create a subdirectory in the `input` directory, whose name should correspond to the `input_folder` variable given in the `paramters.py` file

### Transport Parameters

A yaml file.

### Transport Network

#### Roads

Two shapefiles
- `road_nodes`
- `road_edges`

If you set `new_roads = True`, the shapefile containing the extra road edges should be `road_edges_extra`.


### Origin--Destination Points (OD points)

A csv file with the following strucutre.

odpoint | district | geometry | long | lat
--- | --- | --- | --- | --- | ---
7121 | 16-08 | POINT (30.09349 -4.56182) | 30.09349 | -4.56182
... | ... | ... | ... | ... 


### Technical coefficients

A csv file with the following strucutre.

 | AGI | FOR | ...
--- | --- | --- | ---
AGI | 0.11 | 0 | ... 
FOR | 0.01 | 0.2 | ... 
... | ... | ... | ... 


### District Sector Importance

A csv file with the following strucutre.

district | sector | importance
--- | --- | --- 
01-01 | AGR | 0.0033
... | ... | ...


### Inventory Duration Target (optional)

A csv file with the following strucutre.

input_sector | buying_sector | inventory_duration_target
--- | --- | --- 
TRD | AGR | 3.5
... | ... | ...

