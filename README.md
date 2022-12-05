# DisruptSC Model: Tanzania version

This is the first version of the DisruptSC model. It was applied to the United Republic of Tanzania; see [this report](https://openknowledge.worldbank.org/handle/10986/31909) and [this paper](https://www.nature.com/articles/s41893-020-00649-4).

## License

This model is published under the CC BY-NC License. The text of this license can be found [here](https://creativecommons.org/licenses/by-nc/4.0/legalcode). The explanation of the license can be found [here](https://creativecommons.org/licenses/by-nc/4.0/deed.en).

## Installation

It is based on python. The necessary modules are described in the file `requirements.txt`.
It can be launched by calling
  
	python code/mainBase.py <reuse_data>
  
where `reuse_data` is either `0` or `1`. If `0`, then the model creates a new transport network. If `1`, then the model reuse a previously-generated network, saved as pickle file in the `tmp` folder.

The following folder structure is necessary:
- code
- input
- output
- tmp

The model needs multiple files in the input folder, which are described below.


## Inputs

**Transport Parameters**: A yaml file.

**Road Network**: Two shapefiles, one for the road nodes, another for the road edges.
- `road_nodes`
- `road_edges`
If you set `new_roads = True`, the shapefile containing the extra road edges should be `road_edges_extra`.

**Origin--Destination Points (OD points)**: A csv file with the following strucutre.

odpoint | district | long | lat
--- | --- | --- | ---
7121 | 16-08 | 30.09349 | -4.56182
... | ... | ... | ...

**Technical coefficients**: A csv file with the following strucutre.

|  | AGI | FOR | ...
--- | --- | --- | --- 
**AGI** | 0.11 | 0 | ...
**FOR** | 0.01 | 0.2 | ...
... | ... | ... | ...

**District Sector Importance**: A csv file with the following strucutre.

district | sector | importance
--- | --- | --- 
01-01 | AGR | 0.0033
... | ... | ...

**Inventory Duration Target** (optional): A csv file with the following strucutre.

input_sector | buying_sector | inventory_duration_target
--- | --- | --- 
TRD | AGR | 3.5
... | ... | ...

**Sector Table**: A csv file providing, for each sector:
- the type of sector ('agriculture', 'manufacturing', 'utility', 'transport', 'services')
- the average monetary value, in USD, of a ton of good, 
- the percentage of the firms that export per sector.
- the total yearly output, is USD (not kUSD, not mUSD, USD)

sector | type | usd_per_ton | share_exporting_firms | output
--- | --- | --- | ---  | --- 
AGR | agriculture | 950 | 0.16 | 415641365
... | ... | ... | ... | ... 

**Final Demand**: A csv file with the yearly final demand per sector.

sector | final_demand
--- | --- 
AGR | 1230489103  
... | ... 

**Population**: A csv file with the population per district.

district | population
--- | --- 
01-01 | 123456  
... | ... 

**Transit Matrix**: A csv file representing a double-entry table. Country codes are row and column headers.

|  | BDI | COD | ...
--- | --- | --- | --- 
**BDI** | 4563 | 4516 | ...
... | ... | ... | ...

**Import Table**: A csv file representing a double-entry table. Country codes are row headers. Sector codes are column headers.

|  | AGR | FOR | ...
--- | --- | --- | --- 
**BDI** | 132 | 0 | ...
... | ... | ... | ...

**Export Table**: A csv file representing a double-entry table. Country codes are row headers. Sector codes are column headers.

|  | AGR | FOR | ...
--- | --- | --- | --- 
**BDI** | 2 | 456 | ...
... | ... | ... | ...

**Country Transit Points**: A csv file with the following strucutre.

country | entry_point | weight
--- | --- | --- 
BDI | 7112 | 1
... | ... | ...


## Output

For each simulation, a subfolder is automatically created in the output folder, and filled with the output files.
