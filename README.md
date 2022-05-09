# DisruptSupplyChain Model

## Concepts

### Geographic structure

The model focuses on a country. The country is divided into districts. Those districts are Polygons, labelled using the country-relevant classification, for instance the string `XXXXXX` where X are digits.

The transport network is composed of edges and nodes. Edges are LineString, Nodes are Points. Edges are identified by an integer ID, so are nodes.

Each district is associated with one node in the transport network.


### Sector structure

We use one sector classification, such as ISIC Rev. 4, which depends on the data available on firms and on the input-output table available, and on their granularity. Each sector is identified by a trigram, such as `MTE` for Manufacture of Textiles.


### Objects

The main objects are the economic agents. There are three classes of agents:
- firms
- households
- countries

Firms, households, and countries are associated with nodes in the transport network. There is at most one firm per sector per district, one household per sector per district. Countries are associated to nodes which are located outside of the country.


## Inputs

Create an `input` directory in the root of the project. Create a subdirectory in the `input` directory, whose name should correspond to the `input_folder` variable given in the `parameters.py` file. Within this subdirectory, create 5 subdirectories:
- Transport
- Demand
- Disruption
- Supply
- Trade

Each section describes what should go in this directory.

Note that the filepath of each data files are defined in `parameter/filepaths_default.py`. These default filepaths can be overriden using the `parameter/filepaths.py` path.


### Transport

#### Transport network files

There should be two GeoJSON files per transport mode, one for nodes and one for edges. For instance, for roads:
- `road_nodes.geojson`
- `road_edges.geojson`
There is only one file for the multimodal layer, which describe the edges. There is no multimodal node.

The edge's geometry is LineString, the node's geometry is Point.

Nodes should contains at least the following attributes:
- id: int, one unique id per mode

Edges should contains at least the following attributes:
- id: int, one unique id per mode
- surface: paved or unpaved
- class: class category, for instance primary, secondary, etc.
- km: length in km
- multimodes (for multimodal edges only): define which type of multimodel link, for instance "roads-waterways" or "railways-maritime"
- capacity (optional): maximum handling capacity per time step

Based on these input files, the model creates one networkx.Graph object representing the transport network.


#### Transport Parameters

A yaml file `transport_parameters.yaml` with the following structure. It needs to be adjusted to the transport modes modelled.

	speeds: #km/hour
	  roads
	    paved: 31.4
	    unpaved: 15
	  railways: 23
	  waterways: 7
	  maritime: 35

	loading_time: #hours 
	  roads-waterways: 5
	  roads-maritime: 12
	  roads-railways: 12
	  railways-maritime: 24
	  waterways-maritime: 24

	variability: #as fraction of travel time
	  roads:
	    paved: 0.01 
	    unpaved: 0.075
	  railways: 0.02
	  waterways: 0.02
	  maritime: 0.005
	  multimodal:
	    roads-waterways: 0.1
	    roads-maritime: 0.1
	    roads-railways: 0.1
	    railways-maritime: 0.1
	    waterways-maritime: 0.1

	transport_cost_per_tonkm: #USD/(ton*km)
	  roads:
	    paved: 0.053
	    unpaved: 0.1
	  railways: 0.04
	  waterways: 0.029
	  maritime: 0.0017

	loading_cost_per_ton: #USD/ton
	  roads-waterways: 2.2
	  roads-maritime-shv: 2.2
	  roads-maritime-vnm: 2.2
	  roads-railways: 5
	  railways-maritime: 2.2
	  waterways-maritime: 2.2

	custom_cost: #USD/ton
	  roads: 27
	  waterways: 27
	  maritime: 27

	custom_time: #hours
	  roads: 1.5
	  waterways: 6
	  maritime: 2
	  multimodal: 2

	travel_cost_of_time: 0.49 #USD/hour

	variability_coef: 0.44 #USD/hour


#### Contraints on transport modes (optional)

An additional file `transport_modes.csv` can be used to prevent specific supply-chains flows from taking specific transport modes. 
To be further described


### Supply


#### Sector Table

A CSV file `sector_table.csv` providing, for each sector identified by its trigram:
- the type of sector among the following list: 'agriculture', 'manufacturing', 'utility', 'transport', 'services'
- the average monetary value, in USD, of a ton of good. This value can be computed from UN COMTRADE data, in which trade flows are both reported in tons and in USD.
- the percentage of the firms that export per sector. This value can be derived from country-specific data.
- the total yearly output, in USD (not kUSD, not mUSD, USD). This value can be derived from the input-output tables.
- the total yearly final demand, in USD (not kUSD, not mUSD, USD). This value can be derived from the input-output tables.

sector | type | usd_per_ton | share_exporting_firms | output | final_demand
--- | --- | --- | ---  | ---  | --- 
AGR | agriculture | 950 | 0.16 | 415641365| 246213152
... | ... | ... | ... | ... | ... 



#### Technical coefficients

Technical coefficients are derived from the symmetric industry-by-industry input-output table. The matrix of technical coefficients are sometimes directly available. If only the supply and use tables are available, additional steps are required, which are not described here. Make sure that the sectors in rows and columns are the one to be used in the model, i.e., that there are consistent with the firm-level data.

For the model, a CSV file `tech_coef_matrix.csv` should be provided with the following structure. 

|  | AGI | FOR | ...
--- | --- | --- | --- 
**AGI** | 0.11 | 0 | ...
**FOR** | 0.01 | 0.2 | ...
... | ... | ... | ...


#### Geospatial Economic Data

In this file are summarized the data on economic production for each district. For each district, there should be at least one value per sector which capture the size of the sector in this district. For instance, there could be the number of employees in the manufacturing of basic metals, the total sales of construction, and the value of production of agricultural products. Here, each district is associated to a Point, which can be the district capital city, or the Polygon's centroid.

A GeoJSON file `economic_data.geojson` with Points and the following attribute table:

district | nb_workers_MAN | nb_workers_ELE | crop_production
--- | --- | --- | --- 
0101 | 124 | 12 | 465120
... | ... | ... | ...


#### Sector firm cutoff

The model will create firms based on the geospatial economic data. To speed up computation, firms that would be too small are dropped. The CSV files `sector_firm_cutoffs.csv` define the cutoff to apply for each sector.

sector | supply_data | cutoff
--- | --- | --- | --- 
CRO | ag_prod | 3.50E+06
MIN | nb_workers_MIN | 200
MFO | nb_workers_MFO | 200
... | ... | ... | ...


#### Inventory Duration Target


A CSV file with the following strucutre.

input_sector | buying_sector | inventory_duration_target
--- | --- | --- 
TRD | AGR | 3.5
... | ... | ...




### Demand

#### Final Demand

A csv file with the yearly final demand per sector.

sector | final_demand
--- | --- 
AGR | 1230489103  
... | ... 


#### Population

A csv file with the population per district.

district | population
--- | --- 
01-01 | 123456  
... | ... 



### Trade

#### Country Transit Matrix

A csv file representing a double-entry table. Country codes are row and column headers.

|  | BDI | COD | ...
--- | --- | --- | --- 
**BDI** | 4563 | 4516 | ...
... | ... | ... | ...

#### Import Table

A csv file representing a double-entry table. Country codes are row headers. Sector codes are column headers.

|  | AGR | FOR | ...
--- | --- | --- | --- 
**BDI** | 132 | 0 | ...
... | ... | ... | ...


#### Export Table

A csv file representing a double-entry table. Country codes are row headers. Sector codes are column headers.

|  | AGR | FOR | ...
--- | --- | --- | --- 
**BDI** | 2 | 456 | ...
... | ... | ... | ...


#### Country Transit Points

A csv file with the following strucutre.

country | entry_point | weight
--- | --- | --- 
BDI | 7112 | 1
... | ... | ...

