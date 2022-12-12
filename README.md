# DisruptSC Model: Cambodia 2022 version

This is the second version of the DisruptSC model. It was applied to Cambodia. Note that input files are not provided with this release.

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
- parameter
- tmp

The model needs multiple files in the input folder, which are described below.

## Parameter

Four files:
- `filepaths_default.py` specifies the filepath of the input folder. They can be overriden using `filepaths.py`.
- `parameters_default.py` defines all parameter of the model. They can be overriden using `parameters.py`

## Managing different set of input files

In this version, the user can create different sets of inputs files by creating multiple subdirectory within the `input` directory. For instance,
- input
  - cambodia
  - tanzania_2015
  - tanzania_2020

By setting the parameter `input_folder` to, say, `tanzania_2020`, the model will read the input files from the corresponding directory, following the filepaths.

Each input directory should have the following subdirectories
- Demand
- Disruption (optional)
- Supply
- Trade
- Transport

E.g.,
- input
  - cambodia
    - Demand
      - *input files* 
    - Supply
      - *input files* 
    - Trade
      - *input files* 
    - Transport
      - *input files* 

For each simulation, a subfolder is automatically created in the `output` folder, in a subdirectory with the same name as the `input_folder`, e.g., `tanzania_2020`, and filled with the output files. E.g.,
- output
  - tanzania_2020
    - 20201111_101745
      - *output files*


## Input files

### Transport

For each transport mode, two GeoJSON files:
- `<transport_mode>_nodes.geojson`
- `<transport_mode>_edges.geojson`

Where `transport_mode` is one of `roads`, `railways`, `waterways`, `maritime`. Airways are not yet supported. If there are more than two transport modes, then a `multimodal_edges.geojson` should also be present.

Required attributes:
- `<transport_mode>_nodes`: `id` (integer, unique only within the `<transport_mode>_nodes` dataset)
- `<transport_mode>_edges`: `end1` and `end2` (integer, id of the `<transport_mode>_nodes` located at the end of the edges), `km` (float, length of edge), `capacity` (tons per year it can transport, leave blank if unknown), `special` (`custom` if the edge involve custom procedure, refered to in the `transport_parameters.yaml` file, blank otherwise)
- `roads_nodes.geojson`: `surface` (`paved` or `unpaved`)
- `multimodal_edges.geojson`: `multimodes` (one of: `roads-railways`, `roads-waterways`, `roads-maritime`, `railways-waterways`, `railways-maritime`, `waterways-maritime`)

Two extra files are needed in this folder:
- `transport_parameters.yaml`: all cost-related transport parameters
- `transport_modes.csv`: defines the part of the transport network that is available for different types of flow. See example below:

from | to | sector | transport_mode
--- | --- | --- | ---
domestic | domestic | all | roads
domestic | VNM | all | roads
... | ... | ... | ...
domestic | AFR | all | intl_multimodes
... | ... | ... | ...

Here it says that:
- domestic supply-chain flows are transported by roads,
- export flows to Vietnam are transported by roads,
- export flows to Africa are transported by different modes described as intl_multimodes, which are defined in the code.


### Demand

The data are spatially structured by administrative units, called `adminunit` in the code. 

**adminunit_demographic_data**: A geojson file. Each feature is a Point, which defines the location of the agents. The model takes care of linking that point to the closest road node of the transport network. Attributes are: `adminunit_id`, `population`.


### Supply

**adminunit_economic_data**: A geojson file. Each feature is a Point, which defines the location of the agents. It should be the same as in `adminunit_demographic_data`. There should be at least one attribute per sector, with a measure of the size of the sector in this adminunit. E.g., "nb_workers_CON" (nb of workers in the construction sector), "nb_workers_MTE" (nb of workers in the textile manufacturing sector), "ag_prod" (value of production for the agriculture sector). 

**sector_table**: A csv file providing, for each sector:
- it's ID, usually a trigram (e.g., AGR, MIN, MTE, CON)
- the type of sector ('agriculture', 'manufacturing', 'utility', 'transport', 'services')
- the average monetary value, in USD, of a ton of good, 
- the percentage of the firms that export per sector.
- the total yearly output, in USD
- the total yearly final demand, in USD

sector | type | usd_per_ton | share_exporting_firms | output | output
--- | --- | --- | ---  | --- | --- 
AGR | agriculture | 950 | 0.16 | 415641365 | 379412389
... | ... | ... | ... | ... | ... 

**sector_cutoffs**: A csv file providing, for each sector:
- it's ID, usually a trigram (e.g., AGR, MIN, MTE, CON)
- the attribute of "adminunit_economic_data" that is used to assess the sector's size in the adminunits
- a cutoff value: if the size of the sector in one adminunit is lower than this cutoff, then no firm is created in this adminunit for this sector.

sector | supply_data | cutoff
--- | --- | ---
AGR | ag_prod | 3.50E+06
... | ... | ...

**tech_coef**: A csv file with the input--output technical coefficient.

|  | AGI | FOR | ...
--- | --- | --- | --- 
**AGI** | 0.11 | 0 | ...
**FOR** | 0.01 | 0.2 | ...
... | ... | ... | ...

**inventory_duration_targets** (optional): A csv file with the following strucutre.

input_sector | buying_sector | inventory_duration_target
--- | --- | --- 
TRD | AGR | 3.5
... | ... | ...


### Trade

**imports**: A csv file representing a double-entry table. Country codes are row headers. Sector codes are column headers.

|  | AGR | FOR | ...
--- | --- | --- | --- 
**BDI** | 132 | 0 | ...
... | ... | ... | ...

**exports**: A csv file representing a double-entry table. Country codes are row headers. Sector codes are column headers.

|  | AGR | FOR | ...
--- | --- | --- | --- 
**BDI** | 2 | 456 | ...
... | ... | ... | ...

**transit_matrix**: A csv file representing a double-entry table. Country codes are row and column headers.

|  | BDI | COD | ...
--- | --- | --- | --- 
**BDI** | 4563 | 4516 | ...
... | ... | ... | ...

### Disruption (optional)

Here can be placed files that defines a reduced set of transport nodes or edges on which to run the criticality analysis. It can be, for instance, a file called "top_edges.csv" which contains the list of edges to test.
