import json
import pandas as pd
import logging
import os


def loadDictionnaries(input_IO_filename):
    dictionaries = {}
    sectorId_to_sectorName = pd.read_excel(input_IO_filename, sheet_name='sector_name').set_index('sector_id')['merged_sector_name'].to_dict()
    dictionaries['sectorId_to_sectorName'] = sectorId_to_sectorName
    countryId_to_countryName = pd.read_excel(input_IO_filename, sheet_name='country_name').set_index('country_id')['country_name'].to_dict()
    dictionaries['countryId_to_countryName'] = countryId_to_countryName
    sectorId_to_sectorColor = pd.read_excel(input_IO_filename, sheet_name='sector_color').set_index('sector_id')['color'].to_dict()
    dictionaries['sectorId_to_sectorColor'] = sectorId_to_sectorColor
    sectorId_to_volumeCoef = pd.read_excel(input_IO_filename, sheet_name='sector_volume').set_index('sector_id')['volume_per_unit'].to_dict()
    dictionaries['sectorId_to_volumeCoef'] = sectorId_to_volumeCoef
    with open(os.path.join('tmp', 'dictionaries.json'), 'w') as fp:
        json.dump(dictionaries, fp)
    logging.info('Temporary file created: dictionaries.json')
    return dictionaries
    

def getDicOdpointidDistrict(sector_ODpoint_filename):
    odpoints_with_firms = pd.read_excel(sector_ODpoint_filename)
    odpoints_with_firms['nodenumber'] = odpoints_with_firms['nodenumber'].astype(int)
    return odpoints_with_firms.set_index('nodenumber')['loc_small_code'].to_dict()
    

def getDicLocationRegion(od_table):
    return od_table.set_index('od_point')['loc_small_code'].str[:2].to_dict()


def getDicIdLocation(firm_table):
    return firm_table.set_index('id')['location'].to_dict()