from collections import OrderedDict
from io import BytesIO
from io import StringIO

import pandas as pd
import math
import requests as req
import bs4
from zipfile import ZipFile
import os
import csv
import shapefile
from geopy import distance

from naps import Naps
from naps import Input

def point_distance(point1, point2):
    return distance.distance(point1, point2).m

class Stations:
    df = pd.read_csv('Station Inventory EN.csv')

    @classmethod
    def print(self):
        print(self.df)

    @classmethod
    def get_station(self, row):
        return self.df.iloc[row]

    @classmethod
    def get_station_with_name(self, name):
        ids = self.df.index[self.df['Name'] == name].tolist()
        if not ids:
            print(f'Did not find {name}\n')
        return Stations.get_station(ids[0])

class Station:
    df = pd.DataFrame()
    
    def __init__(self, name):
        self.info = Stations.get_station_with_name(name)

    def get_coords():
        return [self.info['Latitude (Decimal Degrees)'], self.info['Longitude (Decimal Degrees)']]

    def print(self):
        print(self.info)
        
    def climate_data(self, day=0, month=0, year=0):
        local_path = f"./data_cache/climate_data/{self.info['Station ID']}_{year}_{month}.csv"

        if not os.path.exists(local_path):
            # this link returns the whole month regardless of day specified
            url = f"https://climate.weather.gc.ca/climate_data/bulk_data_e.html?format=csv"\
                  f"&stationID={self.info['Station ID']}"\
                  f"&Year={year}"\
                  f"&Month={month}"\
                  f"&Day=1"\
                  f"&timeframe=1"    

            print(url)
            response = req.get(url)
            assert response.status_code == 200

            data = BytesIO(response.content)
            df = pd.read_csv(data)
            df.to_csv(local_path)
        else:
            df = pd.read_csv(local_path)

        self.df = pd.concat([self.df, df], ignore_index=True)
        return df

class NFDB:
    # https://cwfis.cfs.nrcan.gc.ca/datamart
    shp_file = None # where to store the shape file

    def load_shp_file(self):
        if self.shp_file is None:
            self.shp_file = shapefile.Reader("NFDB_point.zip")
    
    def data(self, year):
        local_csv_path = f'./data_cache/fires/{year}.csv'
        
        if not os.path.exists(local_csv_path):
            self.load_shp_file()
            
            records = []
            shps = []
            index = 0
            count = len(self.shp_file.shapes())
    
            while index < count:
                r = self.shp_file.record(index)
                if r.YEAR == year:
                    records.append(r)
                    shps.append(self.shp_file.shape(index))

                index += 1
    
            fields = [x[0] for x in self.shp_file.fields][1:]
            df = pd.DataFrame(columns=fields, data=records)
            df = df.assign(coords=shps)
    
            df.to_csv(local_csv_path)
        else:
            df = pd.read_csv(local_csv_path)
        
        return df

import pickle

def create_data_cache():
    paths = ["./data_cache", "./data_cache/naps", "./data_cache/climate_data/"]
    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path)

def main():    
    create_data_cache()
    
    #station = Station("CALGARY INTL A")

    #training_data = pd.DataFrame(columns=['Current', 'Target'])
    #naps_2021 = Naps.data(year=2021)
    #for i in range(10):
        #day = naps_2021.iloc[i]
        #new_train = { "Current": Naps.PM25(day, 12), "Target": Naps.PM25(day, 17) }
        #training_data.loc[i] = new_train
    #print(training_data)
    naps = Naps()
    pm25_data, coords_data = naps.get_year(2021)
    input = Input(pm25_data, coords_data)
    f = open("./data_cache/test.data", "wb")
    pickle.dump(input, f)
    f.close()

    
if __name__ == '__main__':
    main()
