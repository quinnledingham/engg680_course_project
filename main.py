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
import datetime
import numpy as np
from concurrent.futures import ProcessPoolExecutor

from scipy.spatial import KDTree
from scipy.spatial import distance_matrix

from naps import Naps
from gpt import Input_Data

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
        
        # Store fire data with coordinates and report dates
        self.fire_data = df[['LATITUDE', 'LONGITUDE', 'REP_DATE', 'OUT_DATE']]
        self.fire_data['REP_DATE'] = pd.to_datetime(self.fire_data['REP_DATE'])
        self.fire_data['OUT_DATE'] = pd.to_datetime(self.fire_data['OUT_DATE'])
        
        # Calculate a general max_distance (95th percentile of distances)
        fire_coords = self.fire_data[['LATITUDE', 'LONGITUDE']].to_numpy()
        dist_matrix = distance_matrix(fire_coords, fire_coords)
        self.max_distance = np.percentile(dist_matrix, 100)

        self.active_fire_dates = self.fire_data[['REP_DATE', 'OUT_DATE']]

        return df
    
    def get_years(self, years):
        for year in years:
            df = self.data(year)

        return df

    def find_closest_active_fire(self, point, timestamp):
         # Filter fires active on or before the timestamp
        active_mask = (self.active_fire_dates['REP_DATE'] <= timestamp)

        # Handle the case where OUT_DATE is missing: fires without OUT_DATE are considered inactive after 3 weeks
        # We filter for fires where REP_DATE + 3 weeks >= timestamp
        active_mask = active_mask & ((self.active_fire_dates['OUT_DATE'].notna() & (self.active_fire_dates['OUT_DATE'] >= timestamp)) | (self.active_fire_dates['OUT_DATE'].isna() & (self.active_fire_dates['REP_DATE'] + datetime.timedelta(weeks=3) >= timestamp)))

        active_fires = self.fire_data[active_mask]

        if active_fires.empty:
            # No active fires; return a default large distance indicating no nearby fire
            return 0  # or another indicator for no active fire

        # Create a KDTree for the active fires
        active_fire_coords = active_fires[['LATITUDE', 'LONGITUDE']].to_numpy()
        active_fire_tree = KDTree(active_fire_coords)

        # Query the closest active fire
        dist, _ = active_fire_tree.query(point)

        # Normalize distance to a 0-9 scale
        fire_proximity = max(0, min(9, 9 * (1 - np.log(dist + 1) / np.log(self.max_distance + 1)) ** 2))
        return fire_proximity
    
import pickle

def create_data_cache():
    paths = ["./data_cache", "./data_cache/naps", "./data_cache/climate_data/", "./data_cache/fires"]
    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path)

def feature_data_frame(years):
    naps = Naps()
    df = naps.get_years(years)

    #nfdb = NFDB()
    #fires = []
    #f_df = nfdb.data(years)
    #for index, row in f_df.iterrows():
    #    fires.append([row['LATITUDE'], row['LONGITUDE']])

    date = datetime.date(years[0], 1, 1)
    end_date = datetime.datetime.combine(datetime.date(years[-1], 12, 31), datetime.datetime.max.time())

    df = df.sort_values(by=['Longitude//Longitude', 'Latitude//Latitude'])
    station_ids = {row_id: idx for idx, row_id in enumerate(df['NAPS ID//Identifiant SNPA'].unique())}

    pm25_columns = [f"{station_id}_PM25" for station_id in station_ids.keys()]
    #fire_proximity_columns = [f"{station_id}_FireProximity" for station_id in station_ids.keys()]
    #columns = pm25_columns + fire_proximity_columns
    columns = pm25_columns

    date_range = pd.date_range(start=date, end=end_date, freq='h')  # hourly timestamps for the entire year
    pm25_df = pd.DataFrame(index=date_range, columns=columns)

    df['DateTime'] = pd.to_datetime(df['Date//Date'])

    last = 0
    for index, row in df.iterrows():
        row_id = row['NAPS ID//Identifiant SNPA']
        #fire_proximity = nfdb.find_closest(naps.coords(row), timestamp)

        # Calculate fire proximity only for fires active at this timestamp
        #fire_proximity = nfdb.find_closest_active_fire(
        #    [row['Latitude//Latitude'], row['Longitude//Longitude']],
        #    row['DateTime']
        #)
        #fire_proximity = round(fire_proximity)

        for i in range(1, 25):
            timestamp = row['DateTime'] + datetime.timedelta(hours=i - 1)
            pm25 = Naps.PM25(row, i, last)
            last = pm25

            pm25_df.loc[timestamp, f"{row_id}_PM25"] = pm25
            #pm25_df.loc[timestamp, f"{row_id}_FireProximity"] = fire_proximity

        if index % 1000 == 0:
            print(index)


    pm25_df = pm25_df.fillna(0)

    return pm25_df

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

    df = feature_data_frame([2020, 2021])
    print(df)
    df.to_csv('./data_cache/station_features_v2.csv')

if __name__ == '__main__':
    main()
