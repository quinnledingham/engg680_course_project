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
        
        return df

import pickle

def create_data_cache():
    paths = ["./data_cache", "./data_cache/naps", "./data_cache/climate_data/"]
    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path)

block_size = 256

def collect_day_data(date_str, df_by_date, closest_stations_cache, station_ids):
    day_data = [[], [], []]
    if date_str in df_by_date:
        day_df = df_by_date[date_str]

        # Collect PM2.5 values for each station's closest stations for all hours
        for station_id, closest_stations in closest_stations_cache.items():
            for hour in range(1, 25):
                hour_column = f'H{hour:02d}//H{hour:02d}'
                
                # Collect PM2.5 data for each close station in closest_stations
                for close_id in closest_stations:
                    pm25_values = day_df[day_df['NAPS ID//Identifiant SNPA'] == close_id].get(hour_column)
                    if not pm25_values.empty:
                        pm25_value = pm25_values.values[0]
                        day_data[0].append(pm25_value)
                        day_data[1].append(hour - 1)  # Hour index
                        day_data[2].append(station_ids[close_id])
    return day_data

def create_blocks(year):
    start_date = datetime.date(2021, 1, 1)
    end_date = datetime.date(2021, 12, 31)
    naps = Naps()
    df = naps.data(year=year)
    station_ids = {}
    num = 0

    # Create a lookup table for station ids
    for row_id in df['NAPS ID//Identifiant SNPA'].unique():
        station_ids[row_id] = num
        num += 1

    # Filter stations once by relevant station IDs
    filtered_stations_df = naps.stations_df[naps.stations_df['NAPS_ID'].isin(station_ids.keys())]

    # Cache closest stations for each station ID
    closest_stations_cache = {
        row['NAPS_ID']: naps.find_5_closest(filtered_stations_df, naps.station_coords(row))
        for _, row in filtered_stations_df.iterrows()
    }

    # Prepare data structure to store results
    all_data_blocks = [[], [], []]

    # Group data by date once to avoid repeated filtering
    df_by_date = {date: day_df for date, day_df in df.groupby('Date//Date')
                  if start_date <= datetime.datetime.strptime(date, "%Y-%m-%d").date() <= end_date}

    # Parallel data collection by date
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(
            collect_day_data,
            df_by_date.keys(),
            [df_by_date] * len(df_by_date),
            [closest_stations_cache] * len(df_by_date),
            [station_ids] * len(df_by_date)
        ))

    # Combine results into all_data_blocks
    for result in results:
        all_data_blocks[0].extend(result[0])
        all_data_blocks[1].extend(result[1])
        all_data_blocks[2].extend(result[2])

    # Convert to numpy arrays for faster manipulation
    all_data_blocks = [np.array(data) for data in all_data_blocks]

    # Initialize and save data
    input_data = Input_Data(all_data_blocks, station_ids)
    input_data.save_data("./data_cache/new.data")

def sorted_by_time(year):
    naps = Naps()
    df = naps.data(year=year)
    date = datetime.date(2021, 1, 1)
    end_date = datetime.date(2021, 12, 31)
    delta = datetime.timedelta(days=1)

    value = []
    variable = []
    time = []
    station_ids = {}

    time_index = 0
    num = 0
    while(date <= end_date):
        one_day = df[df['Date//Date'] == date.strftime("%Y-%m-%d")]

        for i in range(1, 25):
            for index, row in one_day.iterrows():
                row_id = row['NAPS ID//Identifiant SNPA']

                # create lookup table for station ids
                if row_id not in station_ids:
                    station_ids[row_id] = num
                    num += 1

                pm25 = Naps.PM25(row, i)
                value.append(pm25)
                variable.append(row_id)
                time.append(time_index)
                
            time_index += 1

        date += delta

        
    input = Input_Data(value, variable, time, station_ids)
    input.save_data("./data_cache/test.data")

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

   #sorted_by_time(2021)
    test = create_blocks(2021)

    
if __name__ == '__main__':
    main()
