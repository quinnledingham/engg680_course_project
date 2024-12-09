from io import BytesIO

import pandas as pd
import requests as req
import os
from geopy import distance
import datetime

from naps import Naps
from nfdb import NFDB

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

def create_data_cache():
    paths = ["./data_cache", "./data_cache/naps", "./data_cache/climate_data/", "./data_cache/fires"]
    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path)

def process_station(args):
    station_id, naps, nfdb, date_range = args
    station_data = naps.df[naps.df['NAPS ID//Identifiant SNPA'] == station_id].copy()
    
    pm25_name = f"pm25_{station_id}"
    fire_name = f"fire_{station_id}"
    station_df = pd.DataFrame(index=date_range, columns=[pm25_name, fire_name])
    
    last = 0
    for _, row in station_data.iterrows():
        base_datetime = pd.to_datetime(row['DateTime'])
        
        fire_proximity = nfdb.find_closest_active_fire(
            [row['Latitude//Latitude'], row['Longitude//Longitude']],
            row['DateTime']
        )
        fire_proximity = round(fire_proximity)

        # Process each hour
        for hour in range(1, 25):
            timestamp = base_datetime + pd.Timedelta(hours=hour-1)
            pm25_value = Naps.PM25(row, hour, last)
            last = pm25_value
            
            if timestamp in station_df.index:
                station_df.loc[timestamp, pm25_name] = pm25_value
                station_df.loc[timestamp, fire_name] = fire_proximity
    
    return station_df

def feature_data_frame_parallel(years):
    import multiprocessing as mp
    
    naps = Naps()
    naps.get_years(years)

    nfdb = NFDB()
    nfdb.get_years(years)

    start_date = datetime.datetime(years[0], 1, 1)
    end_date = datetime.datetime(years[-1], 12, 31, 23, 59, 59)
    date_range = pd.date_range(start=start_date, end=end_date, freq='h')
    
    naps.df['DateTime'] = pd.to_datetime(naps.df['Date//Date'])
    
    station_ids = naps.df['NAPS ID//Identifiant SNPA'].unique()
    
    args_list = [(station_id, naps, nfdb, date_range) for station_id in station_ids]
    
    with mp.Pool(processes=mp.cpu_count()) as pool:
        station_dfs = pool.map(process_station, args_list)
    
    result_df = pd.concat(station_dfs, axis=1)
    result_df = result_df.fillna(0)
    
    return result_df

def feature_data_frame(years):
    naps = Naps()
    naps.get_years(years)

    nfdb = NFDB()
    nfdb.get_years(years)
    
    start_date = datetime.datetime(years[0], 1, 1)
    end_date = datetime.datetime(years[-1], 12, 31, 23, 59, 59)
    date_range = pd.date_range(start=start_date, end=end_date, freq='h')
    
    naps.df['DateTime'] = pd.to_datetime(naps.df['Date//Date'])
    
    station_ids = naps.df['NAPS ID//Identifiant SNPA'].unique()
    
    station_dfs = []
    for station_id in station_ids:
        station_df = process_station((station_id, naps, nfdb, date_range))
        print(station_df)
        station_dfs.append(station_df)
    
    result_df = pd.concat(station_dfs, axis=1)
    result_df = result_df.fillna(0)
    
    return result_df

def main():    
    create_data_cache()
    
    #naps = Naps()
    #naps.print_station_analysis([2018, 2019, 2020, 2021, 2022])
    df = feature_data_frame_parallel([2018, 2019, 2020, 2021, 2022])
    print(df)
    df.to_csv('./data_cache/station_features_v2.csv')

if __name__ == '__main__':
    main()
