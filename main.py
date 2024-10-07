from collections import OrderedDict
from io import BytesIO

import pandas as pd
import math
import requests
import bs4
from zipfile import ZipFile
import os

class Vector2:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def print(self):
        print(f'{self.x}, {self.y}')

    @staticmethod
    def distance(a, b):
        return math.sqrt((b.x - a.x)**2 + (b.y - a.y)**2)

api_host = 'https://climate.weather.gc.ca'
    
class Stations:
    df = pd.read_csv('Station Inventory EN.csv')

    @classmethod
    def get_coordinates(self, row):
        latitude = self.df.iloc[row]['Latitude (Decimal Degrees)']
        longitude = self.df.iloc[row]['Longitude (Decimal Degrees)']
        return Vector2(latitude, longitude)

    @classmethod
    def print(self):
        print(self.df)

    # used to find stations close to low air quality
    # find the num stations closest to position
    @classmethod
    def find_closest_stations(self, position, num):
        distance = Vector2.distance(position, Stations.get_coordinates(0))
        print(distance)

    @classmethod
    def get_station(self, row):
        return self.df.iloc[row]

    @classmethod
    def get_station_with_name(self, name):
        ids = self.df.index[self.df['Name'] == name].tolist()
        return Stations.get_station(ids[0])

def get_data(station, day, month, year):
    local_path = f"./data_cache/climate_data/{station['Station ID']}_{year}_{month}.csv"

    if not os.path.exists(local_path):
        # this link returns the whole month regardless of day specified
        url = f"https://climate.weather.gc.ca/climate_data/bulk_data_e.html?format=csv"\
              f"&stationID={station['Station ID']}"\
              f"&Year={year}"\
              f"&Month={month}"\
              f"&Day=1"\
              f"&timeframe=1"    
    
        response = requests.get(url)

        data = BytesIO(response.content)
        df = pd.read_csv(data)
        df.to_csv(local_path)
    else:
        df = pd.read_csv(local_path)

    return df

def naps_data(day=0, month=0, year=0):
    local_path = f"./data_cache/naps/{year}.csv"
    
    # check local store
    if not os.path.exists(local_path):
        # data comes in zip by year
        url = f"https://data-donnees.az.ec.gc.ca/api/file?path=%2Fair%2Fmonitor%2Fnational-air-pollution-surveillance-naps-program%2FData-Donnees%2F{year}%2FIntegratedData-DonneesPonctuelles%2F{year}_NAPSReferenceMethodPM25-PM25MethodeReferenceSNPA.zip"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36",
        }
        response = requests.get(url, headers=headers)
        assert response.status_code == 200

        # write zip to csv in data cache
        res_data = BytesIO(response.content)
        in_zip = ZipFile(res_data)

        files = []
        for filename in in_zip.namelist():
            check = '_FR' in filename or filename.endswith(".xlsx")
            if not check:
                file = in_zip.read(filename)
                files.append(BytesIO(file))
        in_zip.close()
        df = pd.concat(map(pd.read_csv, files), ignore_index=True)
        df.to_csv(local_path)
        
    df = pd.read_csv(local_path)
    return df

def create_data_cache():
    paths = ["./data_cache", "./data_cache/naps", "./data_cache/climate_data/"]
    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path)

def main():    
    create_data_cache()
    
    station = Stations.get_station_with_name("FOGGY LO")
    print(f'Station ID: {station['Station ID']}')
    print(f'Climate ID: {station['Climate ID']}')
    data = get_data(station, 24, 9, 2024)
    # print(data['Wind Spd (km/h)'])
    print(naps_data(year=2021))
    
if __name__ == '__main__':
    main()
