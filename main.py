from collections import OrderedDict
from io import BytesIO
from io import StringIO

import pandas as pd
import math
import requests
import bs4
from zipfile import ZipFile
import os
import csv

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
        if not ids:
            print(f'Did not find {name}\n')
        return Stations.get_station(ids[0])

class Station:
    df = pd.DataFrame()
    
    def __init__(self, name):
        self.info = Stations.get_station_with_name(name)

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
            response = requests.get(url)
            assert response.status_code == 200

            data = BytesIO(response.content)
            df = pd.read_csv(data)
            df.to_csv(local_path)
        else:
            df = pd.read_csv(local_path)

        self.df = pd.concat([self.df, df], ignore_index=True)
        return df

def naps_data(day=0, month=0, year=0):
    local_path = f"./data_cache/naps/{year}.csv"
    
    # check local store
    if not os.path.exists(local_path):
        # data comes in zip by year
        #url = f"https://data-donnees.az.ec.gc.ca/api/file?path=%2Fair%2Fmonitor%2Fnational-air-pollution-surveillance-naps-program%2FData-Donnees%2F{year}%2FIntegratedData-DonneesPonctuelles%2F{year}_NAPSReferenceMethodPM25-PM25MethodeReferenceSNPA.zip"
        url = f"https://data-donnees.az.ec.gc.ca/api/file?path=%2Fair%2Fmonitor%2Fnational-air-pollution-surveillance-naps-program%2FData-Donnees%2F{year}%2FContinuousData-DonneesContinu%2FHourlyData-DonneesHoraires%2FPM25_{year}.csv"
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

def remove_lines_from_csv(input_file, output_file, lines_to_remove):
    #with open(input_file, 'r', newline='', encoding='utf-8') as infile:
    reader = csv.reader(input_file)
    # Skip the specified number of lines
    for _ in range(lines_to_remove):
        next(reader)

    # Write the remaining lines to the output file
    with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
        writer = csv.writer(outfile)
        for row in reader:
            writer.writerow(row)
                                                                                                                        
def naps_cont_data(day=0, month=0, year=0):
    local_path = f"./data_cache/naps/PM2.5_{year}.csv"
    
    # check local store
    if not os.path.exists(local_path):
        # data comes in zip by year
        url = f"https://data-donnees.az.ec.gc.ca/api/file?path=%2Fair%2Fmonitor%2Fnational-air-pollution-surveillance-naps-program%2FData-Donnees%2F{year}%2FContinuousData-DonneesContinu%2FHourlyData-DonneesHoraires%2FPM25_{year}.csv"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36",
        }
        response = requests.get(url, headers=headers)
        assert response.status_code == 200

        remove_lines_from_csv(StringIO(response.content.decode()), local_path, 7)
        
    df = pd.read_csv(local_path)
    return df
    

def create_data_cache():
    paths = ["./data_cache", "./data_cache/naps", "./data_cache/climate_data/"]
    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path)

def main():    
    create_data_cache()
    
    #station = Station("CALGARY INTL A")
    #station.print()
    #station.climate_data(2, 3, 2024)
    #station.climate_data(24, 9, 2024)
    
    #print(naps_data(year=2023))
    print(naps_cont_data(year=2021))
    
if __name__ == '__main__':
    main()
