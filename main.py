from collections import OrderedDict
from io import BytesIO

import pandas as pd
import math
import requests
import bs4
import wget

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
    database = pd.read_csv('Station Inventory EN.csv')

    @classmethod
    def get_coordinates(self, row):
        latitude = self.database.iloc[row]['Latitude (Decimal Degrees)']
        longitude = self.database.iloc[row]['Longitude (Decimal Degrees)']
        return Vector2(latitude, longitude)

    @classmethod
    def print(self):
        print(self.database)

    # used to find stations close to low air quality
    # find the num stations closest to position
    @classmethod
    def find_closest_stations(self, position, num):
        distance = Vector2.distance(position, Stations.get_coordinates(0))
        print(distance)

    @classmethod
    def get_station(self, row):
        return self.database.iloc[row]

    @classmethod
    def get_station_with_name(self, name):
        ids = self.database.index[self.database['Name'] == name].tolist()
        return Stations.get_station(ids[0])

def get_data(station, day, month, year):
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

    return df

def main():
    station = Stations.get_station_with_name("FOGGY LO")
    print(f'Station ID: {station['Station ID']}')
    print(f'Climate ID: {station['Climate ID']}')
    data = get_data(station, 24, 9, 2024)
    print(data['Wind Spd (km/h)'])
    
if __name__ == '__main__':
    main()
