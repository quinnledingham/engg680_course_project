import pandas as pd
import math

class Vector2:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def print(self):
        print(f'{self.x}, {self.y}')

    @staticmethod
    def distance(a, b):
        return math.sqrt((b.x - a.x)**2 + (b.y - a.y)**2)

class Stations:
    database = pd.read_csv('Station Inventory EN.csv')

    @classmethod
    def get_coordinates(self, row):
        latitude = self.database.iloc[row]['Latitude (Decimal Degrees)']
        longitude = self.database.iloc[row]['Longitude (Decimal Degrees)']
        return Vector2(latitude, longitude)

    # used to find stations close to low air quality
    # find the num stations closest to position
    @classmethod
    def find_closest_stations(self, position, num):
        distance = Vector2.distance(position, Stations.get_coordinates(0))
        print(distance)

def main():
    Stations.find_closest_stations(Vector2(0, 0), 1)
    
if __name__ == '__main__':
    main()
