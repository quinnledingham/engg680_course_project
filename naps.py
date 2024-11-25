import pandas as pd
import os
import csv
from io import BytesIO
from io import StringIO
import requests as req
import datetime
from geopy import distance

def point_distance(point1, point2):
    return distance.distance(point1, point2)

class Naps:
    stations_df = pd.read_csv('StationsNAPS.csv')

    @classmethod
    def get(self):
        return self.stations_df

    # day is a row from the dataframe returned by data
    @classmethod
    def PM25(self, day, hour, last):
        value = 0
        if (hour < 10):
            value = day[f'H0{hour}//H0{hour}']
        else:
            value = day[f'H{hour}//H{hour}']

        if value < 0 :
            value = last # set -999 values to 0

        return value

    @classmethod
    def station_coords(self, station):
        return [station['Latitude'], station['Longitude']]


    @classmethod
    def coords(self, day):
        return [day['Latitude//Latitude'], day['Longitude//Longitude']]
    
    def remove_lines_from_csv(self, input_file, output_file, lines_to_remove):
        reader = csv.reader(input_file)
        # Skip the specified number of lines
        for _ in range(lines_to_remove):
            next(reader)

        # Write the remaining lines to the output file
        with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
            writer = csv.writer(outfile)
            for row in reader:
                writer.writerow(row)
   
    def find_5_closest(self, stations, point):
        distances = []
        
        for index, row in stations.iterrows():
            station_id = row['NAPS_ID']
            temp_point = [row['Latitude'], row['Longitude']]
            dist = point_distance(point, temp_point)
            distances.append((station_id, dist))
        
        # Sort distances by the distance value and take the 5 closest
        distances = sorted(distances, key=lambda x: x[1])[:5]
        
        # Convert to a dictionary with station_id as key and distance as value
        ids = {station_id: dist for station_id, dist in distances}
        
        return ids

    def data(self, year):
        local_path = f"./data_cache/naps/PM2.5_{year}.csv"
    
        # check local store
        if not os.path.exists(local_path):
            # data comes in zip by year
            url = f"https://data-donnees.az.ec.gc.ca/api/file?path=%2Fair%2Fmonitor%2Fnational-air-pollution-surveillance-naps-program%2FData-Donnees%2F{year}%2FContinuousData-DonneesContinu%2FHourlyData-DonneesHoraires%2FPM25_{year}.csv"
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36",
            }
            response = req.get(url, headers=headers)

            if response.status_code != 200:
                print(f"Couldn't find {year}")
            assert response.status_code == 200

            #with open(f'{year}.csv', 'w', encoding="utf-8") as file:
            #    file.write(response.content.decode())
            
            self.remove_lines_from_csv(StringIO(response.content.decode()), local_path, 7)
        
        df = pd.read_csv(local_path)
        return df            

    def get_years(self, years):
        years.sort()

        station_sets = []
        for year in years:
            df = self.data(year)
            stations = set(df['NAPS ID//Identifiant SNPA'].unique())
            station_sets.append(stations)
        
        # Find intersection of all station sets
        common_stations = set.intersection(*station_sets)

        dfs = []
        for year in years:
            df = self.data(year)
            df = df[df['NAPS ID//Identifiant SNPA'].isin(common_stations)]
            df = df.sort_values(by=['Longitude//Longitude', 'Latitude//Latitude'])
            dfs.append(df)

        combined_df = pd.concat(dfs, ignore_index=True)
        self.df = combined_df

    def find_missing_stations(self, years):
        year_stations = {}
        all_stations = set()
        
        for year in sorted(years):
            df = self.data(year)
            stations = set(df['NAPS ID//Identifiant SNPA'].unique())
            year_stations[year] = stations
            all_stations.update(stations)
        
        # Find stations missing in each year
        missing_by_year = {
            year: all_stations - stations 
            for year, stations in year_stations.items()
        }
        
        # Find stations not present in all years
        stations_in_all_years = set.intersection(*year_stations.values())
        inconsistent_stations = all_stations - stations_in_all_years
        
        # Create detailed presence map
        station_presence = {}
        for station in all_stations:
            presence = {
                year: station in year_stations[year]
                for year in years
            }
            if not all(presence.values()):  # Only include if not in all years
                station_presence[station] = presence
        
        return {
            'all_stations': sorted(list(all_stations)),
            'stations_in_all_years': sorted(list(stations_in_all_years)),
            'inconsistent_stations': sorted(list(inconsistent_stations)),
            'missing_by_year': {
                year: sorted(list(stations))
                for year, stations in missing_by_year.items()
            },
            'station_presence_map': station_presence
        }

    def print_station_analysis(self, years):
        analysis = self.find_missing_stations(years)
        
        print(f"Total unique stations: {len(analysis['all_stations'])}")
        print(f"Stations present in all years: {len(analysis['stations_in_all_years'])}")
        print(f"Stations with inconsistent presence: {len(analysis['inconsistent_stations'])}\n")
        
        print("Missing stations by year:")
        for year, missing in analysis['missing_by_year'].items():
            if missing:
                print(f"{year}: {len(missing)} stations missing")
        
        print("\nDetailed presence map for inconsistent stations:")
        for station, presence in analysis['station_presence_map'].items():
            years_present = [year for year, present in presence.items() if present]
            years_missing = [year for year, present in presence.items() if not present]
            
            print(f"\nStation {station}:")
            print(f"  Present in: {', '.join(map(str, years_present))}")
            print(f"  Missing in: {', '.join(map(str, years_missing))}")

    @classmethod
    def get_station_table(self):
        num = 0
        station_ids = {}
        for index, row in self.stations_df.iterrows():
            if row['Status'] == 1 and row['PM_25_Continuous'] == 'X':
                station_ids[row['NAPS_ID']] = num
                num += 1

        return station_ids
    
def main():
    naps = Naps()
    stations = naps.get()
    day = stations.iloc[1]
    print(day)
    naps.find_5_closest(naps.station_coords(day))
    #pm25_data, station_data, station_ids = naps.get_year2(2021)

if __name__ == '__main__':
    main()
