import pandas as pd
import os
import csv
from io import BytesIO
from io import StringIO
import requests as req

class Naps:
    stations_df = pd.read_csv('StationsNAPS.csv')

    @classmethod
    def get(self):
        return self.stations_df

    # day is a row from the dataframe returned by data
    @classmethod
    def PM25(self, day, hour):
        value = 0
        if (hour < 10):
            value = day[f'H0{hour}//H0{hour}']
        else:
            value = day[f'H{hour}//H{hour}']

        if value < 0 :
            value = 0 # set -999 values to 0

        return value

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
   
    def data(self, day=0, month=0, year=0):
        local_path = f"./data_cache/naps/PM2.5_{year}.csv"
    
        # check local store
        if not os.path.exists(local_path):
            # data comes in zip by year
            url = f"https://data-donnees.az.ec.gc.ca/api/file?path=%2Fair%2Fmonitor%2Fnational-air-pollution-surveillance-naps-program%2FData-Donnees%2F{year}%2FContinuousData-DonneesContinu%2FHourlyData-DonneesHoraires%2FPM25_{year}.csv"
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36",
            }
            response = req.get(url, headers=headers)
            assert response.status_code == 200

            #with open(f'{year}.csv', 'w', encoding="utf-8") as file:
            #    file.write(response.content.decode())
            
            self.remove_lines_from_csv(StringIO(response.content.decode()), local_path, 7)
        
        df = pd.read_csv(local_path)
        return df            

    def get_year(self, year):
        max = 0
        df = self.data(year=year)
        
        pm25_data = list()
        coords_data = list()
        
        for index, row in df.iterrows():
            row = df.iloc[index]
            for i in range(1, 25):
                pm25 = Naps.PM25(row, i)
                if pm25 > max:
                    max = pm25

                pm25_data.append(pm25)
                coords_data.append(Naps.coords(row))
        print(f"Max PM2.5: {max}")
        return pm25_data, coords_data
    
class Input:
    def __init__(self, pm25, coords):
        self.pm25_data = pm25
        self.coords_data = coords