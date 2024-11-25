import pandas as pd
import numpy as np
import shapefile
import os
import datetime

from sklearn.neighbors import BallTree, KDTree
from scipy.spatial import distance_matrix

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
    
    def get_years(self, years):
        years.sort()

        dfs = []
        for year in years:
            df = self.data(year)
            dfs.append(df)

        combined_df = pd.concat(dfs, ignore_index=True)
        self.df = combined_df

        self.fire_data = self.df[['LATITUDE', 'LONGITUDE', 'REP_DATE', 'OUT_DATE']].copy()
        self.fire_data['REP_DATE'] = pd.to_datetime(self.fire_data['REP_DATE'])
        self.fire_data['OUT_DATE'] = pd.to_datetime(self.fire_data['OUT_DATE'])

        self.fire_data['OUT_DATE'] = self.fire_data.apply(
            lambda row: row['REP_DATE'] + datetime.timedelta(days=7) if pd.isna(row['OUT_DATE']) else row['OUT_DATE'],
            axis=1
        )

        self.active_fire_dates = self.fire_data[['REP_DATE', 'OUT_DATE']]

    def find_closest_active_fire(self, point, timestamp):     
        # Ensure timestamp is datetime
        if isinstance(timestamp, str):
            timestamp = pd.to_datetime(timestamp)
    
        active_mask = ((self.active_fire_dates['REP_DATE'] <= timestamp) & (self.active_fire_dates['OUT_DATE'] >= timestamp))
        active_fires = self.fire_data[active_mask]

        if active_fires.empty:
            return 0

        # Create a KDTree for the active fires
        active_fire_coords = active_fires[['LATITUDE', 'LONGITUDE']].to_numpy()
        active_fire_tree = KDTree(active_fire_coords, metric='euclidean')

        # Get distance to closest fire (in degrees lat/long)
        dist, idx = active_fire_tree.query([point], k=1)
        
        # More generous distance thresholds (in degrees, roughly 111km per degree)
        if dist <= 0.5:  # ~55km
            return 9
        elif dist <= 1.0:  # ~111km
            return 8
        elif dist <= 2.0:  # ~222km
            return 7
        elif dist <= 3.0:  # ~333km
            return 6
        elif dist <= 4.0:  # ~444km
            return 5
        elif dist <= 5.0:  # ~555km
            return 4
        elif dist <= 6.0:  # ~666km
            return 3
        elif dist <= 7.0:  # ~777km
            return 2
        elif dist <= 8.0:  # ~888km
            return 1
        else:
            return 0