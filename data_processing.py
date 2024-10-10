import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from haversine import haversine, Unit
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from meteostat import Point, Hourly
from datetime import datetime

# Select relevant features
features = ['tripduration', 'distance', 'user_type_encoded', 'speed',
                    'temp', 'wspd', 'prcp', 'coco', 
                    'start_hour', 'start_dayofweek', 'start_month', 
                    'end_hour', 'end_dayofweek', 'end_month']

def process_data(city_name = 'boston', start = datetime(2023, 1, 1), end = datetime(2023, 1, 31), pca=True, scaling=False):
    bike_weather_data = read_and_connect_data(city_name, start, end)
    bike_weather_data = feature_engineering(bike_weather_data)

    if scaling:   bike_weather_data = scale_data(bike_weather_data)
    if pca:       bike_weather_data = apply_pca(bike_weather_data)

    return bike_weather_data

def read_and_connect_data(city_name, start, end):
    # Load bike data
    if city_name == 'boston':
        bike_data = pd.read_csv('data/boston/202301-bluebikes-tripdata.csv')
        boston_position = Point(42.3601, -71.0589)  # Latitude and Longitude for Boston
    
    weather_data_hourly = Hourly(boston_position, start, end).fetch().reset_index()

    # Ensure both are in datetime format
    bike_data['starttime'] = pd.to_datetime(bike_data['starttime'])
    weather_data_hourly['time'] = pd.to_datetime(weather_data_hourly['time'])

    bike_weather_data = pd.merge_asof(bike_data, weather_data_hourly, left_on='starttime', right_on='time', direction='nearest')

    return bike_weather_data

def feature_engineering(bike_weather_data):
    bike_weather_data = distance_feature(bike_weather_data)

    bike_weather_data = date_related_features(bike_weather_data)

    bike_weather_data = speed_feature(bike_weather_data)

    bike_weather_data = user_type_feature_encoding(bike_weather_data)

    bike_weather_data = bike_weather_data[features]

    return bike_weather_data

# Extract the hour/day/month as features
def date_related_features(bike_weather_data):
    bike_weather_data['start_hour'] = bike_weather_data['starttime'].dt.hour
    bike_weather_data['start_dayofweek'] = bike_weather_data['starttime'].dt.dayofweek
    bike_weather_data['start_month'] = bike_weather_data['starttime'].dt.month

    bike_weather_data['end_hour'] = pd.to_datetime(bike_weather_data['stoptime']).dt.hour
    bike_weather_data['end_dayofweek'] = pd.to_datetime(bike_weather_data['stoptime']).dt.dayofweek
    bike_weather_data['end_month'] = pd.to_datetime(bike_weather_data['stoptime']).dt.month

    return bike_weather_data

# Calculate Distance (in kilometers) 
def distance_feature(bike_weather_data):
    start_coords = list(zip(bike_weather_data['start station latitude'], bike_weather_data['start station longitude']))
    end_coords = list(zip(bike_weather_data['end station latitude'], bike_weather_data['end station longitude']))
    bike_weather_data['distance'] = np.array([haversine(start, end, unit=Unit.KILOMETERS) for start, end in zip(start_coords, end_coords)])

    return bike_weather_data

# Create a new feature as Speed to capture relation between duration and distance better
# Short distances covered at high speeds (potentially indicating misuse or errors).
# Long distances taking a very long time (could indicate that the bike was left unlocked or a user took an extended break).
def speed_feature(bike_weather_data):
    bike_weather_data['speed'] = bike_weather_data['distance'] / (bike_weather_data['tripduration'] / 3600)  # Speed in km/h

    return bike_weather_data

def user_type_feature_encoding(bike_weather_data):
    label_encoder = LabelEncoder()
    bike_weather_data['user_type_encoded'] = label_encoder.fit_transform(bike_weather_data['usertype'])

    return bike_weather_data

def scale_data(bike_weather_data):
    scaler = StandardScaler()
    bike_weather_data = scaler.fit_transform(bike_weather_data)

    return bike_weather_data

def apply_pca(bike_weather_data, n_components=5):
    pca = PCA(n_components=n_components)
    bike_weather_data = pca.fit_transform(bike_weather_data)

    return bike_weather_data