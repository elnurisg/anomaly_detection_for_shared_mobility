import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from haversine import haversine, Unit
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from meteostat import Point, Hourly
from datetime import datetime
from shapely.geometry import Point as GeoPoint
import geopandas as gpd
import partridge as pt
import holidays


# Select relevant features
features = ['tripduration', 'distance', 'user_type_encoded', 'speed',
                    'temp', 'wspd', 'prcp', 'coco', 
                    'start_hour', 'start_dayofweek', 'start_month', 
                    'end_hour', 'end_dayofweek', 'end_month',
                    'special_day',
                    'start_nearby_transit_stops', 'end_nearby_transit_stops']

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

    bike_weather_data = add_holidays_to_data(bike_weather_data)
    # bike_weather_data = add_mass_transit_data(bike_weather_data)

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

def categorize_tripduration(bike_weather_data):
    bike_weather_data['tripduration_type'] = np.where(
        bike_weather_data['tripduration'] < 900, 0,  # "very_short_trip"
        np.where(
            (bike_weather_data['tripduration'] >= 900) & (bike_weather_data['tripduration'] < 3600), 1,  # "short_trip"
            np.where(
                (bike_weather_data['tripduration'] >= 3600) & (bike_weather_data['tripduration'] <= 6 * 3600), 2,  # "long_trip"
                3  # "very_long_trip"
            )
        )
    )

                            ### Mass Transit Data

def load_stops_data():
    """Load and prepare stops data as a GeoDataFrame."""
    feed = pt.load_feed("gtfs")
    stops = feed.stops
    stops['geometry'] = stops.apply(lambda x: GeoPoint(x['stop_lon'], x['stop_lat']), axis=1)
    stops_gdf = gpd.GeoDataFrame(stops[['stop_id', 'geometry']], geometry='geometry')
    stops_gdf.set_crs(epsg=4326, inplace=True)
    stops_gdf = stops_gdf.to_crs(epsg=3857)
    
    return stops_gdf


def prepare_bike_data(bike_weather_data):
    """Prepare bike data with geometry columns and convert to a GeoDataFrame."""
    bike_weather_data['start_geometry'] = bike_weather_data.apply(
        lambda x: GeoPoint(x['start station longitude'], x['start station latitude']), axis=1
    )
    bike_weather_data['end_geometry'] = bike_weather_data.apply(
        lambda x: GeoPoint(x['end station longitude'], x['end station latitude']), axis=1
    )
    bike_gdf = gpd.GeoDataFrame(bike_weather_data, geometry='start_geometry')
    
    return bike_gdf


def calculate_nearby_stops(bike_gdf, stops_gdf, radius, geometry_column):
    """Calculate the number of nearby transit stops for the given geometry column."""
    bike_gdf.set_geometry(geometry_column, inplace=True)
    bike_gdf = bike_gdf.set_crs(epsg=4326, inplace=True).to_crs(epsg=3857)
    bike_gdf['buffer'] = bike_gdf.geometry.buffer(radius)
    nearby_stops = gpd.sjoin(stops_gdf, bike_gdf.set_geometry('buffer'), how='inner', predicate='within')
    nearby_counts = nearby_stops.groupby('index_right').size()
    
    return nearby_counts


def add_nearby_stops_counts(bike_weather_data, start_counts, end_counts):
    """Add nearby stops counts to bike_weather_data and encode them."""
    bike_weather_data['start_nearby_transit_stops'] = bike_weather_data.index.map(start_counts).fillna(0).astype(int)
    bike_weather_data['end_nearby_transit_stops'] = bike_weather_data.index.map(end_counts).fillna(0).astype(int)
    
    # Apply encoding
    bike_weather_data['start_nearby_transit_stops'] = bike_weather_data['start_nearby_transit_stops'].apply(encode_nearby_stops)
    bike_weather_data['end_nearby_transit_stops'] = bike_weather_data['end_nearby_transit_stops'].apply(encode_nearby_stops)
    
    return bike_weather_data


def add_mass_transit_data(bike_weather_data, radius=500):
    """Main function to add mass transit data to bike_weather_data."""
    stops_gdf = load_stops_data()
    bike_gdf = prepare_bike_data(bike_weather_data)
    
    # Calculate nearby stops for start and end stations
    start_counts = calculate_nearby_stops(bike_gdf, stops_gdf, radius, 'start_geometry')
    end_counts = calculate_nearby_stops(bike_gdf, stops_gdf, radius, 'end_geometry')
    
    # Add counts and encode them
    bike_weather_data = add_nearby_stops_counts(bike_weather_data, start_counts, end_counts)
    
    return bike_weather_data


def encode_nearby_stops(count):
    if count <= 5:
        return 0  # 0 = Badly Connected
    elif 6 <= count <= 15:
        return 1  # 1 = Moderately Connected
    else:
        return 2  # 2 = Well Connected


# Add holidays to the data

def add_holidays_to_data(bike_weather_data, start_date="2023-01-01", end_date="2023-01-31"):

    # Generate US holidays for 2023
    us_holidays = holidays.US(years=2023)
    date_range = pd.date_range(start=start_date, end=end_date)

    # Create a DataFrame with dates and corresponding holiday names
    special_days = [us_holidays.get(date, None) for date in date_range]

    df_special_days = pd.DataFrame({
        'date': pd.to_datetime(date_range).normalize,  # Normalize to ensure datetime64[ns] type
        'special_day': special_days
    })

    # Convert 'starttime' to date
    bike_weather_data['date'] = pd.to_datetime(bike_weather_data['starttime']).dt

    # Merge the dataframes
    bike_weather_data = bike_weather_data.merge(df_special_days, how='left', on='date')

    # Drop the 'date' column as it's no longer needed
    bike_weather_data.drop(columns=['date'], inplace=True)

    return bike_weather_data