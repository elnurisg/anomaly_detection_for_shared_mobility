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
from shapely.geometry import LineString


# Select relevant features
features = ['tripduration', 'distance', 'user_type_encoded', 'speed',
                    'temp', 'wspd', 'prcp', 'coco', 
                    'start_hour', 'start_dayofweek', 'start_month', 'start_day',
                    'end_hour', 'end_dayofweek', 'end_month', 'end_day',
                    'special_day', 'start_is_weekend', 'end_is_weekend',
                    'start_nearby_transit_stops', 'end_nearby_transit_stops',
                    'start_neighborhood', 'end_neighborhood',
                    'route_area', 'route',
                    'start_geometry', 'end_geometry',
                    'start station id', 'end station id',
                    'start station name', 'end station name']

# Outlier thresholds
MAX_TRIPDURATION = 86400  # 1 day in seconds
MAX_DISTANCE = 100  # 100 km

def prepare_station_data(file_path):
    """
    Splits the input data into start and end datasets, renames columns for consistency,
    and adds an is_start flag.
    """
    # Start dataset
    data = gpd.read_parquet(file_path)

    start_data = data[[
        'start station id', 'start station name', 'start_neighborhood', 'start_hour',
        'start_dayofweek', 'start_month', 'special_day', 'tripduration', 'distance',
        'speed', 'user_type_encoded', 'temp', 'prcp', 'wspd', 'coco',
        'start_nearby_transit_stops', 'start_geometry',
        'start_day', 'start_is_weekend'
    ]].copy()

    start_data.rename(columns={
        'start station id': 'station_id',
        'start station name': 'station_name',
        'start_neighborhood': 'neighborhood',
        'start_hour': 'hour',
        'start_dayofweek': 'dayofweek',
        'start_day': 'day',
        'start_is_weekend': 'is_weekend',
        'start_month': 'month',
        'start_nearby_transit_stops': 'nearby_transit_stops',
        'start_geometry': 'geometry'
    }, inplace=True)
    start_data['is_start'] = 1

    # End dataset
    end_data = data[[
        'end station id', 'end station name', 'end_neighborhood', 'end_hour',
        'end_dayofweek', 'end_month', 'special_day', 'tripduration', 'distance',
        'speed', 'user_type_encoded', 'temp', 'prcp', 'wspd', 'coco',
        'end_nearby_transit_stops', 'end_geometry',
        'end_day', 'end_is_weekend'
    ]].copy()

    end_data.rename(columns={
        'end station id': 'station_id',
        'end station name': 'station_name',
        'end_neighborhood': 'neighborhood',
        'end_hour': 'hour',
        'end_dayofweek': 'dayofweek',
        'end_day': 'day',
        'end_is_weekend': 'is_weekend',
        'end_month': 'month',
        'end_nearby_transit_stops': 'nearby_transit_stops',
        'end_geometry': 'geometry'
    }, inplace=True)
    end_data['is_start'] = 0

    # Combine datasets
    combined_data = pd.concat([start_data, end_data], ignore_index=True)
    return combined_data


def filter_outliers(data, max_tripduration=MAX_TRIPDURATION, max_distance=MAX_DISTANCE):
    """
    Filters out extreme outliers based on tripduration and distance.
    """
    return data[
        (data['tripduration'] <= max_tripduration) &
        (data['distance'] <= max_distance)
    ]


def extract_station_metadata(data):
    """
    Extracts station-specific metadata, ensuring alignment with the grouped data.
    Includes time-dependent features like day, month, and is_start.
    """
    station_metadata = data[[
        'station_id', 'hour', 'day', 'month', 'is_start',  # Include grouping keys
        'station_name', 'neighborhood', 'geometry', 
        'nearby_transit_stops', 'dayofweek', 'is_weekend', 'special_day'
    ]].drop_duplicates(subset=['station_id', 'hour', 'day', 'month', 'is_start'])
    return station_metadata


def aggregate_station_time_metrics(data):
    """
    Groups the data by station_id, hour, day, month, and is_start,
    and computes aggregated metrics (excluding station-specific columns).
    """
    agg_functions = {
        'tripduration': ['count', 'mean', 'std', 'median'],
        'distance': ['mean', 'std', 'median'],
        'speed': ['mean', 'std', 'median'],
        'user_type_encoded': 'mean',
        'temp': 'mean',
        'prcp': 'mean',
        'wspd': 'mean',
        'coco': 'mean'
    }

    aggregated = data.groupby(
        ['station_id', 'hour', 'day', 'month', 'is_start']
    ).agg(agg_functions).reset_index()

    # Flatten column names
    aggregated.columns = [
        '_'.join(col).strip('_') if isinstance(col, tuple) else col
        for col in aggregated.columns
    ]

    # Replace NaN std with 0
    std_columns = [col for col in aggregated.columns if '_std' in col]
    aggregated[std_columns] = aggregated[std_columns].fillna(0)
    aggregated.rename(columns={'tripduration_count': 'count'}, inplace=True)

    return aggregated


def from_trip_to_station_focused(file_path):
    """
    High-level function to process trip data and generate station-time-focused metrics.
    Ensures station-specific features remain consistent.
    """
    # Step 1: Prepare start and end datasets
    station_data = prepare_station_data(file_path)

    # Step 2: Extract station-specific metadata
    station_metadata = extract_station_metadata(station_data)

    # Step 3: Filter outliers
    station_data = filter_outliers(station_data)

    # Step 4: Aggregate metrics (excluding station-specific features)
    aggregated_data = aggregate_station_time_metrics(station_data)

    # Step 5: Merge station-specific metadata
    final_data = aggregated_data.merge(
        station_metadata,
        on=['station_id', 'hour', 'day', 'month', 'is_start'],  # Match aggregation keys
        how='left'
        )
    return final_data

def bike_trip_process_data_and_save(city_name = 'boston', start = datetime(2023, 1, 1), end = datetime(2023, 1, 31), pca=True, scaling=False):
    bike_weather_data = bike_trip_read_and_connect_data(city_name, start, end)
    bike_weather_data = bike_trip_feature_engineering(bike_weather_data)

    if scaling:   bike_weather_data = scale_data(bike_weather_data)
    if pca:       bike_weather_data = apply_pca(bike_weather_data)
    bike_weather_data.to_parquet("../data/boston/bike_trip_focused_data.parquet")

    print("bike trip focused data saved successfully as Parquet!")
    return bike_weather_data

def bike_trip_read_and_connect_data(city_name, start, end):
    # Load bike data
    if city_name == 'boston':
        bike_data = pd.read_csv('../data/boston/202301-bluebikes-tripdata.csv')
        boston_position = Point(42.3601, -71.0589)  # Latitude and Longitude for Boston
    
    weather_data_hourly = Hourly(boston_position, start, end).fetch().reset_index()

    # Ensure both are in datetime format
    bike_data['starttime'] = pd.to_datetime(bike_data['starttime'])
    weather_data_hourly['time'] = pd.to_datetime(weather_data_hourly['time'])
   
    bike_weather_data = pd.merge_asof(bike_data, weather_data_hourly, left_on='starttime', right_on='time', direction='nearest')
    

    bike_weather_data = add_holidays_to_data(bike_weather_data)
    bike_weather_data = add_mass_transit_data(bike_weather_data)

    neighborhoods = gpd.read_file('../data/neighborhood_data/boston_cambridge_neighborhoods.geojson')
    bike_weather_data = add_neighborhoods(bike_weather_data, neighborhoods)

    return bike_weather_data

def bike_trip_feature_engineering(bike_weather_data):
    bike_weather_data = distance_feature(bike_weather_data)

    bike_weather_data = date_related_features(bike_weather_data)

    bike_weather_data = speed_feature(bike_weather_data)

    bike_weather_data = user_type_feature_encoding(bike_weather_data)

    bike_weather_data = create_route_feature(bike_weather_data)

    # encode route_area
    bike_weather_data = encode_label(bike_weather_data, 'route_area')

    #encode neighbiorhoods
    bike_weather_data = encode_label(bike_weather_data, 'start_neighborhood')
    bike_weather_data = encode_label(bike_weather_data, 'end_neighborhood')

    bike_weather_data = bike_weather_data[features]

    return bike_weather_data

# Extract the hour/day/month as features
def date_related_features(bike_weather_data):
    bike_weather_data['start_hour'] = bike_weather_data['starttime'].dt.hour
    bike_weather_data['start_dayofweek'] = bike_weather_data['starttime'].dt.dayofweek
    bike_weather_data['start_day'] = bike_weather_data['starttime'].dt.day
    bike_weather_data['start_month'] = bike_weather_data['starttime'].dt.month
    bike_weather_data['start_is_weekend'] = bike_weather_data['start_dayofweek'].apply(lambda x: 1 if x >= 5 else 0)

    bike_weather_data['end_hour'] = pd.to_datetime(bike_weather_data['stoptime']).dt.hour
    bike_weather_data['end_dayofweek'] = pd.to_datetime(bike_weather_data['stoptime']).dt.dayofweek
    bike_weather_data['end_day'] = pd.to_datetime(bike_weather_data['stoptime']).dt.day
    bike_weather_data['end_month'] = pd.to_datetime(bike_weather_data['stoptime']).dt.month
    bike_weather_data['end_is_weekend'] = bike_weather_data['end_dayofweek'].apply(lambda x: 1 if x >= 5 else 0)

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
    feed = pt.load_feed("../data/gtfs")
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
    bike_gdf = gpd.GeoDataFrame(bike_weather_data, geometry='start_geometry', crs='EPSG:4326')
    
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


                        ### Add holidays to the data

def add_holidays_to_data(bike_weather_data, start_date="2023-01-01", end_date="2023-01-31"):

    # Generate US holidays for 2023
    us_holidays = holidays.US(years=2023)
    date_range = pd.date_range(start=start_date, end=end_date)

    # Create a DataFrame with dates and corresponding holiday names
    special_days = [us_holidays.get(date, None) for date in date_range]

    df_special_days = pd.DataFrame({
        'date': date_range.date,  # Normalize to ensure datetime64[ns] type
        'special_day': special_days
    })

    # Convert 'starttime' to date
    bike_weather_data['date'] = pd.to_datetime(bike_weather_data['starttime']).dt.date

    # Merge the dataframes
    bike_weather_data = bike_weather_data.merge(df_special_days, how='left', on='date')

    # Drop the 'date' column as it's no longer needed
    bike_weather_data.drop(columns=['date'], inplace=True)

    return bike_weather_data


                        ### Neighborhoods


def combine_and_save_neighborhoods(boston_neigh = '../data/neighborhood_data/bpda_neighborhood_boundaries.geojson', cambridge_neigh = '../data/BOUNDARY_CDDNeighborhoods/BOUNDARY_CDDNeighborhoods.shp'): 
    # Load Boston neighborhoods GeoJSON
    boston_neighborhoods = gpd.read_file(boston_neigh)

    # Load Cambridge neighborhoods shapefile
    cambridge_neighborhoods = gpd.read_file(cambridge_neigh)

    # Ensure both GeoDataFrames have the same coordinate reference system (CRS)
    if boston_neighborhoods.crs != cambridge_neighborhoods.crs:
        cambridge_neighborhoods = cambridge_neighborhoods.to_crs(boston_neighborhoods.crs)

    # Standardize the column name for Boston and Cambridge neighborhoods
    boston_neighborhoods = boston_neighborhoods.rename(columns={'name': 'neighborhood'})
    cambridge_neighborhoods = cambridge_neighborhoods.rename(columns={'NAME': 'neighborhood'})

    # Concatenate the GeoDataFrames after standardizing column names
    combined_neighborhoods = gpd.GeoDataFrame(pd.concat([boston_neighborhoods, cambridge_neighborhoods], ignore_index=True))

    # Save the combined data for future use
    combined_neighborhoods.to_file('../data/neighborhood_data/boston_cambridge_neighborhoods.geojson', driver='GeoJSON')

def classify_out_of_area(points, polygons, threshold_meters=2500):
    """
    Classify points as 'out_of_area_nearby' or 'out_of_area_far' based on the nearest polygon and a threshold.
    """
    results = []
    for point in points:
        if not isinstance(point, GeoPoint):  # Ensure valid geometry
            results.append(None)
            continue

        # Convert threshold from meters to degrees
        latitude = point.y
        degrees_lat, degrees_lon = meters_to_degrees(threshold_meters, latitude)

        # Calculate minimum distance to all polygons
        distances = [point.distance(poly) for poly in polygons]
        min_distance = min(distances)

        # Classify based on the threshold in degrees
        if min_distance < degrees_lat:  # Only using latitude for simplicity
            results.append("out_of_area_nearby")
        else:
            results.append("out_of_area_far")

    return results


def add_neighborhoods(bike_weather_data, neighborhoods, threshold_meters=2500):
    """
    Adds start and end neighborhood information to the bike_weather_data,
    and classifies out-of-area points as 'nearby' or 'far'.
    """
    # Prepare bike data as a GeoDataFrame
    bike_gdf = prepare_bike_data(bike_weather_data)

    # Ensure CRS compatibility between bike data and neighborhoods
    neighborhoods = neighborhoods.to_crs(bike_gdf.crs)

    # Spatial join for start neighborhoods
    bike_gdf = gpd.sjoin(bike_gdf, neighborhoods, how='left', predicate='intersects', rsuffix='_start')

    # Spatial join for end neighborhoods
    bike_gdf.set_geometry('end_geometry', inplace=True)
    bike_gdf = gpd.sjoin(bike_gdf, neighborhoods, how='left', predicate='intersects', rsuffix='_end')

    bike_gdf = bike_gdf.rename(columns={'neighborhood_left': 'start_neighborhood'})
    bike_gdf = bike_gdf.rename(columns={'neighborhood__end': 'end_neighborhood'})

    # Convert neighborhoods' geometry to a list for distance calculation
    polygons = neighborhoods.geometry.tolist()

    # Classify 'out_of_area' points for start_neighborhood
    missing_start = bike_gdf[bike_gdf['start_neighborhood'].isna()]
    bike_gdf.loc[missing_start.index, 'start_neighborhood'] = classify_out_of_area(
        missing_start['start_geometry'], polygons, threshold_meters
    )

    # Classify 'out_of_area' points for end_neighborhood
    missing_end = bike_gdf[bike_gdf['end_neighborhood'].isna()]
    bike_gdf.loc[missing_end.index, 'end_neighborhood'] = classify_out_of_area(
        missing_end['end_geometry'], polygons, threshold_meters
    )

    # Clean up: drop unnecessary column
    bike_gdf = bike_gdf.drop(columns=['geometry'], errors='ignore')

    return bike_gdf

def meters_to_degrees(threshold_meters, latitude):
    """
    Convert distance from meters to degrees at a given latitude.
    """
    # Convert threshold in meters to degrees for latitude
    degrees_lat = threshold_meters / 111320

    # Convert threshold in meters to degrees for longitude
    degrees_lon = threshold_meters / (111320 * np.cos(np.radians(latitude)))

    return degrees_lat, degrees_lon

def classify_route_area(row):
    """
    Classify a route as 'in_area', 'near_out_of_area', or 'far_out_of_area'
    based on start and end neighborhoods.
    """
    if 'out_of_area_far' in [row['start_neighborhood'], row['end_neighborhood']]:
        return 'far_out_of_area'
    elif 'out_of_area_nearby' in [row['start_neighborhood'], row['end_neighborhood']]:
        return 'near_out_of_area'
    else:
        return 'in_area'

def create_route_feature(bike_weather_data):
    """
    Create the 'route_area' feature and corresponding LineString geometries for routes.

    Parameters:
        bike_weather_data (pd.DataFrame): DataFrame containing bike trip data
            with 'start_geometry' and 'end_geometry' columns.

    Returns:
        gpd.GeoDataFrame: Updated GeoDataFrame with 'route_area' and 'route' features.
    """
    # Create LineString geometries for each route
    bike_weather_data['route'] = bike_weather_data.apply(
        lambda row: LineString([row['start_geometry'], row['end_geometry']]) 
        if row['start_geometry'] and row['end_geometry'] else None,
        axis=1
    )

    # Classify each route into 'in_area', 'near_out_of_area', or 'far_out_of_area'
    bike_weather_data['route_area'] = bike_weather_data.apply(classify_route_area, axis=1)

    return bike_weather_data

def encode_label(df, column):
    encoder = LabelEncoder()
    df[column] = encoder.fit_transform(df[column])
    return df
