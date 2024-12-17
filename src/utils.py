
def calculate_special_day_accuracy(bike_weather_data, anomalies):

    total_special_count = bike_weather_data['special_day'].notna().sum()
    anomaly_special_count = anomalies['special_day'].notna().sum()

    if total_special_count == 0:
        return 0.0  # Return 0 accuracy if there are no special days in the data

    # Calculate accuracy as the percentage of special day trips identified as anomalies
    special_day_accuracy = (anomaly_special_count / total_special_count) * 100

    return special_day_accuracy