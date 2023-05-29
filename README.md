# geo_spatial

This Python code performs various operations on a dataset containing latitude and longitude information. The code includes functions to identify the most frequent locations, perform clustering on latitude and longitude, generate density plots and perform clustering based on distance and time thresholds. The code also includes a main function that executes a series of functions.

## Dependencies

The code requires the following dependencies to be installed which are provided in the requirements.txt file:

- `pandas`
- `numpy`
- `seaborn`
- `sklearn`
- `matplotlib`
- `dask`
- `openpyxl`

You can install the dependencies using the following command but it is recommended to create a virtual environment beforehand:

```
pip install -r requirements.txt
```

## Functions

The code includes several functions that perform specific tasks. Here's an overview of each function:

### 1. `most_frequent_locations(df)`

This function identifies and logs the top 5 locations with the highest frequency of occurrences in the given DataFrame.

### 2. `nearly_located_locations_clusters(df)`

This function performs clustering on latitude and longitude using the K-means algorithm and logs the clusters.

### 3. `density_plot_lat_long(df)`

This function generates and displays a density plot of latitude and longitude pairs.

### 4. `density_plot_few_records()`

This function generates and displays a density plot of a subset of latitude and longitude pairs.

### 5. `check_ip_correlations(partition)`

This function checks for IP correlations within a partition of data and extracts unique locations. It returns a DataFrame with unique latitude and longitude values for each IP.

### 6. `get_correlated_ips_location(df)`

This function identifies and logs IP addresses that are correlated to multiple locations.

### 7. `degrees_to_radians(degrees)`

This function converts degrees to radians.

### 8. `is_within_time_threshold(time1, time2)`

This function checks if two datetime values are within a specified time threshold.

### 9. `clustered_distance_time_entities(df)`

This function performs clustering based on distance and time thresholds and logs the clusters.

### 10. `main_caller()`

This is the main calling function that executes a series of functions.

## Execution

The code is executed in the `__main__` block. It reads an Excel file ("Data.xlsx") into a pandas DataFrame, performs data preprocessing, sets the logging configuration, and calls the `main_caller()` function to execute the series of functions.

## Usage

To use this code, follow these steps:

1. Install the required dependencies using the command mentioned above.
2. Save the code in a Python file (e.g., `script.py`) and make sure to have the input data file ("Data.xlsx") in the same directory.
3. Run the Python script using the following command:

```
python script.py
```

4. The code will execute the functions and log the output to the "log_file.log" file.
