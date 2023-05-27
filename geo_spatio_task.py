# import dependencies
import warnings, math, logging, os, sys
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import dask.dataframe as dd

# ignore warnings
warnings.filterwarnings("ignore")

def most_frequent_locations(df):
    """
    Identifies and logs the top 5 locations with the highest frequency of occurrences.

    Args:
    - df: Pandas DataFrame containing latitude and longitude columns.
    """

    # Group the data by latitude and longitude and count the occurrences
    location_counts = df.groupby(["LATITUDE", "LONGITUDE"]).size().reset_index(name="Frequency")

    # Sort the locations by frequency in descending order
    sorted_locations = location_counts.sort_values(by="Frequency", ascending=False)

    # Print the top 5 locations with the highest frequency
    most_frequent_locations = sorted_locations.head()
    logging.info("Top 5 locations with the highest frequency:")
    logging.info(most_frequent_locations)

def nearly_located_locations_clusters(df):
    """
    Performs clustering on latitude and longitude using the K-means algorithm and logs the clusters.

    Args:
    - df: Pandas DataFrame containing latitude and longitude columns.
    """

    # Perform clustering on latitude and longitude using K-means algorithm
    kmeans = KMeans(n_clusters=10)
    kmeans.fit(coordinates)

    # Add the cluster labels to the DataFrame
    df["Cluster"] = kmeans.labels_

    # Print the clusters and their corresponding lat-lon pairs
    clusters = df.groupby("Cluster")[["LATITUDE", "LONGITUDE"]].apply(lambda x: list(x.values))
    logging.info("Clusters of lat-lon pairs:")
    for cluster, locations in clusters.items():
        logging.info(f"Cluster {cluster + 1}:")
        for location in locations:
            logging.info(location)
        logging.info("\n")

def density_plot_lat_long(df):
    """
    Generates and displays a density plot of latitude and longitude pairs.

    Args:
    - df: Pandas DataFrame containing latitude and longitude columns.
    """

    # Extract longitude and latitude columns
    longitude = df['LONGITUDE']
    latitude = df['LATITUDE']

    # Set the number of bins for the histogram
    bins = 5

    # Create 2D histogram using numpy
    H, xedges, yedges = np.histogram2d(longitude, latitude, bins=bins)

    # Set the extent of the plot
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

    # Plot the density using matplotlib
    plt.imshow(H.T, extent=extent, origin='lower', cmap='viridis')
    plt.colorbar(label='Density')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Density Plot')
    plt.show()

def density_plot_few_records():
    """
    Generates and displays a density plot of a subset of latitude and longitude pairs.

    This function assumes the `coordinates` variable is defined and contains a subset of latitude and longitude pairs.
    """

    # Create a scatter plot with density estimation
    sns.kdeplot(data=coordinates[:1000], x='LONGITUDE', y='LATITUDE', fill=True, cmap='viridis')

    # Set plot title and axis labels
    plt.title('Density Plot of Latitude-Longitude Pairs')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')

    # Display the plot
    plt.show()

def check_ip_correlations(partition):
    """
    Checks for IP correlations within a partition of data and extracts unique locations.

    Args:
    - partition: Pandas DataFrame partition containing IP, latitude, and longitude columns.

    Returns:
    - Pandas DataFrame with unique latitude and longitude values for each IP.
    """

    return partition.groupby("IP").apply(lambda x: x[["LATITUDE", "LONGITUDE"]].drop_duplicates()).reset_index(drop=True)

def get_correlated_ips_location(df):
    """
    Identifies and logs IP addresses that are correlated to multiple locations.

    Args:
    - df: Pandas DataFrame containing IP, latitude, and longitude columns.
    """

    # Convert the pandas DataFrame to a dask DataFrame for parallel computing due to larger dataset
    ddf = dd.from_pandas(df, npartitions=10)

    # Use dask's parallel computing with apply
    meta = pd.DataFrame(columns=["LATITUDE", "LONGITUDE"], dtype=float)
    results = ddf.groupby("IP").apply(check_ip_correlations, meta=meta).compute()

    # Filter out partitions with only one unique location
    correlated_ips = results[results.groupby("IP")["LATITUDE"].transform("nunique") > 1]

    # Print the correlated IP addresses and their associated locations
    if not correlated_ips.empty:
        logging.info("IP addresses correlated to multiple locations:")
        for ip, group in correlated_ips.groupby("IP"):
            logging.info(group[["LATITUDE", "LONGITUDE"]])
            logging.info("\n")
    else:
        logging.info("No IP addresses correlated to multiple locations.")

def degrees_to_radians(degrees):
    """
    Converts degrees to radians.

    Args:
    - degrees: Numeric value representing an angle in degrees.

    Returns:
    - Equivalent angle in radians.
    """

    return degrees * math.pi / 180

def is_within_time_threshold(time1, time2):
    """
    Checks if two datetime values are within a specified time threshold.

    Args:
    - time1: First datetime value.
    - time2: Second datetime value.

    Returns:
    - Boolean value indicating whether the time difference is within the threshold.
    """
    
    time_difference = abs((time1 - time2).total_seconds() / 60)  # Calculate time difference in minutes
    return time_difference <= time_threshold_minutes

def clustered_distance_time_entities(df):
    """
    Performs clustering based on distance and time thresholds and logs the clusters.

    Args:
    - df: Pandas DataFrame containing latitude, longitude, and date_time columns.
    """

    # Convert the date_time column to datetime format
    df['date_time'] = pd.to_datetime(df['date_time'])

    # Prepare data for clustering
    data = df[['LATITUDE', 'LONGITUDE']].values

    # Convert latitude and longitude to radians for faster processing
    data[:, 0] = degrees_to_radians(data[:, 0])
    data[:, 1] = degrees_to_radians(data[:, 1])

    # Perform clustering using DBSCAN
    dbscan = DBSCAN(eps=distance_threshold_km / 6371, min_samples=2, metric='haversine', algorithm='ball_tree')
    labels = dbscan.fit_predict(data)

    # Create a dictionary to store the clusters
    clusters = {}

    # Assign entities to clusters
    for i, entity_id in enumerate(df['ID']):
        cluster_id = labels[i]
        if cluster_id != -1:  # Ignore outliers
            if cluster_id in clusters:
                clusters[cluster_id].append(entity_id)
            else:
                clusters[cluster_id] = [entity_id]

    # Print the clusters and entities in each cluster
    logging.info("Clusters of entities:")
    for cluster_id, entities in clusters.items():
        logging.info(f"Cluster {cluster_id + 1}:")
        logging.info(entities)

def main_caller():
    """
    Main calling function that executes a series of functions.
    """

    # write output to log file and not to terminal
    std_file = open(os.devnull, 'w')

    try:
        # Redirect sys.stdout and sys.stderr
        sys.stdout = std_file
        sys.stderr = std_file

        # Execute the functions
        most_frequent_locations(df)
        nearly_located_locations_clusters(df)
        density_plot_lat_long(df)
        density_plot_few_records()
        get_correlated_ips_location(df)
        clustered_distance_time_entities(df)

    finally:
        # Restore the original sys.stdout and sys.stderr
        sys.stdout = sys.stdout
        sys.stderr = sys.stderr
        std_file.close()


if __name__ == "__main__":

    # set logging configuration
    logging.basicConfig(filename="log_file.log", format='%(message)s', level=logging.INFO)

    logging.info("Reading excel file")
    # Read the Excel file into a pandas DataFrame
    df = pd.read_excel("Data.xlsx")

    # Replace "-" with 0 in the longitude and latitude columns to make them numeric values and strip for removing whitespaces
    df["LONGITUDE"] = [str(elem).strip().replace("-", "0") for elem in df["LONGITUDE"]]
    df["LATITUDE"] = [str(elem).strip().replace("-", "0") for elem in df["LATITUDE"]]

    # convert column type to float
    df["LATITUDE"] = df["LATITUDE"].astype(float)
    df["LONGITUDE"] = df["LONGITUDE"].astype(float)

    # dataframe of latitude and longitude
    coordinates = df[["LATITUDE", "LONGITUDE"]]
    
    # Define distance threshold of 3 kilometers and time threshold of 30 minutes
    distance_threshold_km = 3
    time_threshold_minutes = 30
    
    # execute script
    main_caller()
    

    
