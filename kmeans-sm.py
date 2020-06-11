# Name:         Sharice Mayer
# Due Date:     June 13, 2020
# Course:       CS445 ML - Spring 2020
# Assignment:   Program #3
# Topic:        K-Means and Fuzzy C-Means

import numpy as np
import pandas as pd
import math
import statistics
import time

# Cluster and Classify cluster_dataset.txt
# using K-Means clustering algorithm, then Fuzzy C-Means

# Data set:
# The data set is 2d data (for ease of visualization)
# simulated from 3 Gaussians,
# with considerable overlap. 
# There are 500 points from each Gaussian,
# ordered together in the file.
# *** The dataset is included in this repository.

# Program implements K-Means clustering using Euclidian distance,
# and evaluates resulting clustering using:
# sum-of-squares-error

# Part 1 - K-Means
# Repeat r times with different random number seeds
# - Run clustering on training data with k = {2,3,5,7,9,10,15}
#     Obtain k final cluster centers
#     Initial cluster centers chosen at random from training examples
#     Compute Weights
#     Update Centroids
#     Stop iterating K-Means when all cluster centers stop changing
#     ** some empty clusters are ok?
# - Choose the run(from the r number of runs) that yields the smallest sum-of-squares-error
#     For this run in report give sum-of-squares-error,
#     and include 2-d plot of data points and clusters
#     for the resulting clustering on the data
#     ** Leave out empty clusters in calculating these metrics?
# - Repeat for several values of k

# Part 2 - Fuzzy C-Means
# Repeat r times with different random number seeds
# - Run clustering on data with k = {2,3,5,7,9,10,15}
#     Obtain k final cluster centers:
#     Initial cluster centers chosen at random from training examples
#     Compute Weights
#     Update Centroids
#     Stop iterating Fuzzy C-Means when all cluster centers stop changing
# - Select the solution giving smallest sum-of-squares error over r runs
#     For this run in report give sum-of-squares-error,
#     and include 2-d plot of data points and clusters
#     for the resulting clustering on the data
# Again run for several values of k

# Include in report:
#     Short Description of experiments
#     Experimental Results Summary
#     2-d Plots
# Send in report and readable code

# 1. Specify number of clusters K.
# 2. Initialize centroids by first shuffling the dataset and then randomly selecting K data points for the centroids
# without replacement.
# 3. Keep iterating until there is no change to the centroids. i.e assignment of data points to clusters isn't changing.
#        Compute the sum of the squared distance between data points and all centroids.
#        Assign each data point to the closest cluster (centroid).
#        Compute the centroids for the clusters by taking the average of the all data points that belong to each cluster.


# my kmeans algorithm
    # for each run
    
    # initialize a list of centroids

    # make a copy of given dataset as stated above

    # perform k-means clustering

    # calculate Sum of Squared Errors

    # return the results


# return the closest point to the mean given
def nearest_point(points, mean):
    distances = []
    for point in points:
        # calculate the euclidian distance between given point and centroid
        distances.append(np.sum((mean - point)**2))
    return np.argmin(distances)


# return the closest centroid index
def nearest_centroid(centroids, point):
    distances = []
    for centroid in centroids:
        # calculate the euclidian distance between given point and centroid
        distances.append(np.sum((point - centroid)**2))
    return np.argmin(distances)


# given a start time, print block time information
def get_time(start_time, end_time, name, optional_num):
    current_time = time.time()
    elapsed = end_time - start_time
    if elapsed == 0:
        elapsed = current_time - start_time
    n_sec = (elapsed % 60)
    n_min = elapsed / 60
    if optional_num < 0:
        print("%s run time: %d minutes, %d seconds\n" % (name, n_min, n_sec))
    else:
        print("%s %d \nRun time: %d minutes, %d seconds\n" % (name, optional_num, n_min, n_sec))


# main program starts here
if __name__ == "__main__":

    # Start a timer for program
    start = time.time()

    # Print Header information
    template = "{0:14}{1:20}"
    print("")
    print(template.format("Course:", "CS445-ML"))
    print(template.format("Date:", "06/13/20"))
    print(template.format("Name:", "Sharice Mayer"))
    print(template.format("Assignment:", "Program 3"))
    print(template.format("Topic:", "k-means and fuzzy c-means clustering"))
    print("")

    print("----------------------Pre-processing data----------------------")

    print("")

    # print(f" type: {type()} values = \n {}\n")

    # Request k number of clusters
    k = int(input("Please input number of k clusters: "))
    print(f"k is set to value: {k}")
    # k = [2, 3, 5, 7, 9, 10, 15]  # to run with several k values in succession

    # Request r number of runs to run each set
    r = int(input("Please input number of r runs: "))
    print(f"r is set to value: {r}")

    # Read in data set as pd dataframe
    # c_data = pd.read_csv('data/cluster_dataset.csv', header=None, delim_whitespace=True, na_filter=False).values ##RL
    # print(f"c_data type: {type(c_data)} values = \n {c_data}")
    data_reference = pd.read_csv("data/cluster_dataset.csv", delim_whitespace=True, names=['xcoord','ycoord'])
    # print(f"\ndata_reference type: {type(data_reference)} values = \n {data_reference}\n")
    '''
    data_reference type: <class 'pandas.core.frame.DataFrame'> values = 
             xcoord    ycoord
    0    -0.169513 -0.243970
    1    -1.462618 -1.333294
    2     0.769671  0.849244
    ...        ...       ...
    1497  2.114334  1.031347
    1498  2.061401 -0.067838
    1499  1.885700  1.003853
    
    [1500 rows x 2 columns]
    '''

    # end timer for processing
    curr = time.time()
    get_time(start, curr, "Pre-process data", -1)

    print("---------------------------------------------------------------")


    # For each k number of clusters:
    # for num_clusters in k:  # when k = [3, 5] instead of an integer
    if k > 0:
        # print(f"\nFor num_clusters {num_clusters}\n")
        lsse = 999999999.999999999
        sse_runs = list(range(r))
        # run r times
        for run in range(r):
            # start a timer for the run
            run_start = time.time()

            # add a column with just points
            data_reference['(xcoord,ycoord)'] = data_reference.apply(
                lambda point: np.array((point['xcoord'], point['ycoord'])), axis=1)
            # print(f"data_reference type: {type(data_reference)} values = \n {data_reference}\n")

            # Add a column with a cluster label, initializing to 0
            data_reference['Cluster'] = 0
            # print(f"data_reference type: {type(data_reference)} values = \n {data_reference}\n")

            # convert points into np array for calculations
            data_points = np.array(data_reference['(xcoord,ycoord)'])
            # print(f"data_points type: {type(data_points)} values = \n {data_points}\n")
            # print(f"data_points[0] type: {type(data_points[0])} values = \n {data_points[0]}\n")
            '''
            data_points type: <class 'numpy.ndarray'> values = 
             [array([-0.169513, -0.24397 ]) array([-1.462618, -1.333294])
             array([0.769671, 0.849244]) ... array([2.114334, 1.031347])
             array([ 2.061401, -0.067838]) array([1.8857  , 1.003853])]
            data_points[0] type: <class 'numpy.ndarray'> values = 
             [-0.169513 -0.24397 ]
            '''

            # make a copy of the data set for run
            data_values = data_reference.copy()  # do I need this?

            # run k-means algorithm

            # select k random centroids(centers) -- centroids represent the mean after initial rand selection
            # rand_indices = np.random.choice(c_data.shape[0], size=k)
            # centroids =  c_data[rand_indices, :]
            # centroids = c_data[np.random.choice(c_data.shape[0], k, replace=False), :]
            # print(f"centroids type: {type(centroids)} centroids value: \n{centroids}")
            # centroids.append(data_points[np.random.randint(0, data_points.shape[0])]) # picks a single rand point

            # select k random centroids(centers) -- centroids represent the mean after initial rand selection
            initial_centroids = data_points[np.random.choice(data_points.shape[0], k, replace=False)]
            print(f"initial_centroids type: {type(initial_centroids)} centroids value: \n{initial_centroids}")

            centroids = initial_centroids
            new_centroids = centroids
            # print(f"data_points shape: {data_points.shape}")  # (1500,)
            # While centroids are changing
            change = True
            while change:
            # for i in range(3):  #  for testing...
                # print("\nNEW LOOP\n")

                # Form K clusters by assigning each point to its closest centroid
                data_values['NewCluster'] = data_values['(xcoord,ycoord)'].apply(lambda p: nearest_centroid(centroids, p))

                # set new clusters by group
                new_clusters = data_values.groupby('NewCluster')
                # print(f"new clusters: {new_clusters}\n")
                # new clusters: <pandas.core.groupby.generic.DataFrameGroupBy object at 0x10fa86e50>

                # calculate new cluster centroids
                for index, cluster in new_clusters:
                    # print(f"cluster: \n {cluster}\n")
                    cluster_points = np.array(cluster['(xcoord,ycoord)'])
                    # print(f"cluster_points dimension: {cluster_points.shape}\n {cluster_points}\n")
                    # calculate the new centroid of each cluster closest to the mean
                    cluster_mean = np.array([cluster['xcoord'].mean(), cluster['ycoord'].mean()])
                    new_centroids[index] = cluster_points[nearest_point(cluster_points, cluster_mean)]
                    # print(f"new_centroids[index]: {new_centroids[index]}\n")

                # print current and new centroids
                # print(f"\nnew_centroids: {new_centroids}\n")
                # print(f"\ncentroids: {centroids}\n")

                # check to see if cluster points have changed
                if data_values['Cluster'].equals(data_values['NewCluster']):
                    print("\n\tEqual! Woot. \n")  # done!
                    final_clusters = data_values['Cluster']
                    change = False
                else:
                    print("\n\tNot equal. Keep calculating.\n")
                    # update cluster column
                    data_values['Cluster'] = data_values['NewCluster']
                    # update centroids
                    centroids = new_centroids

            sse = 0
            final_clusters = data_values.groupby('NewCluster')
            for index, cluster in final_clusters:
                cluster_points = np.array(cluster['(xcoord,ycoord)'])
                centroid = centroids[index]
                for point in cluster_points:
                    sse += np.sum(np.sum((point - centroid)**2))
            # prrnt sse
            # print(f"\nRun's final sse: {sse}\n")

            # sse append
            if sse < lsse:
                lsse = sse
            sse_runs[run] = sse

            # calculate and print run time
            curr = time.time()
            get_time(run_start, curr, "r-run ", (run+1))

        # print lowest sse of runs
        print(f"\nsse all runs = {sse_runs}\n")   # for testing
        print(f"\nlowest sse of all runs = {lsse}\n")

        print("---------------------------------------------------------------")

    # print total program time
    end = time.time()
    get_time(start, end, "\nTotal program", -1)

    print("---------------------------------------------------------------")



# STUFF TO USE FOR TESTING/PRINTS/ETC


# random_indices = np.random.choice(num_rows, k, replace=False)  # from numrows choose 3rowsof5(k=[3,5])
# print(f"random indices {random_indices}")

# num_rows = c_data.shape[0]
# print(f"num_rows {num_rows}")
# random_indices = np.random.choice(number_of_rows, size=2, replace=False)
# random_rows = an_array[random_indices, :]

# print(f"num_rows: {type(num_rows)} {len(num_rows)}")
# centroids = np.random.choice(c_data, num_clusters, replace=False)
# print(f"centroids chosen: {centroids}")
# print(f"row 5: {c_data[5]}")
# print(f"c_data: {type(c_data)} {c_data.shape}")
# print(f"{num_rows}, {num_cols}")
# clusters = []  # Initialize seeds randomly
# clusters = np.random.uniform(low=-0.05, high=0.05, size=(r, c))
# np.random.choice(number_of_rows, size=2, replace=False)
# end timer for r run

# num_rows, num_cols = c_data.shape
            # random_indices = np.random.choice(num_rows, size=num_clusters, replace=False)
            # print(f"indices chosen: {random_indices}")
            # centroids = c_data[random_indices:]
            # print(f"centroids chosen: {centroids}")

# ------------------------------------
# select k random centroids(centers)
# centroids represent the mean after initial rand selection
# centroids = c_data[np.random.choice(c_data.shape[0], num_clusters, replace=False), :]
# centroids = c_data[np.random.choice(c_data.shape[0], k, replace=False), :]
# print(f"centroids chosen: \n{centroids}")

# test if this does the same thing -- yes, it does
# random_indices = np.random.choice(c_data.shape[0], size=k)
# print(f"random indices chosen: \n{random_indices}")
# alt_cent =  c_data[random_indices, :]
# print(f"alt_centroids chosen: \n{alt_cent}")
# ------------------------------------
