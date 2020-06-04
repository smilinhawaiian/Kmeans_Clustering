# Sharice Mayer
# June 3, 2020
# CS445 - ML - Spring 2020
# Program #3
# K-Means and Fuzzy C-Means

import numpy as np
import pandas as pd
import math
#import statistics
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


# given a start time, print block time information
def get_time(start_time, end_time, name, optional_num):
    current_time = time.time()
    elapsed = end_time - start_time
    if elapsed == 0:
        elapsed = current_time - start_time
    n_sec = (elapsed % 60)
    n_min = elapsed / 60
    if optional_num < 0:
        print("%s run time: %d minutes, %d seconds" % (name, n_min, n_sec))
    else:
        print("%s %d \nRun time: %d minutes, %d seconds" % (name, optional_num, n_min, n_sec))


# main program starts here
if __name__ == "__main__":

    # Start a timer for program
    start = time.time()

    # Print Header information
    template = "{0:14}{1:20}"
    print("")
    print(template.format("Course:", "CS445-ML"))
    print(template.format("Date:", "06/03/20"))
    print(template.format("Name:", "Sharice Mayer"))
    print(template.format("Assignment:", "Program 3"))
    print(template.format("Topic:", "k-means and fuzzy c-means clustering"))
    print("")

    print("----------------------Pre-processing data----------------------")

    # Read in data set
    c_data = pd.read_csv('data/cluster_dataset.csv', header=None).values

    # Request r number of runs to run each set
    r = 3  # edit to make sure we request this
    print(f"r is set to default: {r}")
    r = int(input("Please input number of r runs: "))
    print(f"r is set to input value: {r}")

    # Do any other processing necessary here
    # k = [2, 3, 5, 7, 9, 10, 15]
    k = [3, 5]  # try with just a couple k values to make sure code works

    # end timer for processing
    curr = time.time()
    get_time(start, curr, "Pre-process data", -1)

    print("---------------------------------------------------------------")

    # start a timer for the run
    run_start = time.time()

    # For each k number of clusters:
    for num_clusters in k:
        print("---------------------------------------------------------------")
        print(f"\nFor num_clusters {num_clusters}\n")
        for run in range(r):
            num_rows, num_cols = c_data.shape
            # print(f"row 5: {c_data[5]}")
            print(f"c_data: {type(c_data)} {c_data.shape}")
            # print(f"{num_rows}, {num_cols}")
            # clusters = []  # Initialize seeds randomly
            # clusters = np.random.uniform(low=-0.05, high=0.05, size=(r, c))
            # np.random.choice(number_of_rows, size=2, replace=False)
            # end timer for r run
            curr = time.time()
            get_time(run_start, curr, "r-run ", run)

    print("---------------------------------------------------------------")


    # print total program time
    end = time.time()
    get_time(start, end, "\nTotal program", -1)
    print("")

    print("---------------------------------------------------------------")

