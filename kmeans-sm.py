# Name:         Sharice Mayer
# Due Date:     June 13, 2020
# Course:       CS445 ML - Spring 2020
# Assignment:   Program #3 Part 1
# Topic:        K-Means

import numpy as np
import pandas as pd
import math
import statistics
import time
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Cluster and Classify cluster_dataset.txt
# using K-Means clustering algorithm

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
#     Iterate next steps until no change to centroids
#       Compute Weights - Compute the sum of the squared distance between data points and all centroids.
#       Assign each data point to the closest cluster (centroid).
#       Update Centroids for the clusters by taking the average of the all data points that belong to each cluster.
#     Stop iterating K-Means when all cluster centers stop changing
# - Choose the run(from the r number of runs) that yields the smallest sum-of-squares-error
#     For this run in report give sum-of-squares-error,
#     and include 2-d plot of data points and clusters
#     for the resulting clustering on the data
# - Repeat for several values of k


# my kmeans algorithm
    # for each run
    # initialize a list of centroids
    # make a copy of given dataset
    # perform k-means clustering
    # calculate Sum of Squared Errors
    # return the results


# print final clusters
def plot_clusters(clusters, sse, runs):
    # make it pretty
    num_clusters = len(clusters)
    intsse = int(sse)
    colors = cm.gist_rainbow(np.linspace(0, 1, len(clusters)))
    for index, cluster in clusters:
        # clusters are different colors
        plt.scatter(cluster['xcoord'], cluster['ycoord'], s=num_clusters, color=colors[index])

    plt.title('K-means Clustering with K = ' +str(len(clusters))+' runs = ' + str(runs)+' (SSE = '+str(sse)+')')
    plt.xlabel('attribute 1')
    plt.ylabel('attribute 2')
    plt.savefig('Kmeans_' + str(len(clusters)) + '_best_r' + str(runs) + '_sse' + str(intsse) + '.png')


def plot_run(clusters, sse, run, runs):
    # make it pretty
    num_clusters = len(clusters)
    intsse = int(sse)
    colors = cm.gist_rainbow(np.linspace(0, 1, len(clusters)))
    for index, cluster in clusters:
        # clusters are different colors
        plt.scatter(cluster['xcoord'], cluster['ycoord'], s=num_clusters, color=colors[index])

    plt.title('K-means Clustering with K = ' + str(len(clusters)) + ' at run ' + str(run) +' of '+ str(runs)+ ' (SSE = '+str(sse)+')')
    plt.xlabel('attribute 1')
    plt.ylabel('attribute 2')
    plt.savefig('Kmeans_' + str(len(clusters)) + '_r' + str(run) + 'of' + str(runs) + '_sse' + str(intsse) + '.png')


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
    # k = int(input("Please input number of k clusters: "))
    # print(f"k is set to value: {k}")
    # k = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]  # to run with several k values in succession
    k = [3]

    # Request r number of runs to run each set
    r = int(input("Please input number of r runs: "))
    print(f"r is set to value: {r}")

    # Read in data set as pd dataframe
    data_reference = pd.read_csv("data/cluster_dataset.csv", delim_whitespace=True, names=['xcoord','ycoord'])
    # print(f"\ndata_reference type: {type(data_reference)} values = \n {data_reference}\n")

    # end timer for processing
    curr = time.time()
    get_time(start, curr, "Pre-process data", -1)

    print("---------------------------------------------------------------")

    # For each k number of clusters:
    for num_clusters in k:  # when k = [3, 5] instead of an integer
    # if k > 0:
        print(f"\n\tK Clusters = {num_clusters} \n")
        lsse = 999999999.999999999
        sse_runs = list(range(r))
        # best_clusters = list(range(k))
        best_clusters = list(range(num_clusters))
        # run r times
        for run in range(r):
            # start a timer for the run
            run_start = time.time()

            # add a column with just points
            data_reference['(xcoord,ycoord)'] = data_reference.apply(
                lambda point: np.array((point['xcoord'], point['ycoord'])), axis=1)

            # Add a column with a cluster label, initializing to 0
            data_reference['Cluster'] = 0

            # convert points into np array for calculations
            data_points = np.array(data_reference['(xcoord,ycoord)'])

            # make a copy of the data set for run
            data_values = data_reference.copy()  # do I need this?

            # run k-means algorithm
            # select k random centroids(centers) -- centroids represent the mean after initial rand selection
            # initial_centroids = data_points[np.random.choice(data_points.shape[0], k, replace=False)]
            initial_centroids = data_points[np.random.choice(data_points.shape[0], num_clusters, replace=False)]

            centroids = initial_centroids
            new_centroids = centroids
            # While centroids are changing
            change = True
            while change:
                # Form K clusters by assigning each point to its closest centroid
                data_values['NewCluster'] = data_values['(xcoord,ycoord)'].apply(lambda p: nearest_centroid(centroids, p))

                # set new clusters by group
                new_clusters = data_values.groupby('NewCluster')

                # calculate new cluster centroids
                for index, cluster in new_clusters:
                    # calculate the new centroid of each cluster closest to the mean
                    cluster_points = np.array(cluster['(xcoord,ycoord)'])
                    cluster_mean = np.array([cluster['xcoord'].mean(), cluster['ycoord'].mean()])
                    new_centroids[index] = cluster_points[nearest_point(cluster_points, cluster_mean)]

                # print current and new centroids
                # print(f"\nnew_centroids: {new_centroids}\n")
                # print(f"\ncentroids: {centroids}\n")

                # check to see if cluster points have changed
                if data_values['Cluster'].equals(data_values['NewCluster']):
                    # print("\n\tEqual! Woot. \n")  # done!
                    final_clusters = data_values['Cluster']
                    change = False
                else:
                    # print("\n\tNot equal. Keep calculating.\n")
                    # update cluster column
                    data_values['Cluster'] = data_values['NewCluster']
                    # update centroids
                    centroids = new_centroids

            # calculate sse for run
            sse = 0
            final_clusters = data_values.groupby('NewCluster')
            for index, cluster in final_clusters:
                cluster_points = np.array(cluster['(xcoord,ycoord)'])
                centroid = centroids[index]
                for point in cluster_points:
                    sse += np.sum(np.sum((point - centroid)**2))
            # print sse
            # print(f"\nRun's final sse: {sse}\n")

            # sse append
            if sse < lsse:
                lsse = sse
                best_clusters = final_clusters
            sse_runs[run] = sse

            # print this run
            plot_run(final_clusters, sse, run+1, r)

            # calculate and print run time
            curr = time.time()
            get_time(run_start, curr, "run ", (run+1))

        # print lowest sse of runs
        print(f"\nsse all {r} runs = {sse_runs}\n")
        print(f"\nlowest sse of all {r} runs = {lsse}\n")

        # plot the best run for this k
        plot_clusters(best_clusters, lsse, r)

        print("---------------------------------------------------------------")

    # print total program time
    end = time.time()
    get_time(start, end, "\nTotal program", -1)

    print("---------------------------------------------------------------")

    print("")
