# Sharice Mayer
# Date: March 10, 2019
# CS445 - ML
# Program 5
# K-means clustering

import numpy as np
import pandas as pd
import math
import statistics

# Cluster and classify Optdigits data
#  originally from UCI ML repository
#  using K-means clustering clustering algorithm

# each instance has 64 attributes, each of which with values 0-16
# each instance also has a label specifying which of 10 digit classes it belongs to


# Program implements k-means clustering using Euclidian distance, and
# evaluates resulting clustering using 
#   average mean-square-error
#   mean-square-separation
#   mean entropy
#   accuracy


# Experiment 1
# Repeat 5 times with different random number seeds
# - Run clustering on training data with K = 10 
#     Obtain 10 final cluster centers
#     ** remove class attribute before clustering
#     Initial cluster centers chosen at random from training examples
#     Stop iterating K-Means when all cluster centers stop changing
#     ** some empty clusters are ok
# - Choose the run(from the 5) that yields the smallest average mean-square-error
#     For this run in report give average mean-square-error,
#       mean-square-separation, and mean entropy(using the class labels)
#       of the resulting clustering on the training data
#     ** Leave out empty clusters in calculating these metrics.
# - Now use this clustering to classify the test data as follows:
#     Associate each non-empty cluster center with the most frequent class it contains
#       in the taining data. 
#     ** Break ties at random
#     Assign each test instance the class of the closest cluster center center
#     ** Break ties at random
#     ** It's possible a particular class won't be the most common one for any cluster,
#          therefore no test digit will ever get that label.
# - Calculate the accuracy on the test data and create a confusion matrix
#     for the results on the test data
# - Visualize the resulting non-empty cluster centers.
#     For each of the 10 cluster centers, use the cluster center's attributes
#       to draw the corresponding digit on an 8x8 grid.
#       - Each value in the cluster center's feature vector is interpreted
#           as a grayscale value for it's associated pixel.
#         ** Use any matrix-to-gray-scale format
#         ** ex: http://en.wikipedia.org/wiki/Netpbm_format#PGM_example
#         ** Visualization must be large enough to see the pixels of the cluster center
# - In report, include the following:
#     Average mean-square-error, mean-squareseparation, & mean entropy of resulting
#       clustering on the training data in the best run of the five
#     Classification accuracy on the test data and the confusion matrix
#       over the test data(SAME RUN)
#     Visualization results(SAME RUN)
#     Discussion paragraph
#       Summarize your results
#       Answer: Do the visualized cluster centers look like their associated digits?


# Experiment 2
# Repeat 5 times with different random number seeds
# - Run K-means clustering on same training data but with K = 30
#     Initial cluster centers chosen at random from training examples
# - In report, include same things as experiment 1
#   ** For discussion, compare Exp1 results to Exp2 results
#        ** include comparison for various metrics(mse,mss,mean entropy,accuracy)


# ////////
if __name__ == "__main__":

    # Read in traning and test sets
    train_data = pd.read_csv("data/optdigits.train", header=None).values
    test_data = pd.read_csv("data/optdigits.test", header=None).values
    print("\n")

    # for testing ////
    print("train_data: \n",train_data)
    print("\n")
    print("test_data: \n",test_data)
    print("\n")
    #

