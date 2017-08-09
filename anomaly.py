################################################################################
# anomaly.py
# Author: Preston Scott pdscott2
# CSC 591
# Implementation of NetSimile anomaly detection algorithm.
# https://cs.ucsb.edu/~victor/pub/ucsb/mae/references/berlingerio-netsimile-2012.pdf
################################################################################

import networkx as nx
import statistics
import scipy.stats
import scipy.spatial.distance
import matplotlib.pyplot as plt
import os
import sys


# Given a NeworkX graph, determines the degree of each node 
# graph: NetworkX graph
# returns a dictionary of node:degree pairs
def get_degree(graph):
   deg_dict = {node: graph.degree(node) for node in graph.nodes()} 
   return deg_dict


# Given a NeworkX graph, determines the clustering coefficientof each node 
# graph: NetworkX graph
# Returns a dictionary of node:clustering coefficient pairs
def get_cc(graph):
    cc_dict  ={node: nx.clustering(graph,node) for node in graph.nodes()}
    return cc_dict

# Given a NeworkX graph, determines the average number of two hop neighbors (thn)
# of each node 
# graph: NetworkX graph
# Returns a dictionary of node:thn pairs
def get_thn(graph):
    thn_dict = {}
    for node in graph.nodes():
        neighbors = nx.all_neighbors(graph, node)
        mean_neighbor_degree = statistics.mean([nx.degree(graph, neighbor) for neighbor in neighbors])
        thn_dict[node] = mean_neighbor_degree
    return thn_dict

# Given a NeworkX graph, determines the average clustering coefficient of each 
# node's neighbors (ncc)
# graph: NetworkX graph
# Returns a dictionary of node:ncc pairs
def get_ncc(graph):
    ncc_dict = {}
    for node in graph.nodes():
        neighbors = nx.all_neighbors(graph, node)
        ncc = statistics.mean([nx.clustering(graph, neighbor) for neighbor in neighbors])
        ncc_dict[node] = ncc
    return ncc_dict

# Given a NeworkX graph, determines the number of edges in each node's egonet
# of each node (ee)
# graph: NetworkX graph
# Returns a dictionary of node:ncc pairs
def get_ee(graph):
    ee_dict = {node: len(nx.ego_graph(graph, node).edges()) for node in graph.nodes()}
    return ee_dict

# Given a NeworkX graph, determines the number of outgoing edges extending from the 
# egonet of each node (eeo)
# graph: NetworkX graph
# Returns a dictionary of node:eeo pairs
def get_eeo(graph):
    eeo_dict = {}
    for node in graph.nodes():
        src_nodes = set(nx.all_neighbors(graph, node))
        src_nodes.add(node)
        num_out_edges = 0
        for src_node in src_nodes:
            dst_nodes = set(nx.all_neighbors(graph,src_node))
            new_out_edges = dst_nodes.difference(src_nodes)
            num_out_edges += len(new_out_edges)
        eeo_dict[node] = num_out_edges
    return eeo_dict

# Given a NeworkX graph, determines the number of neighbors of each node's egonet (egn)
# graph: NetworkX graph
# Returns a dictionary of node:egn pairs
def get_egn(graph):
    egn_dict = {}
    for node in graph.nodes():
        egonet = set(nx.all_neighbors(graph, node))
        egonet.add(node)
        egonet_neighbors = set()
        for ego_node in egonet:
            neighbors = set(nx.all_neighbors(graph,ego_node))
            egonet_neighbors = egonet_neighbors.union(neighbors)
        filtered_neighbors = egonet_neighbors.difference(egonet)
        egn_dict[node] = len(filtered_neighbors)

    return egn_dict

# Given a NeworkX graph, performs the feature aggregation described in the NetSimile 
# algorithm
# graph: NetworkX graph
# Returns a feature signature vector for the graph
def aggregate_features(graph):
    f1 = get_degree(graph)
    f2 = get_cc(graph)
    f3 = get_thn(graph)   
    f4 = get_ncc(graph)   
    f5 = get_ee(graph)   
    f6 = get_eeo(graph)   
    f7 = get_egn(graph)   
    feature_matrix = [[f1[u], f2[u], f3[u], f4[u], f5[u], f6[u], f7[u]] for u in graph.nodes()]
    signature = []
    for feature in range(0, len(feature_matrix[0])):
        mean = statistics.mean([feature_matrix[row][feature] for row in range(0,len(feature_matrix))])
        median = statistics.median([feature_matrix[row][feature] for row in range(0,len(feature_matrix))])
        stdev = statistics.stdev([feature_matrix[row][feature] for row in range(0,len(feature_matrix))])
        skewness = scipy.stats.skew([feature_matrix[row][feature] for row in range(0,len(feature_matrix))])
        kurtosis = scipy.stats.kurtosis([feature_matrix[row][feature] for row in range(0,len(feature_matrix))])
        feature_set = [mean,median,stdev,skewness,kurtosis]
        signature += feature_set
    return signature

# Given a time series of distances between adjacent veectors, returns the calculated
# threshold for identifying out of family measurements,
# distances: a list of distances between adjacent vectors
# Returns a tuple of lower and upper thresholds
def get_thresholds(distances):
    moving_range = 0
    for i in range(0, len(distances) - 1):
        moving_range += abs(distances[i] - distances[i+1])
    moving_range_avg = moving_range / (len(distances) - 1)
    median = statistics.median(distances)
    return (median - (3 * moving_range_avg), median + (3 * moving_range_avg))

# Given the relative path to a directory of edgelists, reads each edgelist one-
# by-one, and builds a NetworkX graph.  
# input_directory: relative path to directory of graph files
# Returns a list of graphs sorted by name index
def read_files(input_directory):
    graphs = {}
    sorted_graphs = []
    listing = os.listdir(input_directory)
    file_count = len(listing)
    file_index = 1
    for file in listing:
        path = input_directory + file
        id = int(file.split('_')[0])
        graphs[id] = read_graph(path)
        show_progress('Reading graphs: ',file_index, file_count)
        file_index += 1
    print ''
    sorted_keys = sorted(graphs)
    for key in sorted_keys:
        sorted_graphs.append(graphs[key])
    return sorted_graphs

# Given a file path, reads the file and returns a NetworkX graph 
# infile: full path to an edgelist
# Note: This method ignores the first line as the node/edge count
# is not used for NetworkX
# Returns a NetworkX graph
def read_graph(infile):
    file = open(infile, 'r')
    
    # throw away first line
    file.readline()

    graph=nx.parse_edgelist(file, nodetype=int)
    return graph

# Given a time series of distances between adjacent signature vectors,
# a threshold, writes a graph image to the specified imagefile.
# distances: a list of distances between adjacent vectors
# UCL: upper threshold for determining out of family values
# imagefile: output file
def draw_plot(distances, UCL, imagefile):
    fig = plt.figure()
    plt.plot(distances, '-+')
    plt.axhline(y=UCL,c='g')
    plt.title('Similarity (Canberra Distance) vs Time for P2P-Gnutella Data')
    plt.xlabel('Time')
    plt.ylabel("Similarity")
    #plt.show()
    plt.savefig(imagefile)

# Given a time series of distances between adjacent signature vectors,
# and an output data file, writes a summary to the datafile.
# distances: a list of distances between adjacent vectors
# datafile: output data file
def write_data(distances, datafile):
    file = open(datafile,'w')
    for index, distance in enumerate(distances):
        file.write('%s %s\n' % (index, distance))
    file.close

# Given the current progress of an incrementer, prints a percentage
# progress to stdout.
# state: string representing current operation
# index: current index of incrementer
# count: end index of incrementer
def show_progress(state,index, count):
    percentage = float(index) / float(count) * 100
    sys.stdout.write('\r%s %d%% Complete' % (state, percentage))
    sys.stdout.flush()

# Starts the program and runs the NetSimile algorithm
# USAGE: python anomaly.py [relative path to input directory] [data outputfile] [image outputfile]
def main():

    if not len(sys.argv) == 4:
        print "USAGE: python anomaly.py [relative path to input directory] [data outputfile] [image outputfile]"
        sys.exit()

    input_directory = sys.argv[1]
    data_outfile = sys.argv[2]
    image_outfile = sys.argv[3]
    graphs = []
    signatures = []
    distances = []
    anomalies = []
    
    graphs = read_files(input_directory)

    for index, graph in enumerate(graphs):
        signature = aggregate_features(graph)
        signatures.append(signature)
        show_progress('Aggregating features: ',index + 1, len(graphs))

    for i in range(0, len(signatures) - 1):
        distance = abs(scipy.spatial.distance.canberra(signatures[i], signatures[i+1]))
        distances.append(distance)
    LCL, UCL = get_thresholds(distances)

    for i in range(0, len(distances) - 1):
        if distances[i] >= UCL and distances[i+1] >= UCL:
            anomalies.append(i+1)

    sys.stdout.write('\nAnomalies based on threshold of %s:\n' % UCL)
    for anomaly in anomalies:
        print anomaly
    if len(anomalies) == 0:
        print 'No anomalies detected.'

    write_data(distances, data_outfile)

    draw_plot(distances, UCL, image_outfile)

if __name__ == "__main__":
    main()


