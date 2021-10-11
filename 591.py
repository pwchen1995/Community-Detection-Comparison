#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 14:36:34 2021

@author: kevinchen
"""

import timeit
import igraph as ig
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
psql = lambda q: sqldf(q, globals())

from networkx.generators.community import LFR_benchmark_graph
import networkx as nx
from community import community_louvain
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler
    
def read_file():
    
    Amazon_0302 = ig.Graph.Read_Ncol('/Users/kevinchen/Desktop/WSU_Spring2021/Element of network sicence/Project/amazon0302.txt', directed = False)
    Amazon_0312 = ig.Graph.Read_Ncol('/Users/kevinchen/Desktop/WSU_Spring2021/Element of network sicence/Project/amazon0312.txt', directed = False)
    Amazon_0505 = ig.Graph.Read_Ncol('/Users/kevinchen/Desktop/WSU_Spring2021/Element of network sicence/Project/amazon0505.txt', directed = False)
    Amazon_0601 = ig.Graph.Read_Ncol('/Users/kevinchen/Desktop/WSU_Spring2021/Element of network sicence/Project/amazon0601.txt', directed = False)
    
    return Amazon_0302, Amazon_0312, Amazon_0505, Amazon_0601

def plot_runtime(tag, time, name): #time, #nodes, #edges, #diameter
   
   plt.plot(tag,time, label = name)
   #plt.plot(tag,nodes, label = "Nodes differential")
   #plt.plot(tag,edges, label = "Edges differential")
   plt.xlabel('Date of Amazon co-purchase dataset')
   plt.xticks(rotation = 45)
   plt.ylabel('Run time')
   plt.title('Run time for {}'.format(name))
   plt.legend()
   plt.show()
    
def plot_modularity_score(tag, score, name): #score, #nodes, #edges, #diameter

   # Score compare 
   plt.plot(tag,score, label = name)
   #plt.plot(tag,nodes, label = "# of nodes")
   #plt.plot(tag,edges, label = "# of edges")

   plt.xlabel('Date of Amazon co-purchase dataset')
   plt.xticks(rotation = 45)
   plt.ylabel('Modularity Score')
   plt.title('Modularity Score for ()'.format(name))
   plt.legend()
   plt.show()

def plot_node_edge(tag, nodes, edges):
    
   plt.plot(tag,nodes, label = "# of nodes")
   #plt.plot(tag,edges, label = "# of edges")

   plt.xlabel('Date of Amazon co-purchase dataset')
   plt.xticks(rotation = 45)
   plt.ylabel('# of Nodes')
   plt.title('# of Nodes correlated to date')
   plt.legend()
   plt.show()
   
   plt.plot(tag,edges, label = "# of edges")
   plt.xlabel('Date of Amazon co-purchase dataset')
   plt.xticks(rotation = 45)
   plt.ylabel('# of Edges')
   plt.title('# of Edges correlated to date')
   plt.legend()
   plt.show()
   
def plot_compare_score(tag, Louvain, Info, Propa, Leiden, Newman):
   
    plt.plot(tag, Louvain, label = "Louvain")
    plt.plot(tag, Info, label = "Infomap")
    plt.plot(tag, Propa, label = "Label Propagation")
    plt.plot(tag, Leiden, label = "Leiden")
    plt.plot(tag, Newman, label = "Newman")
    
    plt.xlabel('Date of co-purchase dataset')
    plt.xticks(rotation = 45)
    plt.ylabel('Score')
    plt.title('Comparisonof 5 algorithm')
    plt.legend()
    plt.show()

if __name__ == "__main__":
   
   Amazon_0302, Amazon_0312, Amazon_0505, Amazon_0601 = read_file()
   
   # data gather
   Amazon = [Amazon_0302,Amazon_0312,Amazon_0505,Amazon_0601]
   Nodes = [262111,400727,410236,403394]
   Edges = [1234877,3200440,3356824,3387388]
   Diameter = [32,18,20,21]
   DataTag = ["Amazon co-purchase 0302", "Amazon co-purchase 0312", "Amazon co-purchase 0505", "Amazon co-purchase 0601"]
   Date = ["03/02","03/12","05/05","06/01"]
   
   # run time array
   Multi_runtime = []
   Info_runtime = []
   Propa_runtime = []
   Leiden_runtime = []
   Newman_runtime = []
   
   # score array
   Multi_score = []
   Info_score = []
   Propa_score = []
   Leiden_score = []
   Newman_score = []
   
   print(type(Amazon_0312))
   i = 0
   for item in Amazon:
       network = item
       
       print("------------------------------{}------------------------------".format(DataTag[i]))
       
       # multilevel Louvain method (DONE)
       Louvain_start = timeit.default_timer()
       network_cluster = network.community_multilevel()
       network_multilevel = network.modularity(network_cluster)
       Louvain_stop = timeit.default_timer()
       runtime_Louvain = Louvain_stop - Louvain_start
       
       print("{} Multilevel: {}, Run time: {}" .format(DataTag[i], network_multilevel, runtime_Louvain))
       Multi_runtime.append(runtime_Louvain)
       Multi_score.append(network_multilevel)
       
       # infomap (DONE)
       Infomap_start = timeit.default_timer()
       network_cluster = network.community_infomap()
       network_infomap = network.modularity(network_cluster)
       Infomap_stop = timeit.default_timer()
       runtime_Info = Infomap_stop - Infomap_start
       
       print("{} Infomap: {}, Run time: {}" .format(DataTag[i], network_infomap, runtime_Info))
       Info_runtime.append(runtime_Info)
       Info_score.append(network_infomap)
       
       # propagation
       Prop_start = timeit.default_timer()
       network_cluster = network.community_label_propagation()
       network_propagation = network.modularity(network_cluster)
       Prop_stop = timeit.default_timer()
       runtime_Prop = Prop_stop - Prop_start
       
       print("{} Propagation: {}, Run time: {}" .format(DataTag[i], network_propagation, runtime_Prop))
       Propa_runtime.append(runtime_Prop)
       Propa_score.append(network_propagation)
       
       # Leiden
       Leiden_start = timeit.default_timer()
       network_cluster = network.community_leiden()
       network_leiden = network.modularity(network_cluster)
       Leiden_stop = timeit.default_timer()
       runtime_Leiden = Leiden_stop - Leiden_start
       
       print("{} Leiden: {}, Run time: {}" .format(DataTag[i], network_leiden, runtime_Leiden))
       Leiden_runtime.append(runtime_Leiden)
       Leiden_score.append(network_leiden)
       
       # Newman's Leaning eigenvector
       Newman_start = timeit.default_timer()
       network_cluster = network.community_leading_eigenvector()
       network_newman = network.modularity(network_cluster)
       Newman_stop = timeit.default_timer()
       runtime_Newman = Newman_stop - Newman_start
       
       print("{} Newman: {}, Run time: {}" .format(DataTag[i], network_newman, runtime_Newman))
       Newman_runtime.append(runtime_Newman)
       Newman_score.append(network_newman)
       
       print("")
       i += 1    
   # normalize run time
   #Multi_runtime = normailzed(Multi_runtime)
   #Info_runtime = normailzed(Info_runtime)
   #Propa_runtime = normailzed(Propa_runtime)
   #Leiden_runtime = normailzed(Leiden_runtime)
   #Newman_runtime = normailzed(Newman_runtime)

   # normalize nodes and edges
   #Nodes = normalized(Nodes)
   #Edges = normalized(Edges)
    
   plot_runtime(Date, Multi_runtime, "Louvain")
   plot_runtime(Date, Info_runtime, "Infomap")
   plot_runtime(Date, Propa_runtime, "Label Propagation")
   plot_runtime(Date, Leiden_runtime, "Leiden")
   plot_runtime(Date, Newman_runtime, "Newman")
   
   plot_modularity_score(Date, Multi_score, "Louvain")
   plot_modularity_score(Date, Info_score, "Infomap")
   plot_modularity_score(Date, Propa_score, "Label Propagation")
   plot_modularity_score(Date, Leiden_score, "Leiden")
   plot_modularity_score(Date, Newman_score, "Newman")
   
   plot_node_edge(Date, Nodes, Edges)
   
   plot_compare_score(Date, Multi_score, Info_score, Propa_score, Leiden_score, Newman_score)
   