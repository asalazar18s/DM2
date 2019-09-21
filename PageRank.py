'''
Page Rank example algorithm
'''

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

data_path = './matrix.csv'

data = np.loadtxt(data_path, dtype=int, delimiter=',')
d1 = np.genfromtxt(data_path, dtype=int, delimiter=',', names=True)

d = np.asmatrix(data)

print(d1)

def show_graph_with_labels(adjacency_matrix):
    '''
    Shows graph
    '''
    rows, cols = np.where(adjacency_matrix == 1)
    edges = zip(rows.tolist(), cols.tolist())
    gr = nx.DiGraph()
    gr.add_edges_from(edges)
    nx.draw(gr, node_size=100, with_labels=False)
    plt.show()

show_graph_with_labels(data)




#G = nx.DiGraph()

