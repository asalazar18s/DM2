'''
Page Rank example algorithm
'''

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

data_path = './matrix.csv'
data = np.loadtxt(data_path, dtype=int, delimiter=',')

Adj_matrix = np.asmatrix(d1)

G = nx.from_numpy_matrix(data, create_using=nx.DiGraph)

nx.draw(G, with_labels=True)
plt.show()



























