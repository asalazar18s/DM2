'''
Page Rank example algorithm
'''

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

data_path = './matrix.csv'
data = np.loadtxt(data_path, delimiter=',')

# get the dictionary that has the weights of each node
node_weight_dict ={}
counter = 0
temp_array = []
for val in range(21):
    node_weight_dict[val] = data[val].sum()


adj_matrix = np.asmatrix(data)
weighted_matrix = np.asmatrix(data)

#creates M matrix
for row in range(21):
    counter = 0
    for col in range(21):
        if weighted_matrix[row, counter] == 1:
            if node_weight_dict[col] == 0:
                weighted_matrix[row, col] = 0
            else:
                weighted_matrix[row, col] = (1/ node_weight_dict[col])

        counter += 1


def Power_Iteration(given_matrix):
    r_list = [1/21] * 21
    r_vector = np.asarray(r_list)

    for iteration in range(1):
        print(np.matmul(weighted_matrix, r_vector))


Power_Iteration(weighted_matrix)
print(weighted_matrix)



#G = nx.from_numpy_matrix(data, create_using=nx.DiGraph)

#nx.draw(G, with_labels=True)
#plt.show()

