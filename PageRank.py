'''
Page Rank example algorithm
'''

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import time
import random

data_path = './matrix.csv'
data = np.loadtxt(data_path, delimiter=',')
rrr = np.loadtxt(data_path,dtype = int, delimiter=',')
beta = 0.8

#initial r vector
r_list = [1/21] * 21
r_vector = np.asarray(r_list)

# get the dictionary that has the weights of each node
node_weight_dict ={}
counter = 0
temp_array = []
for val in range(21):
    node_weight_dict[val] = data[val].sum()

#node_weight_dict is correct


adj_matrix = np.asmatrix(rrr)
#print(adj_matrix)
G = nx.from_numpy_matrix(adj_matrix, create_using=nx.DiGraph)
nx.draw(G, with_labels=True)
plt.show()

matrix_size = 21
weighted_matrix = np.zeros(shape=(matrix_size,matrix_size))



#creates M matrix
for row in range(21):
    counter = 0
    for col in range(21):
        #check if the value in that position is a 1
        if adj_matrix[row,col] == 1:
            #if the value in the dictionary is a 0 we dont deivide, we replace value with 0
            if node_weight_dict[col] == 0:
                weighted_matrix[col, row] = 0
            else:
                #if it is then we need to change the value in the w_matrix but inverted
                weighted_matrix[col, row] = round((1/ node_weight_dict[row]),2)


def Power_Iteration(r, given_matrix):
    given_matrix = np.asarray(given_matrix)
    #time.sleep(3)
    r_vector = np.dot(given_matrix, r)
    #print("R_Vector Iteration: " + str(iter + 1))
    print(r_vector)
    print("Sum of Vector: " + str(r_vector.sum()))
    print("Module of Vector: " + str(np.linalg.norm(r_vector)))
    print("------------------------------------------------------------------")

def Random_Walker(prev_node, graph_node):
    print("Current Node: " + str(graph_node))
    print("Neighbors " )
    print([n for n in G.neighbors(graph_node)])
    #gets neighbors
    successor_list = [n for n in G.neighbors(graph_node)]
    #chooses random node from neighbors
    #validate two cases:
    #first case: dead end
    if (len(successor_list) == 0):
        Jumper(graph_node, "DE")
    #second case: random chance/spider trap
    else:
        beta_random = random.uniform(0, 1)
        next_value = random.choice(successor_list)
        print("Beta Probability: " + str(beta_random))
        if (beta_random <= beta):
            Power_Iteration(r_vector, weighted_matrix)
            time.sleep(3)
            Random_Walker(graph_node, next_value)
        else:
            Jumper(None, "ST")


def Jumper(node, case):
    if case == "DE":
        print("Dead End - Transport")
        random_list = [i for i in range(20)]
        starting_value = random.choice(random_list)
        for row in range(21):
            weighted_matrix[row,node] = (1/21)
        print(weighted_matrix)
        Power_Iteration(r_vector, weighted_matrix)
        time.sleep(3)
        Random_Walker(None, starting_value)

    elif case == "ST":
        print("Spider Trap/Random - Transport")
        random_list = [i for i in range(20)]
        starting_value = random.choice(random_list)
        Power_Iteration(r_vector, weighted_matrix)
        time.sleep(3)
        Random_Walker(None, starting_value)



#picks random number in range 1:20 and calls Random_Walker()
print(r_vector)
print("Sum of Vector: " + str(r_vector.sum()))
print("Module of Vector: " + str(np.linalg.norm(r_vector)))

random_list = [i for i in range(20)]
starting_value = random.choice(random_list)
Random_Walker(None, starting_value)
