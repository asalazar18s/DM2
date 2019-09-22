'''
Page Rank example algorithm
'''

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import time

data_path = './matrix.csv'
data = np.loadtxt(data_path, delimiter=',')
rrr = np.loadtxt(data_path,dtype = int, delimiter=',')

# get the dictionary that has the weights of each node
node_weight_dict ={}
counter = 0
temp_array = []
for val in range(21):
    node_weight_dict[val] = data[val].sum()

#node_weight_dict is correct


adj_matrix = np.asmatrix(rrr)
#print(adj_matrix)
#G = nx.from_numpy_matrix(adj_matrix, create_using=nx.DiGraph)
#nx.draw(G, with_labels=True)
#plt.show()

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



#print(weighted_matrix)



def Power_Iteration(given_matrix):
    given_matrix = np.asarray(given_matrix)
    r_list = [1/21] * 21
    print("R_Vector Iteration: " + str(0))
    r_vector = np.asarray(r_list)
    print(r_vector)
    print("Sum of Vector: " +str(r_vector.sum()))
    print("Module of Vector: " +str(np.linalg.norm(r_vector)))
    #time.sleep(3)
    print("------------------------------------------------------------------")
    for iter in range(21):
        r_vector = np.dot(given_matrix, r_vector)
        print("R_Vector Iteration: " + str(iter + 1))
        print(r_vector)
        print("Sum of Vector: " + str(r_vector.sum()))
        print("Module of Vector: " + str(np.linalg.norm(r_vector)))
        print("------------------------------------------------------------------")
        #time.sleep(3)
    high_page = np.amax(r_vector)
    index = np.argmax(r_vector)
    return index,high_page

x,y = Power_Iteration(weighted_matrix)
print("Max Page Rank: " + str(x))
print("Value: " + str(y))

#Figure out the connection between power iteration vs random walker
#do we have to run power iteration until all nodes visited???
# I think we do its the same calculations 






