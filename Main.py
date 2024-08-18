import time
import numpy as np

from Hungarian import HungarianAssignment

def main():

    hungarian_assignment = HungarianAssignment()

    # Cost: 95
    cost_matrix1 = np.array([[57, 41, 78],
                            [17, 17, 97],
                            [70, 44, 37]])
    

    # Cost: 19
    cost_matrix2 = np.array([[4,  2, 5,  7],
                            [8,  3, 10, 8],
                            [12, 5, 4,  5],
                            [6,  3, 7,  14]])
    
    # Cost: 71
    cost_matrix3 = np.array([[21, 8, 61, 93, 2],
                            [51, 45, 82, 7, 37],
                            [33, 68, 9, 89, 76],
                            [42, 26, 21, 30, 54],
                            [98, 74, 74, 52, 8]])
    
    # Cost: 143
    cost_matrix4 = np.array([[20, 51, 49,43,82,80,66,36,49,54],
                            [21,87,14,24,37,74,75,26,46,38],
                            [56,35,65,27,80,30,15,19,18,1],
                            [91,94,67,71,68,3,28,37,35,71],
                            [58,25,8,40,55,55,15,27,19,41],
                            [8,	97,92,35,82,46,36,28,59,38],
                            [53,59,54,63,52,94,83,5,68,31],
                            [74,46,59,37,17,21,67,10,63,89],
                            [70,32,37,39,98,98,48,59,6,10],
                            [71,88,17,57,30,50,77,97,60,7]])
    
    # Cost: 5
    cost_matrix5 = np.array([[4, 1, 3],
                             [2, 0, 5],
                             [3, 2, 2]])

    for s in range(2, 33):

        begin_time = time.time()
        cost_matrix = np.random.randint(0, 99, size=(s, s))

        cost, assignment = hungarian_assignment.solve(cost_matrix)
        delta_time = time.time() - begin_time

        print("Size:", s, "x", s)
        print("Cost:", cost)
        print("Assignment:", assignment)
        print("Delta Time:", delta_time)
        print(50 * "=")


if __name__ == "__main__":

    main()