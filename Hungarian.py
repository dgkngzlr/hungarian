import numpy as np

class HungarianAssignment:
    """
        A class that implements the Hungarian algorithm (also known as the Kuhn-Munkres algorithm)
        for solving the assignment problem. The assignment problem seeks to minimize the total cost
        of assigning `n` tasks to `n` agents such that each task is assigned to exactly one agent,
        and each agent is assigned exactly one task.

        The algorithm is applied to a square cost matrix, where each element represents the cost
        of assigning a particular agent to a particular task. The goal is to find the optimal assignment
        that minimizes the total cost.If the matrix is not square (i.e., the number of rows and columns
        are different), you can make it square by adding "dummy" rows or columns. These dummy
        rows or columns should be filled with large values (or zeros) to avoid influencing the
        optimal assignment but allow the algorithm to proceed.After solving the square matrix,
        the assignments corresponding to the dummy rows or columns should be ignored.

        Attributes:
        -----------
        _size : int
            The size of the cost matrix (number of rows/columns).
        _cost : int
            The total minimum cost of the optimal assignment.
        _assignments : dict
            A dictionary that maps each row (agent) to a column (task) in the optimal assignment.
        _reduced_cost_matrix : np.ndarray
            A copy of the original cost matrix that has been modified during the algorithm's execution
            (rows and columns reduced, and uncovereds/crosses adjusted).

        Methods:
        --------
        _get_uncovered_presented_matrix(horz_line_idxs, vert_line_idxs)
            Generates a boolean matrix indicating the uncovered elements in the cost matrix.
        
        _get_cross_presented_matrix(horz_line_idxs, vert_line_idxs)
            Generates a boolean matrix indicating the cross-covered elements in the cost matrix.
        
        _get_zero_presented_matrix(cost_matrix)
            Generates a matrix indicating the positions of zeros in the cost matrix.
        
        _reduct_rows(cost_matrix)
            Reduces each row of the cost matrix by subtracting the minimum element in that row.
        
        _reduct_cols(cost_matrix)
            Reduces each column of the cost matrix by subtracting the minimum element in that column.
        
        _reduct_uncovereds(cost_matrix, horz_line_idxs, vert_line_idxs)
            Adjusts the cost matrix by subtracting the minimum uncovered value and adding it to the cross-covered positions.
        
        _get_required_lines(zero_presented_matrix)
            Determines the minimum number of lines required to cover all zeros in the matrix and returns the line indices.
        
        _assign_id(cost_matrix)
            Assigns tasks to agents based on the reduced cost matrix, ensuring that the total assignment cost is minimized.
        
        solve(cost_matrix)
            Solves the assignment problem for a given cost matrix and returns the total minimum cost and the optimal assignments.

        Example Usage:
        --------------
        >>> cost_matrix = np.array([[4, 1, 3],
                                    [2, 0, 5],
                                    [3, 2, 2]])

        >>> hungarian = HungarianAssignment()
        >>> cost, assignments = hungarian.solve(cost_matrix)
        >>> print(f"Total minimum cost: {cost}")
        >>> print(f"Optimal assignments: {assignments}")
    """

    def __init__(self) -> None:
        
        self._size = 0
        self._cost = 0
        self._assingments = {}
        self._reduced_cost_matrix = None
    
    def _get_uncovered_presented_matrix(self, horz_line_idxs, vert_line_idxs):
        
        uncovered_represented_matrix = np.zeros((self._size, self._size), dtype=np.bool)

        for i in range(uncovered_represented_matrix.shape[0]):
            for j in range(uncovered_represented_matrix.shape[1]):
                
                if i not in horz_line_idxs and j not in vert_line_idxs:
                    uncovered_represented_matrix[i ,j] = 1
        
        return uncovered_represented_matrix

    def _get_cross_presented_matrix(self, horz_line_idxs, vert_line_idxs):
        
        cross_represented_matrix = np.zeros((self._size, self._size), dtype=np.bool)

        # Cartesian product of line indexes
        cross_pos = []
        for horz_line_idx in horz_line_idxs:
            for vert_line_idx in vert_line_idxs:
                cross_pos.append((horz_line_idx, vert_line_idx))
        
        for pos in cross_pos:
            cross_represented_matrix[pos[0], pos[1]] = 1
        
        return cross_represented_matrix
    
    def _get_zero_presented_matrix(self, cost_matrix):
        return np.where(cost_matrix > 0, 0, 1)
    
    def _reduct_rows(self, cost_matrix):
        
        # Column vector which indicates min of every row
        row_mins = np.min(cost_matrix, axis = 1).reshape(-1, 1)

        # Reduct rows
        cost_matrix = cost_matrix - row_mins

        return cost_matrix

    def _reduct_cols(self, cost_matrix):
        
        # Row vector which indicates min of every col
        col_mins = np.min(cost_matrix, axis = 0).reshape(1, -1)

        # Reduct rows
        cost_matrix = cost_matrix - col_mins

        return cost_matrix

    def _reduct_uncovereds(self, cost_matrix, horz_line_idxs, vert_line_idxs):
        
        uncovered_presented_matrix = self._get_uncovered_presented_matrix(horz_line_idxs, vert_line_idxs)
        cross_presented_matrix = self._get_cross_presented_matrix(horz_line_idxs, vert_line_idxs)

        min_of_uncovereds = np.min(cost_matrix[uncovered_presented_matrix])

        cost_matrix[uncovered_presented_matrix] -= min_of_uncovereds
        cost_matrix[cross_presented_matrix] += min_of_uncovereds

        return cost_matrix
    
    def _get_required_lines(self, zero_presented_matrix):
        
        horz_line_idxs = []
        vert_line_idxs = []

        # Loop until zero_presented_matrix is becomes zero matrix
        while np.any(zero_presented_matrix):

            # Count the number of ones in each row 
            num_ones_row = np.sum(zero_presented_matrix == 1, axis=1)

            # Count the number of ones in each col 
            num_ones_col = np.sum(zero_presented_matrix == 1, axis=0)

            row_idx = int(np.argmax(num_ones_row))
            col_idx = int(np.argmax(num_ones_col))

            if num_ones_row[row_idx] >= num_ones_col[col_idx]:
                zero_presented_matrix[row_idx, :] = 0
                horz_line_idxs.append(row_idx)
            
            else:
                zero_presented_matrix[:, col_idx] = 0
                vert_line_idxs.append(col_idx)
        
        return (len(horz_line_idxs) + len(vert_line_idxs),
                horz_line_idxs,
                vert_line_idxs)
    
    def _assign_id(self, cost_matrix):

        assignment = {}
        unassigned_rows = [i for i in range(self._size)]

        for iter in range(self._size):
            
            row_idx = unassigned_rows[0]
            min_val = np.count_nonzero(cost_matrix[row_idx, :] == 0)
            for r in unassigned_rows:
                
                if np.count_nonzero(cost_matrix[r, :] == 0) < min_val:
                    row_idx = r
                    min_val = np.count_nonzero(cost_matrix[r, :] == 0)
                   
            col_idx = int(np.argmin(cost_matrix[row_idx,:]))
            cost_matrix[:, col_idx] = np.iinfo(np.int32).max
            assignment[row_idx] = col_idx

            unassigned_rows.remove(row_idx)

        return assignment



    def solve(self, cost_matrix):

        # Clear
        self._cost = 0
        self._assingments.clear()

        # Check is it 2D matrix
        assert(len(cost_matrix.shape) == 2)

        # Check is it square matrix
        assert(cost_matrix.shape[0] == cost_matrix.shape[1])

        self._reduced_cost_matrix = cost_matrix.copy()
        self._size = cost_matrix.shape[0]

        self._reduced_cost_matrix = self._reduct_rows(self._reduced_cost_matrix)
        self._reduced_cost_matrix = self._reduct_cols(self._reduced_cost_matrix)

        zero_presented_matrix = self._get_zero_presented_matrix(self._reduced_cost_matrix)
        num_lines, horz_line_idxs, vert_line_idxs = self._get_required_lines(zero_presented_matrix)

        # Loop until reduction complete
        while num_lines < self._size:
            self._reduced_cost_matrix = self._reduct_uncovereds(self._reduced_cost_matrix, horz_line_idxs, vert_line_idxs)

            zero_presented_matrix = self._get_zero_presented_matrix(self._reduced_cost_matrix)
            num_lines, horz_line_idxs, vert_line_idxs = self._get_required_lines(zero_presented_matrix)

        self._assingments = self._assign_id(self._reduced_cost_matrix)

        for key in self._assingments.keys():
            self._cost += int(cost_matrix[key, self._assingments[key]])

        return self._cost, self._assingments
