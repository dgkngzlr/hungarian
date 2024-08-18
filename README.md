### Implementation of Hungarian Algorithm in Python 
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
