## Linear Circuit Solver

### Steps:
1. Start with representation of graph and nodes (matrix or adjacency list)
2. Solve for loops
    * Find traversal of all edges (Chinese Postman problem)
    * Split up into disjoint cycles
3. Set up KVL/KCL equations
4. Solve system of linear equations

### 1. 
Node objects for nodes (measure voltage)  
Edge objects (measure current):
* Resistor
* Source (needs direction)
* ReLu (needs direction)

#### Question: How do we encode direction?

### 2.
1. Solve Chinese Postman problem to get full tour
    1. Add proxy edges to ensure Euler tour exists (pair up odd degree vertices)
    2. Find Euler tour of new graph
2. Split up loops so that each has only one cycle
    1. Begin traversal somewhere at random, keeping track of seen vertices and seen edges in 2 separate stacks
    2. When we reach an already seen vertex: 
        1. Add cycle to our list of loops
        2. Proceed normally until all edges have been seen (no need to move anywhere else)
3. KVL/KCL Equations  
    KCL: 
    * For each node, sum of edges adjacent to node is zero   
    KVL:
    * For each edge:
        * If edge is a source, add voltage to voltage total (RHS vector) (direction matters!)
        * If edge is a resistor, set $i_n = \frac{v_{n-1} - v_n}{R_n}$
4. Solve system  
    `np.linalg.solve`

Continuous update rule: $\Delta R_{ij} = \frac{\gamma}{R^2_{ij}}((V^C_i - V^C_j)^2 - (V^F_i - V^F_j)^2)$  
Continuous update rule: $\Delta Y_{ij} = \gamma \cdot ((V^C_i - V^C_j)^2 - (V^F_i - V^F_j)^2)$

### Issues:
* How do we encode direction when solving for voltage sources?
* How do we deal with nonlinearities using this system?
* Does our matrix representation work or do we need to iteratively generate networks?

