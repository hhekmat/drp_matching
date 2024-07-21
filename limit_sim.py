
import numpy as np
import networkx as nx
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

# generate bipartite graph with 2n nodes, where each edge has a weight sampled from U(0,1)
def create_bipartite_graph(n):
    X = [f'X{i}' for i in range(n)]
    Y = [f'Y{i}' for i in range(n)]
    
    B = nx.Graph()
    B.add_nodes_from(X, bipartite=0)
    B.add_nodes_from(Y, bipartite=1)
    
    weights = np.random.uniform(0, 1, size=(n, n))
    for i, x in enumerate(X):
        for j, y in enumerate(Y):
            B.add_edge(x, y, weight=weights[i, j])
    
    return B, X, Y

# compute and return max weight matching as list of tuples
def max_weight_matching(B):
    matching = nx.max_weight_matching(B, maxcardinality=True, weight='weight')
    return matching

# determine C(n), or how many students in max weight matching aren't matched with most preferred mentor
def count_non_optimal_pairs(B, X, matching):
    non_optimal_count = 0
    for x in X:
        for u, v in matching:
            if u == x:
                matched_node = v
            elif v == x:
                matched_node = u
        if matched_node is not None:
            matched_weight = B[x][matched_node]['weight']
            for neighbor in B.neighbors(x):
                if B[x][neighbor]['weight'] > matched_weight:
                    non_optimal_count += 1
                    break
    return non_optimal_count

# wrapper to call all the above functions and return C(n)
def run_single_iteration(n):
    B, X, Y = create_bipartite_graph(n)
    matching = max_weight_matching(B)
    non_optimal_count = count_non_optimal_pairs(B, X, matching)
    return non_optimal_count

# loop through range of values for n, repeating above process for 10000 iterations per value of n
# compute, store, and print average values for C(n)/n 
def main():
    cn_dict = {}
    start_n = 1
    max_n = 50
    iterations = 10000
    n_step_size = 1

    for n in range(start_n, max_n + 1, n_step_size): 
        print('Now on n = ' + str(n))
        avg = 0

        start_time = time.time() 

        # compute C(n) asynchronously for computational efficiency
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(run_single_iteration, n) for _ in range(iterations)]
            for i, future in enumerate(as_completed(futures)):
                if i % 1000 == 0:
                    print(f'Iteration: {i}')
                
                non_optimal_count = future.result()
                avg += non_optimal_count
        
        end_time = time.time()

        avg_non_optimal = avg / iterations
        cn_dict[n] = avg_non_optimal
        
        total_time = end_time - start_time
        print(f'Average non-optimal pairs for n={n}: {avg_non_optimal}')
        print(f'Estimated C(n)/n for n={n}: {avg_non_optimal / n}')
        print(f'Total time for n={n}: {total_time:.4f} seconds')

    final_ratio_dict = {key: cn_dict[key] / key for key in cn_dict}
    print('Final Ratio Dictionary:', final_ratio_dict)
    print('Non-Optimal Count Dictionary:', cn_dict)

if __name__ == "__main__":
    main()