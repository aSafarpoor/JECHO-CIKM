
import random
import numpy as np
from scipy import sparse
from tqdm import tqdm

def ECD(G, opinions, alpha=0.0, max_iterations=50):
    n = len(G.vs)
    
    edges = G.get_edgelist()
    rows, cols, data = [], [], []
    
    for (i, j) in edges:
        edge_weight = opinions[i] * opinions[j]  
        rows.extend([i, j])
        cols.extend([j, i]) 
        data.extend([edge_weight, edge_weight])
    
    W_sparse = sparse.csr_matrix((data, (rows, cols)), shape=(n, n))
    A_sparse = W_sparse.multiply(W_sparse > 0)
    D_sparse = W_sparse.multiply(W_sparse < 0)
    
    A_indices = [A_sparse[i].indices for i in range(n)]
    A_data = [A_sparse[i].data for i in range(n)]
    D_indices = [D_sparse[i].indices for i in range(n)]
    D_data = [(-D_sparse[i].data) for i in range(n)]  
    
    w = 0.5 * (A_sparse.sum() + (-D_sparse).sum())
    if w == 0:
        w = 1e-10
    
    communities = [[i] for i in range(n)]
    node_to_community = {i: i for i in range(n)}
    neighbors = [set(A_indices[i]) | set(D_indices[i]) for i in range(n)]
    
    def k_fast(i, c, X_indices, X_data):
        total = 0
        community_set = set(communities[c])
        for idx, neighbor in enumerate(X_indices[i]):
            if neighbor in community_set:
                total += X_data[i][idx]
        return total
    
    def has_isolate(c):
        community_nodes = communities[c]
        if len(community_nodes) <= 1:
            return False
        for i in community_nodes:
            if k_fast(i, c, A_indices, A_data) <= alpha:
                return True
        return False
    
    def SE_c(c):
        if has_isolate(c):
            return 0
        
        nodes = communities[c]
        community_set = set(nodes)
        In_A, In_D, Out_A, Out_D = 0.0, 0.0, 0.0, 0.0
        
        for i in nodes:
            for idx, neighbor in enumerate(A_indices[i]):
                if neighbor in community_set:
                    In_A += A_data[i][idx]  
            for idx, neighbor in enumerate(D_indices[i]):
                if neighbor in community_set:
                    In_D += D_data[i][idx] 
        
        In_A *= 0.5
        In_D *= 0.5
        
        for i in nodes:
            for idx, neighbor in enumerate(A_indices[i]):
                if neighbor not in community_set:
                    Out_A += A_data[i][idx]
            for idx, neighbor in enumerate(D_indices[i]):
                if neighbor not in community_set:
                    Out_D += D_data[i][idx]
        
        return 1 + (In_A - Out_A - In_D + Out_D) / w
    
    def delta_SE(i, c_current, c_target):
        if c_current == c_target:
            return 0
        
        term1 = k_fast(i, c_target, A_indices, A_data) - k_fast(i, c_current, A_indices, A_data)
        term2 = -k_fast(i, c_target, D_indices, D_data) + k_fast(i, c_current, D_indices, D_data)
        
        return (2.0 / w) * (term1 + term2)
    
    improved = True
    iteration = 0
    
    pbar = tqdm(total=max_iterations, desc="SEDA iterations")
    while improved and iteration < max_iterations:
        improved = False
        nodes = list(range(n))
        random.shuffle(nodes)
        
        for i in nodes:
            current_community = node_to_community[i]
            candidate_communities = {node_to_community[neighbor] for neighbor in neighbors[i]}
            candidate_communities.add(current_community)
            
            best_community, best_gain = current_community, 0
            for c in candidate_communities:
                gain = delta_SE(i, current_community, c)
                if gain > best_gain:
                    best_gain, best_community = gain, c
            
            if best_community != current_community and best_gain > 0:
                communities[current_community].remove(i)
                communities[best_community].append(i)
                node_to_community[i] = best_community
                improved = True
        
        iteration += 1
        pbar.update(1)
        
        if iteration % 10 == 0:
            import gc
            gc.collect()
    
    pbar.close()
    
    communities = [comm for comm in communities if len(comm) > 0]
    echo_chambers, se_values = [], []
    
    for i, community in enumerate(communities):
        se_value = SE_c(i)
        if se_value > 0:
            echo_chambers.append(community)
            se_values.append(se_value)
    
    avg_SE = np.mean(se_values) if se_values else 0
    
    return echo_chambers, se_values, avg_SE