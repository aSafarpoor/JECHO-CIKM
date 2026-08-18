import pickle
import time
from math import inf
import numpy as np
from sklearn.metrics import pairwise_distances
from utils import compute_stats_and_D


def NormalECD(uG, G, opinion_dict,thresholds,distance_metric='euclidean'):
    communities = uG.community_multilevel()
    community_membership = communities.membership
    
    unique_communities = set(community_membership)
    community_nodes = {}
    for comm_id in unique_communities:
        community_nodes[comm_id] = [i for i, c in enumerate(community_membership) if c == comm_id]
    
    node_ids = range(uG.vcount())
    opinion_matrix = np.array([opinion_dict.get(i, 0) for i in node_ids]).reshape(-1, 1)
    
    # SCALABILITY FIX Use memory-efficient pairwie distances for large graphs
    n_nodes = len(opinion_matrix)
    if n_nodes > 10000:
        chunk_size = 5000
        dist_matrix = np.zeros((n_nodes, n_nodes))
        
        for i in range(0, n_nodes, chunk_size):
            i_end = min(i + chunk_size, n_nodes)
            for j in range(0, n_nodes, chunk_size):
                j_end = min(j + chunk_size, n_nodes)
                chunk_dist = pairwise_distances(
                    opinion_matrix[i:i_end], 
                    opinion_matrix[j:j_end], 
                    metric=distance_metric
                )
                dist_matrix[i:i_end, j:j_end] = chunk_dist
    else:
        dist_matrix = pairwise_distances(opinion_matrix, metric=distance_metric)
    
    if dist_matrix.max() > 0: 
        dist_matrix = dist_matrix / dist_matrix.max() 
    
    community_ecs = {}
    
    for comm_id, nodes in community_nodes.items():
        if len(nodes) < 2:
            community_ecs[comm_id] = 0.0
            continue
            
        community_scores = []
        set_of_nodes = set(nodes)
        
        if len(nodes) > 1000:
            sample_nodes = np.random.choice(nodes, size=1000, replace=False)
        else:
            sample_nodes = nodes
            
        for u in sample_nodes:
            same_comm_nodes = list(set_of_nodes - {u})
            
            if len(same_comm_nodes) > 500:
                same_comm_nodes = np.random.choice(same_comm_nodes, size=500, replace=False)
            
            if len(same_comm_nodes)>0:
                cohesion = np.mean([dist_matrix[u, v] for v in same_comm_nodes])
            else:
                cohesion = 0
            
            separation_values = []
            for other_comm_id, other_nodes in community_nodes.items():
                if other_comm_id != comm_id and other_nodes:
                    if len(other_nodes) > 500:
                        other_nodes_sampled = np.random.choice(other_nodes, size=500, replace=False)
                    else:
                        other_nodes_sampled = other_nodes
                    
                    other_dist = np.mean([dist_matrix[u, v] for v in other_nodes_sampled])
                    separation_values.append(other_dist)
            
            separation = min(separation_values) if separation_values else 1.0
            
            if max(separation, cohesion) > 0:
                ecs_component = (max(separation, cohesion) + separation - cohesion) / (2 * max(separation, cohesion))
            else:
                ecs_component = 0.5  
            
            community_scores.append(ecs_component)
        
        community_ecs[comm_id] = np.mean(community_scores)
    
    bestBestEC = {'R': inf,'EC': [],'V_S': -1,'Delta_E': -1}
    sorted_communities = sorted(community_ecs.items(), key=lambda x: x[1], reverse=True)
    ECs = []
    for comm_id, ecs_value in sorted_communities:
        comm_nodes = community_nodes[comm_id]
        E_S, V_S, delta_E, R = compute_stats_and_D(G, comm_nodes, opinion_dict)
        ECs.append([delta_E,V_S,R,len(comm_nodes)])

        if len(comm_nodes)>=10 and delta_E>=thresholds['theta_E'] and V_S<=thresholds['theta_V'] and R<=bestBestEC['R']:
            bestBestEC = {'R': R,
                        'EC': comm_nodes,
                        'LenEC': len(comm_nodes),
                        'V_S': V_S,
                        'Delta_E': delta_E}  
    return bestBestEC,ECs
  
def LargeECD(uG, G, opinion_dict, thresholds, distance_metric='euclidean'):
   
    communities = uG.community_multilevel()
    community_membership = communities.membership
    
    unique_communities = set(community_membership)
    community_nodes = {}
    for comm_id in unique_communities:
        community_nodes[comm_id] = [i for i, c in enumerate(community_membership) if c == comm_id]
    
    node_opinions = np.array([opinion_dict.get(i, 0) for i in range(uG.vcount())])
    community_ecs = {}
    for comm_id, nodes in community_nodes.items():
        if len(nodes) < 2:
            community_ecs[comm_id] = 0.0
            continue
            
        community_scores = []
        comm_opinions = node_opinions[nodes]
        
        if len(nodes) > 1000:
            sample_indices = np.random.choice(len(nodes), 1000, replace=False)
        else:
            sample_indices = range(len(nodes))
            
        for idx in sample_indices:
            u = nodes[idx]
            u_opinion = node_opinions[u]
            
            other_indices = [i for i in range(len(nodes)) if i != idx]
            if len(other_indices) > 200:  
                other_indices = np.random.choice(other_indices, 200, replace=False)
                
            cohesion = np.mean([abs(u_opinion - comm_opinions[i]) for i in other_indices])
            separation_values = []
            for other_comm_id, other_nodes in community_nodes.items():
                if other_comm_id != comm_id and other_nodes:
                    if len(other_nodes) > 200:
                        other_sample = np.random.choice(other_nodes, 200, replace=False)
                    else:
                        other_sample = other_nodes
                    
                    other_opinions = node_opinions[other_sample]
                    other_dist = np.mean([abs(u_opinion - op) for op in other_opinions])
                    separation_values.append(other_dist)
            
            separation = min(separation_values) if separation_values else 1.0
            
            cohesion = cohesion / 2.0
            separation = separation / 2.0
            
            if max(separation, cohesion) > 0:
                ecs_component = (max(separation, cohesion) + separation - cohesion) / (2 * max(separation, cohesion))
            else:
                ecs_component = 0.5
            
            community_scores.append(ecs_component)
        
        community_ecs[comm_id] = np.mean(community_scores)
    
    sorted_communities = sorted(community_ecs.items(), key=lambda x: x[1], reverse=True)
    bestBestEC = {'R': inf,'EC': [],'V_S': -1,'Delta_E': -1}
    ECs = []
    for comm_id, ecs_value in sorted_communities:
        comm_nodes = community_nodes[comm_id]
        E_S, V_S, delta_E, R = compute_stats_and_D(G, comm_nodes, opinion_dict)
        ECs.append([delta_E,V_S,R,len(comm_nodes)])
        if len(comm_nodes)>=10 and delta_E>=thresholds['theta_E'] and V_S<=thresholds['theta_V'] and R<=bestBestEC['R']:
            bestBestEC = {'R': R,
                        'LenEC': len(comm_nodes),
                        'EC': comm_nodes,
                        'V_S': V_S,
                        'Delta_E': delta_E}  
    
    return bestBestEC, ECs
 
def ECD(UG,G,dataset_name_prime,pre_path,thresholds):
    opinions = pickle.load(open(pre_path + f"{dataset_name_prime}.pkl", "rb"))

    start = time.perf_counter()
    
    if UG.vcount() > 20000:
        bestEC,ECs = LargeECD(UG, G,opinions,thresholds = thresholds)
    else:
        bestEC,ECs = NormalECD(UG, G, opinions,thresholds=thresholds)

    end = time.perf_counter()
    bestEC['time'] = f"{end - start:.2f}"
    return bestEC,ECs
