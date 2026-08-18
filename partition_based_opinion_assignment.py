import pickle
import random
from math import sqrt

import igraph as ig
import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import truncnorm
from tqdm import tqdm
import random
from math import ceil, inf, sqrt
from utils import read_directed_iGraph_from_file
from FJ_Laplacian_large_graphs import FJLargeScale, opinions_dict_to_vector
from utils import set_all_seeds, streaming_neighborhood

def raw_opinion_assignment(partitions,sigma_of_mu, sigma_in_partition, seed = 42):

    set_all_seeds(seed)

    n_clusters = len(partitions)
    # n = sum([len(x) for x  in partitions])
    cluster_means = np.clip(
        np.random.default_rng(seed).normal(0, sigma_of_mu, n_clusters),
        -1, 1
    )

    cluster_params = {}
    for i, comm in enumerate(partitions):
        mu = cluster_means[i]
        cluster_params[i] = (mu, sigma_in_partition)
    
    opinions = {}
    rng = np.random.default_rng(seed)
    for i, comm in enumerate(partitions):
        mu, sigma = cluster_params[i]
        a, b = (-1 - mu) / sigma, (1 - mu) / sigma
        node_opinions = truncnorm.rvs(a, b, loc=mu, scale=sigma, size=len(comm), random_state=rng)
        for node, opinion in zip(comm, node_opinions):
            opinions[node] = opinion
    return opinions

def fj_opinion_calculation(G,opinions_dict):

    n = G.vcount()

    solver = FJLargeScale(G, dtype=np.float32, chunk_size=5_000_000)
    s = opinions_dict_to_vector(G.vcount(), opinions_dict, dtype=np.float32)
    opinions_dict_FJ, _ = solver.solve(s, tol=1e-3, maxiter=2000, verbose=True)

    opinions_FJ = [opinions_dict_FJ[node] for node in range(n)]
    min_val = min(opinions_FJ)
    max_val = max(opinions_FJ)
    if abs(max_val - min_val) < 1e-10:  
        opinions_FJ = [0.0] * len(opinions_FJ)  # Normalize all to 0
    else:
        opinions_FJ = [2 * (x - min_val) / (max_val - min_val) - 1 for x in opinions_FJ]

    opinions_dict_FJ = {i: opinions_FJ[i] for i in range(n)}
    return opinions_dict_FJ

def normalize_opinions_minimal(opinions_dict, do_scale=True):
    values = np.array(list(opinions_dict.values()))
    if len(values) == 0:
        return {}
    if len(values) == 1 or np.all(values == values[0]):
        return {k: 0.0 for k in opinions_dict}

    # Centering
    centered = values - values.mean()

    if do_scale:
        # Identify fixed nodes (abs >= 0.95)
        fixed_mask = np.abs(values) >= 0.99
        variable_mask = ~fixed_mask

        # Compute scaling only for variable nodes
        if np.any(variable_mask):
            max_abs = np.max(np.abs(centered[variable_mask]))
            if max_abs > 0:
                # Scale variable nodes so their max = 0.95
                scale_factor = 1/max_abs 
                scaled = centered * scale_factor
            else:
                scaled = centered.copy()
        else:
            scaled = centered.copy()

        # Keep fixed nodes at their original value
        scaled[fixed_mask] = values[fixed_mask]

        return {k: float(v) for k, v in zip(opinions_dict.keys(), scaled)}

    # No scaling — just centering
    return {k: float(v) for k, v in zip(opinions_dict.keys(), centered)}

def set_all_seeds_filtering(seed):
    random.seed(seed)
    np.random.seed(seed)

def get_k_neighbors_sorted_opinions_optimized(G, sorted_nodes, community_dict, communities, k, pa=0.1):
    set_all_seeds_filtering(37)
    n = G.vcount()
    node_to_sampled_neighbors = {}
    
    node_index_arr = np.zeros(n, dtype=int)
    for idx, node in enumerate(sorted_nodes):
        node_index_arr[node] = idx
    
    community_sets = [set(comm) for comm in communities]
    all_nodes_set = set(range(n))
    
    for node in range(n):
        comm_id = community_dict[node]
        comm_set = community_sets[comm_id]
        
        # --- Community-based neighbors ---
        comm_nodes = list(comm_set - {node})
        if len(comm_nodes) < k:
            remaining = list(all_nodes_set - comm_set - {node})
            np.random.shuffle(remaining)
            comm_nodes += remaining[:(k - len(comm_nodes))]
        else:
            comm_nodes = np.random.choice(comm_nodes, k, replace=False).tolist()
        
        idx = node_index_arr[node]
        half_k = k // 2
        
        low = max(0, idx - half_k)
        high = min(len(sorted_nodes), idx + (k - half_k))
        
        if low < idx:
            left_indices = np.arange(low, idx)
        else:
            left_indices = np.array([], dtype=int)
            
        if idx + 1 < high:
            right_indices = np.arange(idx + 1, high)
        else:
            right_indices = np.array([], dtype=int)
            
        indices = np.concatenate([left_indices, right_indices])
        sim_candidates = [sorted_nodes[i] for i in indices]
        
        # Pad if needed
        needed = k - len(sim_candidates)
        if needed > 0:
            if low == 0:
                pad_indices = np.arange(high, min(len(sorted_nodes), high + needed))
                sim_candidates.extend([sorted_nodes[i] for i in pad_indices])
            elif high == len(sorted_nodes):
                pad_indices = np.arange(max(0, low - needed), low)
                sim_candidates = [sorted_nodes[i] for i in pad_indices] + sim_candidates
            else:
                pad_left = needed // 2
                pad_right = needed - pad_left
                left_pad_indices = np.arange(max(0, low - pad_left), low)
                right_pad_indices = np.arange(high, min(len(sorted_nodes), high + pad_right))
                sim_candidates = ([sorted_nodes[i] for i in left_pad_indices] + 
                                 sim_candidates + 
                                 [sorted_nodes[i] for i in right_pad_indices])
        
        sim_nodes = sim_candidates[:k]
        
        candidates = comm_nodes + sim_nodes
        weights = np.array([pa] * k + [1 - pa] * k)
        weights /= weights.sum()
        
        selected_indices = np.random.choice(2 * k, size=k, replace=False, p=weights)
        selected_nodes = [candidates[i] for i in selected_indices]
        
        node_to_sampled_neighbors[node] = selected_nodes
    
    return node_to_sampled_neighbors

def filtering_opinion_assignment_optimized(G, opinions, ratio_of_extremes, communities, seed=37, 
                                         number_of_iterations=20, tol=1e-4, k=10, pa=0.1):
   
    set_all_seeds_filtering(seed)
    n = G.vcount()
    
    opinions_array = np.zeros(n)
    for node, val in opinions.items():
        opinions_array[node] = val
    
    community_dict = {}
    for i, C in enumerate(communities):
        for c in C:
            community_dict[c] = i
    
    if ratio_of_extremes > 0:
        extreme_count = int(ratio_of_extremes * n)
        pos_candidates = np.where(opinions_array >= 0)[0]
        neg_candidates = np.where(opinions_array < 0)[0]

        pos_count = extreme_count // 2
        neg_count = extreme_count - pos_count  

        pos_extremes = pos_candidates[np.argpartition(opinions_array[pos_candidates], -pos_count)[-pos_count:]]
        neg_extremes = neg_candidates[np.argpartition(-opinions_array[neg_candidates], -neg_count)[-neg_count:]]

        extreme_indices = np.concatenate([pos_extremes, neg_extremes])
        extremes_set = set(extreme_indices.tolist())

        # Set extreme opinions to ±1
        opinions_array[extreme_indices] = np.sign(opinions_array[extreme_indices])
    else:
        extremes_set = set([])

    communities_sets = [set(community) for community in communities]
    sorted_nodes = np.argsort(opinions_array)  # Get sorted indices
    
    sorted_nodes_list = sorted_nodes.tolist()
    
    for i in tqdm(range(number_of_iterations),desc = "filtering process"):
        node_to_sampled_neighbors = get_k_neighbors_sorted_opinions_optimized(
            G=G, sorted_nodes=sorted_nodes_list, community_dict=community_dict, 
            communities=communities_sets, k=k, pa=pa
        )
        
        new_opinions_array = np.copy(opinions_array)
        max_rel_change = 0.0
        
        for node in range(n):
            if node in extremes_set:
                continue  # Extremes don't change
                
            neighbors = node_to_sampled_neighbors[node]
            new_opinion = np.mean(opinions_array[neighbors])
            new_opinions_array[node] = new_opinion
            
            old_opinion = opinions_array[node]
            if abs(old_opinion) > 1e-10:
                rel_change = abs(new_opinion - old_opinion) / abs(old_opinion)
            else:
                rel_change = abs(new_opinion - old_opinion)
            
            max_rel_change = max(max_rel_change, rel_change)
        
        opinions_array = new_opinions_array
        
        if max_rel_change < tol:
            print(f"Converged after {i+1} iterations (max change: {max_rel_change:.2e} < {tol})")
            break
    
    final_opinions = {}
    for node in range(n):
        final_opinions[node] = float(opinions_array[node])
    
    return final_opinions

def igraph_to_networkx_digraph(G_ig):
    G_nx = nx.DiGraph()
    for v in G_ig.vs:
        G_nx.add_node(v.index, **v.attributes())
    for e in G_ig.es:
        source, target = e.tuple
        G_nx.add_edge(source, target, **e.attributes())
        if not G_ig.is_directed():
            G_nx.add_edge(target, source, **e.attributes())
    return G_nx

def simple_R(cluster,G):
    sset = set(cluster)
    cut_edges = 0
    internal_edges = 0

    for u in sset:
        for v in G[u]:
            if v in sset:
                internal_edges += 1
            else:
                cut_edges += 1

    R = cut_edges / internal_edges if internal_edges > 0 else np.inf
    return R

def plotter(opinions_dict, title, communities):
    opinions = list(opinions_dict.values())
    plt.figure(figsize=(8, 5))
    plt.hist(opinions, bins=100, color='skyblue', edgecolor='black')
    plt.title(title)
    plt.xlabel('Opinion value')
    plt.ylabel('Frequency')
    plt.grid(alpha=0.3)
    plt.show()

    stats = []
    extreme_, pos_, neg_ = 0, 0, 0
    super_Extreme = 0
    for v in opinions_dict.values():
        if abs(v)>=0.8:
            extreme_ += 1
        if abs(v) == 1:
            super_Extreme += 1
        if v > 0:
            pos_ += 1
        else:
            neg_ += 1

    for community in communities:
        
        comm_values = [opinions_dict[n] for n in community]

        mean_val = np.mean(comm_values)
        std_val = np.std(comm_values)
        size = len(comm_values)
        stats.append((size, mean_val, std_val))

    stats.sort(key=lambda x: x[1], reverse=True)
    # print(f"{'Community':<15} {'Size':<10} {'Mean':<10} {'Std Dev':<10}")
    # print("-" * 20)
    small_counter = 0
    for i,temp in enumerate(stats):
        size, mean_val, std_val = temp
        if size<10:
            small_counter+=1
        else:
            pass
            # print(f"{i:<15} {size:<10} {mean_val:<10.3f} {std_val:<10.3f}")
    # print(f"super_ext:{super_Extreme}, extreme:{extreme_}, pos:{pos_}, neg:{neg_}")
    # print(max(opinions_dict.values()), min(opinions_dict.values()))

    # print(f"Smaller than 10: {small_counter}")
        
def insert_silly_noise(opinions_dict, noise_ratio, random_seed=42):
    assert 0 <= noise_ratio <= 1, 'noise ratio is not valid'
    random.seed(random_seed)

    total_nodes = len(opinions_dict)
    # num_noisy     
    half = int(total_nodes * noise_ratio / 2) 

    positives = [k for k, v in opinions_dict.items() if v > 0]
    negatives = [k for k, v in opinions_dict.items() if v < 0]

    positive_nodes = random.sample(positives, min(half, len(positives)))
    negative_nodes = random.sample(negatives, min(half, len(negatives)))

    new_dict = opinions_dict.copy()
    for node in positive_nodes:
        new_dict[node] = random.uniform(0.9, 1.0)
    for node in negative_nodes:
        new_dict[node] = random.uniform(-1.0, -0.9)

    return new_dict

def do_opinion_assignment(G,root_path,dataset_name, number_of_partitions_for_opinion,k_filtering =10 , 
                          k_extreme = 10, pa_filtering = 0.5, pa_extreme = 0.9, 
                          ratio_of_extreme = 0.01, 
                          number_of_filtering_iteration = 10,  seed = 42, sigma_of_mu = 0.5, sigma_of_partition=1,
                          silly_noise_ratio = 0.005,prefix=""):
    random.seed(seed)
    np.random.seed(seed)
    
    partitions_temp =  streaming_neighborhood(G, k = number_of_partitions_for_opinion)
    partitions_list = [[] for i in range(max(partitions_temp)+1)]
    partitions_dict = {}
    for i,partition_id in tqdm(enumerate(partitions_temp)):
        partitions_list[partition_id].append(i)
        partitions_dict[i] = partition_id
    # for x in partitions_list:
    #     print(len(x),len(partitions_temp))
    del(partitions_temp)

    opinions_dict = raw_opinion_assignment(partitions_list,seed= seed, sigma_of_mu = sigma_of_mu, sigma_in_partition=sigma_of_partition)
    opinions_dict = fj_opinion_calculation(G,opinions_dict)
    opinions_dict = normalize_opinions_minimal(opinions_dict,do_scale=True)
    opinions_dict = insert_silly_noise(opinions_dict,silly_noise_ratio/3)
    
    # plotter(opinions_dict,"fj "+dataset_name, partitions_list)
    with open(f"{root_path}/{prefix}_{dataset_name}_fj.pkl", 'wb') as file:
        pickle.dump(opinions_dict, file)

    
    opinions_dict_filtered = filtering_opinion_assignment_optimized(G, opinions_dict, 0, partitions_list, seed=seed, 
                                         number_of_iterations=number_of_filtering_iteration, 
                                         tol=1e-4, k=k_filtering, pa=pa_filtering) # for filtering we have ratio_of_extremes = 0
    
    opinions_dict_filtered = normalize_opinions_minimal(opinions_dict_filtered,do_scale=True)
    opinions_dict_filtered = insert_silly_noise(opinions_dict_filtered,silly_noise_ratio)
    # plotter(opinions_dict_filtered,"filtering "+dataset_name, partitions_list)
    with open(f"{root_path}/{prefix}_{dataset_name}_filtering.pkl", 'wb') as file:
        pickle.dump(opinions_dict_filtered, file)
    

    opinions_dict_extreme = filtering_opinion_assignment_optimized(G, opinions_dict, ratio_of_extreme, partitions_list, seed=seed, 
                                         number_of_iterations=number_of_filtering_iteration,
                                         tol=1e-4, k=k_extreme, pa=pa_extreme)
    opinions_dict_extreme = normalize_opinions_minimal(opinions_dict_extreme,do_scale=True)
    opinions_dict_extreme = insert_silly_noise(opinions_dict_extreme,silly_noise_ratio)

    # plotter(opinions_dict_extreme,"extremes "+dataset_name, partitions_list)
    with open(f"{root_path}/{prefix}_{dataset_name}_extremes.pkl", 'wb') as file:
        pickle.dump(opinions_dict_extreme, file)    

    print("done")

def real_world_data_generation_API(prefix, seed_id, dataset_name, pre_path):

    G, UG = read_directed_iGraph_from_file(
        filepath=f"{pre_path}{dataset_name}_edges.txt"
    )
    del UG

    N = G.vcount()
    similar_to_sbm_blocks = ceil(sqrt(N))
    sigma_of_mu = 0.3
    silly_noise_ratio = 0.03
    sigma_of_partition = 1

    do_opinion_assignment(
        G,
        pre_path,
        dataset_name,
        k_filtering=5,
        k_extreme=5,
        pa_filtering=0.5,
        pa_extreme=0.9,
        ratio_of_extreme=0.01,
        number_of_filtering_iteration=5,
        seed=seed_id,
        number_of_partitions_for_opinion=int(similar_to_sbm_blocks * 2),
        sigma_of_mu=sigma_of_mu,
        sigma_of_partition=sigma_of_partition,
        silly_noise_ratio=silly_noise_ratio,
        prefix=prefix,
    )






def _make_sbm(n, pref_matrix, block_sizes, directed=True):
    """igraph.Graph.SBM signature differs across versions: some require n as the
    first positional argument, others derive it from block_sizes and reject it.
    Try the positional form first, then fall back to the no-n form."""
    import igraph as _ig
    try:
        return _ig.Graph.SBM(int(n), pref_matrix, block_sizes, directed)
    except TypeError:
        return _ig.Graph.SBM(pref_matrix, block_sizes, directed)


def make_sbm_graph_custom_variance(b, variance=None, seed=42, p = 1e-3, q = 1e-1, modularity_th = 0.6):
    np.random.seed(seed)

    mean_block_size = b
    if variance is None:
        variance = b ** 1.5 
    
    std_block_size = np.sqrt(variance)
    
    block_sizes = np.random.normal(mean_block_size, std_block_size, size=b)
    block_sizes = np.maximum(1, np.round(block_sizes).astype(int))
    n = np.sum(block_sizes)
    pref_matrix = [[p if i == j else q for j in range(b)] for i in range(b)]
    
    G_directed = _make_sbm(n, pref_matrix, block_sizes.tolist(), directed=True)
    
    block_assignments = []
    for i, size in enumerate(block_sizes):
        block_assignments.extend([i] * size)

    G_directed.vs['block'] = block_assignments
    block_assignments_np = list(np.array(block_assignments))
  
    Q = G_directed.modularity(block_assignments_np)
    while Q < modularity_th:
        p = min(1, p * 1.1)
        q = max(1e-5, q * 0.9)

        pref_matrix = [[p if i == j else q for j in range(b)] for i in range(b)]
        G_directed = _make_sbm(n, pref_matrix, block_sizes.tolist(), directed=True)
        
        block_assignments = []
        for i, size in enumerate(block_sizes):
            block_assignments.extend([i] * size)
        G_directed.vs['block'] = block_assignments
        block_assignments_np = list(np.array(block_assignments))
        Q = G_directed.modularity(block_assignments_np)
    G_undirected = G_directed.as_undirected(mode="collapse")
    return G_directed, G_undirected,p,q

def check_isolated_nodes(G):
    in_degrees = G.indegree()
    out_degrees = G.outdegree()
    isolated_nodes = [i for i, (in_d, out_d) in tqdm(enumerate(zip(in_degrees, out_degrees))) 
                     if in_d == 0 and out_d == 0]
    if isolated_nodes:
        return True
    else:
        return False

def compute_sbm_parameters_fixed(b, target_modularity, target_density):

    n = b * b  
    
    total_possible_edges = n * (n - 1)
    expected_edges = target_density * total_possible_edges
  
    r = (target_modularity * b / (b-1) + 1) / (1 - target_modularity * b / (b-1))
    
    # From edge count equation:
    intra_edges_per_block = b * (b - 1)
    inter_edges_per_block = b * (n - b)
    
    q = expected_edges / (b * (intra_edges_per_block * r + inter_edges_per_block))
    p = r * q
    
    
    p = min(0.1, max(1e-6, p))  
    q = min(0.1, max(1e-6, q))
    
    estimated_modularity = (p - q) / (p + (b-1)*q) * (b-1)/b
    
    print(f"Computed: p={p:.6f}, q={q:.6f}")
    print(f"Target Q: {target_modularity}, Estimated Q: {estimated_modularity:.4f}")
    
    return p, q

def large_make_sbm_graph_precalculated(b, target_modularity=0.7, seed=42):

    p,q = compute_sbm_parameters_fixed(b, target_modularity = target_modularity, target_density = 0.0005)
    return make_sbm_graph_custom_variance(b, variance=None, seed=seed, p = p, q = q , modularity_th = target_modularity)

def save_edges_to_txt(G, filename):
    edges = [(G.es[e].source, G.es[e].target) for e in range(G.ecount())]
    
    with open(filename, 'w') as f:
        for source, target in edges:
            f.write(f"{source} {target}\n")
    
    print(f"Saved {len(edges)} edges to {filename}")

def SBM_data_generation_API(prefix, seed_id, dataset_name, N, pre_path):
    p = 1e-3
    q = 1e-1
    flag_generate_new_graph = True

    sbm_blocks = ceil(sqrt(N * 1000))
    if flag_generate_new_graph:
        flag_isolated_node = True
        while flag_isolated_node:
            if N < 50:
                G, UG, p, q = make_sbm_graph_custom_variance(
                    b=sbm_blocks, p=p, q=q, seed=seed_id, modularity_th=0.7
                )
            else:
                G, UG, p, q = large_make_sbm_graph_precalculated(
                    b=sbm_blocks, target_modularity=0.7, seed=seed_id
                )
                print(p, q)
            flag_isolated_node = check_isolated_nodes(G)
            seed_id += 1
        save_edges_to_txt(G, f"{pre_path}{prefix}_{dataset_name}_edges.txt")
    else:
        try:
            G, UG = read_directed_iGraph_from_file(
                filepath=f"{pre_path}{prefix}_{dataset_name}_edges.txt"
            )
            del UG
        except:
            raise ValueError("Dataset is not generated yet!")
    do_opinion_assignment(
        G,
        pre_path,
        dataset_name,
        k_filtering=5,
        k_extreme=5,
        pa_filtering=0.5,
        pa_extreme=0.9,
        ratio_of_extreme=0.01,
        number_of_filtering_iteration=5,
        seed=seed_id,
        number_of_partitions_for_opinion=int(sbm_blocks * 2.5),
        sigma_of_mu=0.2,
        sigma_of_partition=1,
        silly_noise_ratio=0.03,
        prefix=prefix,
    )
