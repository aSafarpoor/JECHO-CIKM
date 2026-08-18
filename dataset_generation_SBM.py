import pickle
import random
from math import inf
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from tqdm import tqdm
from utils import Leiden, set_all_seeds
from scipy.stats import truncnorm

def communities_info(G, communities):
    ps = []
    qs = []
    for _, community in enumerate(communities):
        nodes_in_comm = set(community)
        n = len(G)
        nc = len(nodes_in_comm)
        inner_edges = sum(
            1
            for u, v in G.edges(nodes_in_comm)
            if u in nodes_in_comm and v in nodes_in_comm
        )
        boundary_edges = sum(
            1
            for u, v in G.edges(nodes_in_comm)
            if (u in nodes_in_comm) != (v in nodes_in_comm)
        )
        if nc == 1:
            p, q = -1, -1
        else:
            p = round(inner_edges / (nc * (nc - 1) / 2), 4)
            q = round(boundary_edges / (nc * (n - nc)), 4)
            ps.append(p)
            qs.append(q)
    return round(np.mean(ps), 4), round(np.mean(qs), 4)
def FJ_closed_form_Computation(G: nx.Graph, opinions_dict: dict):
    n = len(G.nodes())
    S = np.zeros(n)
    for i in range(n):
        S[i] = opinions_dict[i]
    A = nx.to_numpy_array(G, nodelist=list(range(n)))
    D = np.diag(np.sum(A, axis=1))
    L = D - A
    # FJ model: (I + L)^(-1) * S
    I = np.eye(n)
    try:
        FJ_matrix = np.linalg.inv(I + L)
    except np.linalg.LinAlgError:
        raise ValueError("Matrix (I + L) is singular. Graph may be disconnected.")
    final_opinions = FJ_matrix @ S
    opinions_dict2 = {i: final_opinions[i] for i in range(n)}
    return opinions_dict2
def SBM(
    n_clusters,
    mean_normal_for_clusters_size,
    std_dev_for_clusters_size,
    random_seed,
    p,
    q,
):
    set_all_seeds(random_seed)
    # size of blocks
    numbers = np.random.normal(
        mean_normal_for_clusters_size, std_dev_for_clusters_size, n_clusters
    )
    clusters_sizes = np.round(numbers).astype(int)
    P_matrix = [
        [p if i == j else q for j in range(n_clusters)] for i in range(n_clusters)
    ]
    G = nx.stochastic_block_model(clusters_sizes, P_matrix, seed=random_seed)
    edges_dir = list(G.edges())
    edges = [[u, v] for u, v in edges_dir] + [[v, u] for u, v in edges_dir]
    # Assign cluster IDs
    cluster_ids = {}
    start = 0
    for c, size in enumerate(clusters_sizes):
        for i in range(start, start + size):
            cluster_ids[i] = c
        start += size
    G.number_of_edges(), G.number_of_nodes(), len(edges)
    cluster_dict, communities = Leiden(G)
    return G, cluster_dict, communities, clusters_sizes
def opinion_assignment(
    G,
    mean_normal=0,
    std_dev=0.5,
    std_dev_of_nodes_from_mean_of_their_clusters=0.1,
    lower=-1,
    upper=1,
    seed=42,
):
    set_all_seeds(seed)
    cluster_dict, communities = Leiden(G)
    n_clusters = len(communities)
    a, b = (lower - mean_normal) / std_dev, (upper - mean_normal) / std_dev
    opinion_means = truncnorm.rvs(a, b, loc=mean_normal, scale=std_dev, size=n_clusters)
    n = len(G)
    
    opinions = {}
    std_node = std_dev_of_nodes_from_mean_of_their_clusters
    for node in range(n):
        clstr = cluster_dict[node]
        mean_node = opinion_means[clstr]
        opinions[node] = max(-1, min(1, np.random.normal(mean_node, std_node)))
    # FJ
    opinions_dict = {i: opinions[i] for i in range(n)}
    opinions_dict_FJ = FJ_closed_form_Computation(G, opinions_dict)
    obinions_FJ = [opinions_dict_FJ[node] for node in range(n)]
    # Rescale
    scaled_ops = [
        2 * (x - min(obinions_FJ)) / (max(obinions_FJ) - min(obinions_FJ)) - 1
        for x in obinions_FJ
    ]
    
    scaled_ops_dict = {i: scaled_ops[i] for i in range(len(scaled_ops))}
    return scaled_ops_dict
def get_k_neighbors_sorted_opinions(
    G, opinions, community_dict, communities, k=5, pa=0.1
):
    set_all_seeds(42)
    node_to_sampled_neighbors = {}
    all_nodes = list(G.nodes())
    # Step 1: sort nodes by opinion value
    sorted_nodes = sorted(all_nodes, key=lambda n: opinions[n])
    node_index = {node: idx for idx, node in enumerate(sorted_nodes)}
    for node in all_nodes:
        # --- Community-based neighbors ---
        comm_nodes = [n for n in communities[community_dict[node]] if n != node]
        if len(comm_nodes) < k:
            remaining = list(set(all_nodes) - set(comm_nodes) - {node})
            random.shuffle(remaining)
            comm_nodes += remaining[: (k - len(comm_nodes))]
        else:
            comm_nodes = random.sample(comm_nodes, k)
        # --- Opinion-based similarity neighbors using sorted list ---
        idx = node_index[node]
        half_k = k // 2
        low = max(0, idx - half_k)
        high = min(len(sorted_nodes), idx + (k - half_k))
        sim_candidates = (
            sorted_nodes[low:idx] + sorted_nodes[idx + 1 : high]
        )  # exclude self
        needed = k - len(sim_candidates)
        # If not enough, pad from either end
        if needed > 0:
            if low == 0:
                sim_candidates += sorted_nodes[high : high + needed]
            elif high == len(sorted_nodes):
                sim_candidates = (
                    sorted_nodes[max(0, low - needed) : low] + sim_candidates
                )
            else:
                # Prefer symmetric padding
                pad_left = needed // 2
                pad_right = needed - pad_left
                sim_candidates = (
                    sorted_nodes[max(0, low - pad_left) : low]
                    + sim_candidates
                    + sorted_nodes[high : high + pad_right]
                )
        sim_nodes = sim_candidates[:k]
        # --- Final sampling with weighted probability ---
        candidates = comm_nodes + sim_nodes
        weights = np.array([pa] * k + [1 - pa] * k)
        weights /= weights.sum()
        selected_indices = np.random.choice(2 * k, size=k, replace=False, p=weights)
        selected_nodes = [candidates[i] for i in selected_indices]
        node_to_sampled_neighbors[node] = selected_nodes
    return node_to_sampled_neighbors
def filtering(G, opinions, ratio_of_extremes, k, p, number_of_iterations, seed=42):
    set_all_seeds(seed)
    top_k_nodes = sorted(opinions.items(), key=lambda item: abs(item[1]), reverse=True)[
        : int(ratio_of_extremes * len(opinions))
    ]
    extremes = [node for node, _ in top_k_nodes]
    for node in extremes:
        opinions[node] = np.sign(opinions[node])
    community_dict, communities = Leiden(G)
    communities = [set(community) for community in communities]
    for i in tqdm(range(number_of_iterations)):
        node_to_sampled_neighbors = get_k_neighbors_sorted_opinions(
            G, opinions, community_dict, communities, k=10, pa=p
        )
        new_opinions = {}
        for node, neigs in node_to_sampled_neighbors.items():
            if node in extremes:
                new_opinions[node] = opinions[node]
            else:
                new_opinions[node] = np.mean([opinions[neig] for neig in neigs])
        opinions = new_opinions.copy()
    return opinions
def main(n_clusters, p, q, mean_normal_for_clusters_size, std_dev_for_clusters_size):
    G, cluster_dict, _, sizes = SBM(
        n_clusters,
        mean_normal_for_clusters_size,
        std_dev_for_clusters_size,
        random_seed=42,
        p=p,
        q=q,
    )
    block_list = []
    for i in sizes:
        for _ in range(i):
            block_list.append(i)
    set_all_seeds(42)
    cluster_list = []
    for node in G.nodes():
        cluster_list.append(cluster_dict[node])
    
    opinions_dict = opinion_assignment(
        G,
        mean_normal=0,
        std_dev=0.5,
        std_dev_of_nodes_from_mean_of_their_clusters=0.2,
        lower=-1,
        upper=1,
    )
    pickle.dump(
        opinions_dict,
        open(f"datasets/FastAPPR_Datasets/SBM_opinions_raw_p{p}_q{q}.pkl", "wb"),
    )
    edges = [[u, v] for u, v in list(G.edges())] + [[v, u] for u, v in list(G.edges())]
    edges = list(set(tuple(edge) for edge in edges))
    edges = [list(edge) for edge in edges]
    pickle.dump(
        edges, open(f"datasets/FastAPPR_Datasets/SBM_edges_p{p}_q{q}.pkl", "wb")
    )
    G = nx.Graph()
    G.add_nodes_from(list(range(len(opinions_dict))))
    G.add_edges_from(edges)
    ratio_of_extremes_list = [0, 1]
    for ratio_of_extremes in ratio_of_extremes_list:
        opinions = opinions_dict.copy()
        opinions = filtering(
            G,
            opinions,
            ratio_of_extremes=ratio_of_extremes,
            k=10,
            p=0.1,
            number_of_iterations=50,
            seed=42,
        )
        pickle.dump(
            opinions,
            open(
                f"datasets/FastAPPR_Datasets/SBM_opinions_extreme{ratio_of_extremes}_p{p}_q{q}.pkl",
                "wb",
            ),
        )
