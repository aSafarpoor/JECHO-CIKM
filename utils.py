import random
from math import inf

import igraph as ig
import leidenalg as la
import networkx as nx
import numpy as np
import pickle
import time
import os

def save_pickle(pre_path_results, name, obj):
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"{name}_{timestamp}.pkl"
    path = os.path.join(pre_path_results, filename)

    os.makedirs(pre_path_results, exist_ok=True)

    with open(path, "wb") as f:
        pickle.dump(
            {
                "data": obj,
                "name": name,
                "time": timestamp,
            },
            f
        )

    return path


def streaming_neighborhood(g, k):
    n = g.vcount()
    vertex_to_part = np.full(n, -1, dtype=np.int32)
    part_weights = np.zeros(k, dtype=np.int32)
    
    degrees = g.degree()
    order = np.argsort(degrees)[::-1]
    
    part_counts = np.zeros(k, dtype=np.int32)
    target_size = n / k  # Target size for each partition
    
    for v in order:
        if vertex_to_part[v] != -1:
            continue
            
        part_counts.fill(0)
        neighbors = g.neighbors(v)
        
        for u in neighbors:
            p = vertex_to_part[u]
            if p != -1:
                part_counts[p] += 1
        
        alpha = 0.1  
        scores = part_counts - alpha * (part_weights - target_size)
        
        best_p = np.argmax(scores)
        
        if part_counts[best_p] == 0:
            best_p = np.argmin(part_weights)
        
        vertex_to_part[v] = best_p
        part_weights[best_p] += 1
    
    return vertex_to_part.tolist()

class constant_keeper:
    
    @staticmethod
    def __getattribute__(name):
        thresholds = {'theta_V':0.03, 'theta_E':0.3, 'theta_R':2}
        if name == 'Threshold':
            return thresholds
        else:
            raise ValueError(f"We don't have {name}")

class update_all:
    def __init__(self,E_V,G,opinion_dict):
        self.E_V = E_V
        self.G = G
        self.opinion_dict = opinion_dict
        self.old_S_set = set()
        self.old_E_S = None 
        self.old_V_S = None  
        self.internal_edges = 0
        self.cut_edges = 0

    def re_update_online(self,new_node):
        
        new_S_set = self.old_S_set | {new_node}
        n = len(self.old_S_set)

        opinion_of_new_node = self.opinion_dict[new_node]
        if n == 0:
            new_E_S = opinion_of_new_node
            new_V_S = 0.0
            delta_E = abs(self.E_V - new_E_S)
        else:
            new_E_S = (self.old_E_S * n + opinion_of_new_node) / (n + 1)
            delta1 = opinion_of_new_node - self.old_E_S
            delta2 = opinion_of_new_node - new_E_S
            new_V_S = (self.old_V_S * n + delta1 * delta2) / (n + 1)
            delta_E = abs(self.E_V - new_E_S)

        for neighbor in self.G.neighbors(new_node,mode='OUT'):  
            if neighbor in self.old_S_set:
                self.internal_edges += 1
            else:
                self.cut_edges += 1
                
        for neighbor in self.G.neighbors(new_node,mode='IN'):  
            if neighbor in self.old_S_set:
                self.internal_edges += 1
                self.cut_edges -= 1

        D_S = float('inf') if self.internal_edges == 0 else self.cut_edges / self.internal_edges
        if D_S < 0:
            print(new_node,self.cut_edges,self.internal_edges)
            raise ValueError('Negative Ratio')
        
        
        self.old_S_set = new_S_set
        self.old_E_S = new_E_S 
        self.old_V_S = new_V_S  

        return new_S_set,new_V_S,delta_E,D_S



def update_all_online(
    G,
    new_node,
    opinion_dict,
    old_S_set,
    old_E_S,
    old_V_S,
    E_V,
    internal_edges,
    cut_edges,
):
    new_S_set = old_S_set | {new_node}
    n = len(old_S_set)

    opinion_of_new_node = opinion_dict[new_node]
    if n == 0:
        new_E_S = opinion_of_new_node
        new_V_S = 0.0
        delta_E = abs(E_V - new_E_S)
    else:
        new_E_S = (old_E_S * n + opinion_of_new_node) / (n + 1)
        delta1 = opinion_of_new_node - old_E_S
        delta2 = opinion_of_new_node - new_E_S
        new_V_S = (old_V_S * n + delta1 * delta2) / (n + 1)
        delta_E = abs(E_V - new_E_S)

        for neighbor in G.neighbors(new_node,mode='OUT'):  
            if neighbor in old_S_set:
                internal_edges += 1
            else:
                cut_edges += 1
        for neighbor in G.neighbors(new_node,mode='IN'):  
            if neighbor in old_S_set:
                internal_edges += 1
                cut_edges -= 1

    D_S = float('inf') if internal_edges == 0 else cut_edges / internal_edges
    if D_S < 0:
        print(new_node,cut_edges,internal_edges)
        raise ValueError('Negative Ratio')
    
    return new_S_set, new_E_S, new_V_S, delta_E, internal_edges, cut_edges, D_S

def best_singelton(G, opinions_dictionary, E_V):
    best_node = -1
    delta_E = 0
    for node in G.nodes():
        dif = abs(opinions_dictionary[node] - E_V)
        if dif > delta_E:
            delta_E = dif
            best_node = node

    return best_node, delta_E

def read_directed_iGraph_from_file(filepath: str) -> ig.Graph:
    edges = []
    with open(filepath, 'r') as file:
        for line in file:
            if line.strip():  
                src, dst = map(int, line.strip().split())
                edges.append((src, dst))

    max_node = max(max(u, v) for u, v in edges)
    G = ig.Graph(directed=True)
    G.add_vertices(max_node + 1) 

    G.add_edges(edges)
    # Ensure all nodes are included (even if isolated)
    max_node = max(max(u, v) for u, v in edges)
    if G.vcount() <= max_node:
        G.add_vertices(max_node - G.vcount() + 1)

    G_undirected = G.as_undirected()
    return G,G_undirected

def Leiden_igraph(g, resolution=0.05, seed=42, use_weights=False):
    weights = "weight" if use_weights and "weight" in g.es.attributes() else None
    partition = la.find_partition(g, la.CPMVertexPartition,
                                  resolution_parameter = resolution,
                                  seed = seed);
    return partition

def Leiden(G, seed_value=42):
    """
    !pip install leidenalg
    """
    # import leidenalg as la
    # import igraph as ig
    if isinstance(G, ig.Graph):
        pass
    elif isinstance(G, (nx.DiGraph, nx.Graph)):
        G = ig.Graph.TupleList(G.edges(), directed=False)
    else:
        raise ValueError("G is NOT an igraph.Graph or networkX — handle accordingly")
    assert not G.is_directed(), "Assertion failed: Graph is directed, but expected undirected."

    partition = la.find_partition(
        G, la.ModularityVertexPartition, seed=seed_value
    )
    partion_id = 0
    community_dict = {}
    for C in partition:
        for v in C:
            community_dict[v] = partion_id
        partion_id += 1

    communities = [i for i in partition]

    return community_dict, communities


def update_V_S_and_E_S_online(old_S, opinion_of_new_node, old_E_S, old_V_S, E_V):
    n = len(old_S)
    if n == 0:
        return opinion_of_new_node, 0.0, abs(E_V - opinion_of_new_node)

    new_E_S = (old_E_S * n + opinion_of_new_node) / (n + 1)
    delta_E = abs(E_V - new_E_S)
    delta1 = opinion_of_new_node - old_E_S
    delta2 = opinion_of_new_node - new_E_S
    new_V_S = (old_V_S * n + delta1 * delta2) / (n + 1)
    return new_E_S, new_V_S, delta_E

from math import inf

import numpy as np


def compute_stats_and_D(G, S, opinions_dict,E_V = None):
   
    if len(S) == 0:
        return np.nan, np.nan, np.nan, np.nan

    # Extract opinions for S
    S_opinions = np.array([opinions_dict[u] for u in S], dtype=np.float64)
    if E_V is None:
        E_V = np.mean(list(opinions_dict.values()))  

    E_S = np.mean(S_opinions)
    delta_E = abs(E_S - E_V)
    V_S = np.var(S_opinions, ddof=0) if len(S_opinions) > 1 else 0.0

    # ---- Graph-based D(S) ----
    if len(S) < 2:
        D_S = inf
    else:
        S_set = set(S)
        cut_edges = 0
        internal_edges = 0

        for u in S_set:
            for v in G.neighbors(u,mode='OUT'):  
                if v in S_set:
                    internal_edges += 1
                else:
                    cut_edges += 1
            

        D_S = cut_edges / internal_edges if internal_edges > 0 else inf
    if D_S < 0:
        raise ValueError('Negative Ratio')
    
    return round(E_S, 5), round(V_S, 5), round(delta_E, 5), round(D_S, 5)



def update_V_S_and_E_S_once(S_opinions, E_V):
    S_opinions = np.array(S_opinions, dtype=np.float64)
    if len(S_opinions) == 0:
        return np.nan, np.nan, np.nan  # or raise an error
    E_S = np.mean(S_opinions)
    delta_E = abs(E_S - E_V)
    V_S = np.var(S_opinions, ddof=0) if len(S_opinions) > 1 else 0.0
    return round(E_S, 5), round(V_S, 5), round(delta_E, 5)


def compute_D(G, S):
    if len(S) < 2:
        return inf
    S_set = set(S)
    cut_edges = 0
    internal_edges = 0

    for u in S_set:
        for v in G[u]:
            if v in S_set:
                internal_edges += 1
            else:
                cut_edges += 1
    internal_edges //= 2
    if internal_edges == 0:
        return inf
    D_S = cut_edges / internal_edges
    return D_S


def set_all_seeds(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)


# def set_all_seeds_plus_torch(seed: int = 42):
#     random.seed(seed)
#     np.random.seed(seed)
#     import torch

#     torch.manual_seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed(seed)
#         torch.cuda.manual_seed_all(seed)
#     # Ensure deterministic behavior in cuDNN (may reduce performance)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False


def singleton_baseline(G, opinions_dict, theta_E, E_V=None):
    """Singleton lower-bound baseline (paper Section 5.2).

    Used only when a method returns no valid echo chamber of size >= 10.
    A node v is feasible if |E(o,V) - o_v| >= theta_E; for a feasible
    singleton, structural isolation is I({v}, V) = deg+(v). Returns the
    minimum deg+(v) over feasible singletons, or None if none is feasible.

    This is a reporting fallback applied when aggregating results into the
    final table; the detection runners themselves return R=inf on failure.
    """
    if E_V is None:
        E_V = float(np.mean(list(opinions_dict.values())))
    best = None
    for v in range(G.vcount()):
        if abs(E_V - opinions_dict[v]) >= theta_E:
            d = len(G.neighbors(v, mode="OUT"))
            if best is None or d < best:
                best = d
    return best
