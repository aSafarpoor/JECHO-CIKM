import heapq
import math
import pickle
from typing import Dict, List, Literal, Optional, Tuple
import igraph as ig
import networkx as nx
import numpy as np
from scipy.sparse import csr_matrix
from tqdm import tqdm
from tqdm.notebook import tqdm
from AttriPPR import AttriPPR
from CustomizedPageRank import OurPageRank
from utils import Leiden

def deterministic_selection_extreme_stable(x, n, extreme_ratio):
    """
    Alternative version using stable sort for even more deterministic behavior 
    which is O(nlong n) instead of partition in O(n)
    but it is stable when dealing with identical values.
    """
    '''
    #### Partition version:
    m = min(n, max(2, math.ceil(extreme_ratio * n)))
    hi = m // 2
    lo = m - hi
    low_idx = np.argpartition(x, lo)[:lo]
    high_idx = np.argpartition(-x, hi)[:hi]
    sel_idx = np.concatenate([low_idx, high_idx])
    '''
    x = np.asarray(x)
    
    m = min(n, max(2, math.ceil(extreme_ratio * n)))
    hi = m // 2
    lo = m - hi
    
    sorted_indices = np.argsort(x, kind='stable')
    low_idx = sorted_indices[:lo]
    high_idx = sorted_indices[-hi:] if hi > 0 else np.array([], dtype=int)
    
    sel_idx = np.concatenate([low_idx, high_idx])
    
    return sel_idx
class JiHSeedSelection:
    @staticmethod
    def sim(u, v, ops):
        # return np.e**(-10 * (ops[u] - ops[v])**2)
        return 1-(ops[u]-ops[v])**2/4
    @staticmethod
    def jaccard_closed(G, ebunch):
        for u, v in ebunch:
            Nu = set(G.neighbors(u,mode='out')) | {u}
            Nv = set(G.neighbors(v,mode='out')) | {v}
            denom = len(Nu | Nv)
            yield (u, v, 0.0 if denom == 0 else len(Nu & Nv) / denom)
    @staticmethod
    def score_computer_main(g, candidates, ops, epsilon=1e-2):
        if isinstance(ops, list):
            ops = {i: ops[i] for i in range(len(ops))}
        good_candidates = set([x for x in candidates if g.degree(x,mode="OUT") > 1])
        bad_candidates = set(candidates)-good_candidates
        pairs = [(e.source, e.target) for e in g.es if e.source in good_candidates or e.target in good_candidates]
        jaccard = {(u, v): score for u, v, score in JiHSeedSelection.jaccard_closed(g, pairs)}
        sim_scores = {(u, v): JiHSeedSelection.sim(u, v, ops) for u, v in pairs}
        for (u, v), score in list(jaccard.items()):
            jaccard[(v, u)] = score
        for (u, v), score in list(sim_scores.items()):
            sim_scores[(v, u)] = score
        nodes_scores = {}
        for node in list(good_candidates):
            neighbors = g.neighbors(node, mode='out')
            temp_val = [sim_scores[(node, v)] * jaccard[(node, v)] for v in neighbors]
            sum_jaccard = sum([jaccard[(node, v)] for v in neighbors])
            nodes_scores[node] = sum(temp_val) / sum_jaccard**(1 - epsilon)
        for node in list(bad_candidates):
            nodes_scores[node] = 0 
        return nodes_scores 
    @staticmethod
    def score_compute_base(g,type,candidates):
        if type == 'degree':
            nodes_scores = {}
            for node in tqdm(list(candidates), desc="final score for candidates"):
                nodes_scores[node] = g.degree(node,mode="OUT")
            return nodes_scores
        else:
            raise ValueError(f"not defined type {type}")
    @staticmethod
    def seed_selection(undirected_graph, opinions_dict, ratio_extreme, ratio_score):
        n = undirected_graph.vcount()
        x = np.array([opinions_dict[i] for i in range(n)])
        candidates = deterministic_selection_extreme_stable(x, n, ratio_extreme)
        candidates = candidates.tolist()
        candidates_scores = JiHSeedSelection.score_computer_main(undirected_graph, candidates, opinions_dict, epsilon=1e-2)
        k2 = max(1, int(len(candidates) * ratio_score))
        top_candidates = heapq.nlargest(k2, candidates_scores, key=candidates_scores.get)
        return top_candidates    
    @staticmethod
    def seed_selection_base(undirected_graph, opinions_dict, ratio_extreme, ratio_score):
        n = undirected_graph.vcount()
        x = np.array([opinions_dict[i] for i in range(n)])
        candidates = deterministic_selection_extreme_stable(x, n, ratio_extreme)
        candidates = candidates.tolist()
        type = 'degree'
        candidates_scores = JiHSeedSelection.score_compute_base(undirected_graph, type, candidates)
        k2 = max(1, int(len(candidates) * ratio_score))
        top_candidates = heapq.nlargest(k2, candidates_scores, key=candidates_scores.get)
        return top_candidates

class PPRSeedSelection:
    @staticmethod
    def model_selection(FlagChoosePPRModel, opinions_dict, graph):
        features_all = np.array([[opinions_dict[i]] for i in range(graph.vcount())])
        if FlagChoosePPRModel == "AttriPPR":
            edges = [(e.source, e.target) for e in graph.es]
            model = AttriPPR(
                num_nodes=graph.vcount(),
                edges=edges,
                features=features_all,
                a=0.15,
                b=0.6,
                tol=1e-10,
                max_iter=1000,
                init='v',
            )
        elif FlagChoosePPRModel == "APPR":
            from scipy.sparse import csr_matrix
            A = graph.get_adjacency_sparse().astype(float)
            model = OurPageRank(A, features_all, use_features=True, personalized=False)
        else:
            raise ValueError("Unknown model")
        return model
    @staticmethod
    def ClusteredAttriPPR(graph, opinions_dict, k_ratio, Leiden_random_seed, FlagChoosePPRModel, FlagUseClustering):
        if k_ratio > 1:
            raise ValueError("Ratio more than 1 error.") 
        undirected_graph = graph.as_undirected()
        if FlagUseClustering:
            _, communities = Leiden(undirected_graph, seed_value=Leiden_random_seed)
        else:
            communities = [list(range(graph.vcount()))]
        model = PPRSeedSelection.model_selection(FlagChoosePPRModel, opinions_dict, graph)
        if FlagChoosePPRModel == "AttriPPR":
            scores_pr = model.compute_pi()
        elif FlagChoosePPRModel == "APPR":
            scores_pr = model.compute_pagerank()
        top_candidates = []
        for S in communities:
            scores_S = [scores_pr[x] for x in S]
            k = max(1, int(len(S) * k_ratio))
            top_k = [
                s for s, _ in sorted(zip(S, scores_S), key=lambda x: x[1], reverse=True)[:k]
            ]
            top_candidates += top_k
        if FlagUseClustering:
            return top_candidates, communities
        return top_candidates

def select_low_change_extremes_matrix(
    G: ig.Graph,
    opinions_dict: Dict,
    extreme_ratio: float,
    low_change_ratio: float,
    neighbor_mode="out",
    weight: Optional[str] = None,
    flag_return_delta_scores=False,
):
    assert 0 < extreme_ratio <= 1
    assert 0 < low_change_ratio <= 1
    if not isinstance(G, ig.Graph):
        raise TypeError("G must be an igraph.Graph.")
    n = G.vcount()
    x = np.asarray([opinions_dict[u] for u in range(n)], dtype=float)
    if weight and weight in G.es.attributes():
        A = G.get_adjacency_sparse(attribute=weight).astype(float)
    else:
        A = G.get_adjacency_sparse().astype(float)
    if neighbor_mode == "out":
        s = A @ x
        deg = np.asarray(A.sum(axis=1)).ravel()
    elif neighbor_mode == "in":
        s = A.T @ x
        deg = np.asarray(A.sum(axis=0)).ravel()
    elif neighbor_mode == "both":
        U = (A + A.T).sign().astype(float)
        s = U @ x
        deg = np.asarray(U.sum(axis=1)).ravel()
    else:
        raise ValueError('neighbor_mode must be "out", "in", or "both".')
    deg_safe = np.where(deg > 0, deg, 1)
    delta = (s / deg_safe) - x
    sel_idx = deterministic_selection_extreme_stable(x, n, extreme_ratio)
    r = min(max(1, math.ceil(low_change_ratio * len(sel_idx))), len(sel_idx) - 1)
    part = np.argpartition(delta[sel_idx], r)[:r]
    winners_idx = sel_idx[part]
    top_candidates = list(winners_idx)
    if flag_return_delta_scores:
        delta_selected = {i: float(delta[i]) for i in sel_idx}
        return top_candidates, delta_selected
    return top_candidates
