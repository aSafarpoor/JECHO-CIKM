import pickle
import random
import time
from itertools import product
from math import ceil, log2
import igraph
import numpy as np
from heapdict import heapdict
from tqdm import tqdm
from AttriPPR import AttriPPR
from utils import read_directed_iGraph_from_file, update_all

class score_computation:
    def __init__(self, G, opinions_dict, out_degrees):
        self.G = G
        self.opinions_dict = opinions_dict
        self.n = G.vcount()
        print(self.n, len(opinions_dict), "\n\n")
        self.out_degrees = out_degrees
        self.features = None
        self.APR_model = None
        self.edges = [(e.source, e.target) for e in G.es]
        self.out_neighbor_set = self.preprocess_neighbors(
            mode="out", flag_self_loop=False
        )
        jac_values = self.G.similarity_jaccard(pairs=self.edges, mode="out")
        self.jaccard_dict = {(u, v): j for (u, v), j in zip(self.edges, jac_values)}
    def CC(self, node):
        neighbors_u = self.out_neighbor_set[node]
        deg_u = self.out_degrees[node]
        if deg_u <= 1:
            return 0.0
        neighbors_list = sorted(neighbors_u, key=lambda x: self.out_degrees[x])
        triangles = 0
        for i, v in enumerate(neighbors_list):
            neighbors_v = self.out_neighbor_set[v]
            for j in range(i + 1, len(neighbors_list)):
                w = neighbors_list[j]
                if w in neighbors_v:
                    triangles += 1
        CC = (2 * triangles) / (deg_u * (deg_u - 1))
        return CC
    def pure_entropy_score_computation(self, node):
        o_u = self.opinions_dict.get(node, 0.0)
        p_plus = (o_u + 1) / 2
        p_minus = 1 - p_plus
        p_plus = max(p_plus, 1e-10)
        p_minus = max(p_minus, 1e-10)
        H = -(p_plus * log2(p_plus) + p_minus * log2(p_minus))
        score = 1 - H
        return score
    def entropy_score_computation(self, node):
        o_u = self.opinions_dict.get(node, 0.0)
        neighbors = self.out_neighbor_set[node]
        if not neighbors:
            return 0
        sims = [1 - abs(o_u - self.opinions_dict[n]) / 2.0 for n in neighbors]
        avg_sim = sum(sims) / len(sims)
        # Entropy & polarization
        p_plus = (o_u + 1) / 2
        p_minus = 1 - p_plus
        p_plus = max(p_plus, 1e-10)
        p_minus = max(p_minus, 1e-10)
        H = -(p_plus * log2(p_plus) + p_minus * log2(p_minus))
        polarization = 1 - H
        return avg_sim * polarization
    def absolute_opinion_value(self, node):
        return abs(self.opinions_dict[node])
    def preprocess_neighbors(self, mode="out", flag_self_loop=True):
        neighbor_sets = {}
        mode_map = {"out": igraph.OUT, "in": igraph.IN, "all": igraph.ALL}
        m = mode_map.get(mode, igraph.OUT)
        for v in tqdm(range(self.G.vcount())):
            N = set(self.G.neighbors(v, mode=m))
            if flag_self_loop:
                N.add(v)
            neighbor_sets[v] = N
        return neighbor_sets
    def JiHom(self, node):
        score_total = 0
        jaccard_total = 0
        for v in self.out_neighbor_set[node]:
            jaccard = self.jaccard_dict[(node, v)]
            diff = self.opinions_dict[node] - self.opinions_dict[v]
            sim_scores = 1 - ((diff**2) / 4)
            score_total += jaccard * sim_scores
            jaccard_total += jaccard
        return score_total / jaccard_total if jaccard_total != 0 else 0
    def Hom(self, node):
        score_total = 0
        for v in self.out_neighbor_set[node]:
            diff = self.opinions_dict[node] - self.opinions_dict[v]
            sim_scores = 1 - ((diff**2) / 4)
            score_total += sim_scores
        return (
            score_total / len(self.out_neighbor_set[node])
            if len(self.out_neighbor_set[node]) != 0
            else 0
        )
    def init_APR(self):
        self.features = np.array(
            [[self.opinions_dict[i]] for i in range(self.G.vcount())]
        )
        self.APR_model = AttriPPR(
            self.n,
            self.edges,
            self.features,
            a=0.15,
            b=0.1,
            tol=1e-10,
            max_iter=100000,
            init="v",
        )
        self.pi = self.APR_model.compute_pi(seeds=-1)
    def APR_helper_score_computation(self, node):
        if self.features is None or self.APR_model is None or self.pi is None:
            self.init_APR()
        return self.pi[node]

class expansion_class:
    def __init__(self, G, opinions_dict, random_seed=42):
        self.G = G
        self.opinions_dict = opinions_dict
        self.nodes = list(range(G.vcount()))
        self.E_V = np.mean(list(opinions_dict.values()))
        self.expansion_APPR_model = None
        self.out_degrees = {}
        self.in_neigbors = {}
        self.out_neigbors = {}
        self.edges = [(e.source, e.target) for e in G.es]
        self.opinions_as_features_numpy = np.array(
            [[opinions_dict[i]] for i in range(G.vcount())]
        )
        random.seed(random_seed)
        self.features_all = None
        for node in self.nodes:
            self.out_degrees[node] = len(self.G.neighbors(node, mode="OUT"))
            self.in_neigbors[node] = set(self.G.neighbors(node, mode="IN"))
            self.out_neigbors[node] = set(self.G.neighbors(node, mode="OUT"))
        self.score_instance = score_computation(G, opinions_dict, self.out_degrees)
        self.score_bank = None
    def score_function_manager(self, method, nodes, random_seed=42):
        scores = {}
        if self.score_bank is None or len(self.score_bank) != len(self.opinions_dict):
            self.score_bank = {}
            if method == "Degree":
                for node in range(len(self.opinions_dict)):
                    self.score_bank[node] = self.score_instance.out_degrees[node]
            elif method == "ABS":
                for node in range(len(self.opinions_dict)):
                    self.score_bank[node] = abs(self.opinions_dict[node])
            elif method == "JiHom":
                for node in range(len(self.opinions_dict)):
                    self.score_bank[node] = self.score_instance.JiHom(node)
            elif method == "Hom":
                for node in range(len(self.opinions_dict)):
                    self.score_bank[node] = self.score_instance.Hom(node)
            elif method == "APR":
                for node in range(len(self.opinions_dict)):
                    self.score_bank[node] = (
                        self.score_instance.APR_helper_score_computation(node)
                    )
            elif method == "Entropy":
                for node in range(len(self.opinions_dict)):
                    self.score_bank[node] = (
                        self.score_instance.entropy_score_computation(node)
                    )
            elif method == "EntropyPure":
                for node in range(len(self.opinions_dict)):
                    self.score_bank[node] = (
                        self.score_instance.pure_entropy_score_computation(node)
                    )
            elif method == "CC":
                for node in range(len(self.opinions_dict)):
                    self.score_bank[node] = self.score_instance.CC(node)
            elif method == "Random":
                random.seed(random_seed)
                for node in range(len(self.opinions_dict)):
                    self.score_bank[node] = random.random()
            else:
                raise ValueError(f"{method} is not defined")
        for node in nodes:
            try:
                scores[node] = self.score_bank[node]
            except:
                print(len(self.score_bank), len(self.opinions_dict))
                raise ValueError("node missing from score_bank")
        return scores
        
    def ScoreBasedExpansion(self, G, seed, score_method, thresholds, max_expansion):
        MAX_HEAP_SIZE = 500
        EC = set()
        update_obj = update_all(self.E_V, G, self.opinions_dict)
        # Initialize best EC tracking
        best_EC = {"EC": set([seed]), "V_S": -1, "Delta_E": -1, "R": np.inf}
        hd = heapdict()
        hd[seed] = 0
        added = set([seed])
        opinion_of_seed = self.opinions_dict[seed]
        EC_tracker = []
        node_info = {}
        while max_expansion > 0 and hd:
            max_expansion -= 1
            node, _ = hd.popitem()
            EC, V_S, delta_E, R = update_obj.re_update_online(node)
            EC_tracker.append(node)
            neighbors = (
                set(
                    [
                        x
                        for x in G.neighbors(node, mode="OUT")
                        if self.opinions_dict[x] * opinion_of_seed > 0
                    ]
                )
                - added
            )
            added.update(neighbors)
            changed_nodes = set()
            candidates = set(hd.keys())
            # Heap retains at most MAX_HEAP_SIZE (K=500, Appendix D); stats are
            # refreshed within 2x that bound so boundary candidates stay accurate.
            if len(hd) < MAX_HEAP_SIZE * 2:
                for neighbor in neighbors:
                    neighbors_of_neighbor = self.out_neigbors[neighbor]
                    inside = len(neighbors_of_neighbor & EC)
                    outside = len(neighbors_of_neighbor) - inside
                    node_info[neighbor] = [inside, outside]
                    neighbors_of_neighbor = self.in_neigbors[neighbor] & candidates
                    for n_of_n in neighbors_of_neighbor:
                        if n_of_n in node_info:
                            node_info[n_of_n][0] += 1
                            node_info[n_of_n][1] -= 1
                    changed_nodes |= neighbors_of_neighbor
            temp_nodes = neighbors | changed_nodes
            scores_nodes = self.score_function_manager(score_method, list(temp_nodes))
            for x in temp_nodes:
                if node_info[x][1] == 0:
                    score = -node_info[x][0]
                else:
                    score = (
                        (-node_info[x][0] / node_info[x][1])
                        * scores_nodes[x]
                        * abs(self.opinions_dict[x])
                    )
                if len(hd) < MAX_HEAP_SIZE:
                    hd[x] = score
                elif score > hd.peekitem()[1]:
                    hd.popitem()
                    hd[x] = score
            if len(EC) >= 10:
                # len(EC)==10 accepts the first size-10 candidate regardless of
                # thresholds; invalid chambers are filtered later in seed_handler.
                if len(EC) == 10 or (
                    V_S <= thresholds["theta_V"]
                    and delta_E >= thresholds["theta_E"]
                    and R <= best_EC["R"]
                ):
                    best_EC = {"EC": EC.copy(), "V_S": V_S, "Delta_E": delta_E, "R": R}
        best_EC["EC"] = EC_tracker[: len(best_EC["EC"])].copy()
        return best_EC
    def init_APPR(self):
        self.features_all = np.array(
            [[self.opinions_dict[i]] for i in range(self.G.vcount())]
        )
        self.expansion_APPR_model = AttriPPR(
            num_nodes=self.G.vcount(),
            edges=self.edges,
            features=self.features_all,
            a=0.15,
            b=0.1,
            tol=1e-12,
            max_iter=1000,
            init="v",
        )
    def expansion_APPR(self, G, seed, thresholds, max_appr_nodes):
        if self.features_all is None or self.expansion_APPR_model is None:
            self.init_APPR()
        best_EC = {
            "EC": set([seed]),
            "V_S": 0,
            "Delta_E": abs(self.E_V - self.opinions_dict[seed]),
            "R": np.inf,
        }
        scores_np = self.expansion_APPR_model.compute_pi(seeds=seed)
        nodes_sorted = np.argsort(-scores_np)
        if nodes_sorted[0] != seed:
            nodes_sorted = [seed] + [n for n in nodes_sorted if n != seed]
        update_obj = update_all(self.E_V, G, self.opinions_dict)
        opinion_of_seed = self.opinions_dict[seed]
        for node in nodes_sorted[
            :max_appr_nodes
        ]:  # cap the number of ranked nodes considered for expansion
            if self.opinions_dict[node] * opinion_of_seed <= 0:
                continue
            EC, V_S, delta_E, R = update_obj.re_update_online(node)
            if len(EC) >= 10:
                # len(EC)==10 accepts the first size-10 candidate regardless of
                # thresholds; invalid chambers are filtered later in seed_handler.
                if (
                    len(EC) == 10
                    or V_S <= thresholds["theta_V"]
                    and delta_E >= thresholds["theta_E"]
                    and R <= best_EC["R"]
                ):
                    best_EC = {"EC": EC.copy(), "V_S": V_S, "Delta_E": delta_E, "R": R}
        return best_EC
    def get_legit_nodes(self, degree_dict, extreme_ratio, degree_ratio):
        n = len(self.opinions_dict)
        if extreme_ratio == 1:
            legit_extreme = set(self.opinions_dict.keys())
        else:
            num_extreme = max(1, int(n * extreme_ratio))
            sorted_by_extreme = sorted(
                self.opinions_dict.items(), key=lambda x: abs(x[1]), reverse=True
            )
            cutoff_value = abs(sorted_by_extreme[num_extreme - 1][1])
            legit_extreme = {
                node
                for node, val in self.opinions_dict.items()
                if abs(val) >= cutoff_value
            }
        if degree_ratio == 1:
            legit_degree = set(degree_dict.keys())
        else:
            num_degree = max(1, int(n * degree_ratio))
            sorted_by_degree = sorted(
                degree_dict.items(), key=lambda x: x[1], reverse=True
            )
            cutoff_degree = sorted_by_degree[num_degree - 1][1]
            legit_degree = {
                node for node, val in degree_dict.items() if val >= cutoff_degree
            }
        if extreme_ratio == 1:
            return legit_degree
        if degree_ratio == 1:
            return legit_extreme
        legit_nodes = legit_extreme & legit_degree
        return legit_nodes
    
    def seed_handler(
        self,
        score_method,
        expansion_method,
        thresholds,
        seed_ratio,
        extreme_ratio=1,
        degree_ratio=1,
    ):
        assert 0 < seed_ratio <= 1, "not legal ratio"
        assert 0 < extreme_ratio <= 1, "not extreme ratio"
        assert 0 < degree_ratio <= 1, "not degree ratio"
        all_nodes = list(range(self.G.vcount()))
        legit_number = int(len(all_nodes) * seed_ratio)
        legit_nodes = self.get_legit_nodes(
            self.out_degrees, extreme_ratio, degree_ratio
        )
        # legit_nodes = max(10,legit_nodes)
        scores_dict = self.score_function_manager(
            score_method, list(legit_nodes)
        ).copy()
        max_expantion = ceil(10 * (len(self.nodes) ** 0.5) + 1)
        bestBestEC = {"R": np.inf, "EC": [], "V_S": -1, "Delta_E": -1}
        pbar = tqdm(total=legit_number, desc="Expanding Process")
        while scores_dict:
            pbar.update(1)
            if legit_number <= 0:
                break
            legit_number -= 1
            seed = max(scores_dict, key=scores_dict.get)
            if expansion_method == "APPR":
                bestEC = self.expansion_APPR(self.G, seed, thresholds, max_expantion)
            elif expansion_method == "ScoreBasedExpansion":
                bestEC = self.ScoreBasedExpansion(
                    self.G, seed, score_method, thresholds, max_expantion
                )
            else:
                raise ValueError(f"{expansion_method} is not defined.")
            if (
                len(bestEC["EC"]) >= 10
                and bestEC["R"] <= bestBestEC["R"]
                and bestEC["V_S"] <= thresholds["theta_V"]
                and bestEC["Delta_E"] >= thresholds["theta_E"]
            ):
                bestBestEC = bestEC.copy()
            scores_dict.pop(seed, None)
            if expansion_method == "ScoreBasedExpansion":
                pop_number = len(bestEC["EC"])
                for node in bestEC["EC"][:pop_number]:
                    scores_dict.pop(node, None)
            else:
                for node in list(bestEC["EC"]):
                    scores_dict.pop(node, None)
        pbar.close()
        return bestBestEC
    

    def seed_handler_abl_related(
        self,
        score_method,
        expansion_method,
        thresholds,
        seed_ratios,
        extreme_ratio=1,
        degree_ratio=1,
    ):
        assert 0 < min(seed_ratios) and max(seed_ratios) <= 1, "not legal ratio"
        assert 0 < extreme_ratio <= 1, "not extreme ratio"
        assert 0 < degree_ratio <= 1, "not degree ratio"
        all_nodes = list(range(self.G.vcount()))

        legit_numbers = [int(len(all_nodes) * seed_ratio) for seed_ratio in seed_ratios]
        print('legit_numbers:',legit_numbers)
        legit_number = max(legit_numbers)

        legit_nodes = self.get_legit_nodes(
            self.out_degrees, extreme_ratio, degree_ratio
        )
        # legit_nodes = max(10,legit_nodes)
        scores_dict = self.score_function_manager(
            score_method, list(legit_nodes)
        ).copy()
        max_expantion = ceil(10 * (len(self.nodes) ** 0.5) + 1)
        bestBestEC = {"R": np.inf, "EC": [], "V_S": -1, "Delta_E": -1}
        pbar = tqdm(total=legit_number, desc="Expanding Process")
        to_return = {}
        checked_for_ABL = 0

        start = time.perf_counter()

        while scores_dict:
            pbar.update(1)
            if legit_number <= 0:
                break
            legit_number -= 1
            seed = max(scores_dict, key=scores_dict.get)
            checked_for_ABL += 1
            if expansion_method == "APPR":
                bestEC = self.expansion_APPR(self.G, seed, thresholds, max_expantion)
            elif expansion_method == "ScoreBasedExpansion":
                bestEC = self.ScoreBasedExpansion(
                    self.G, seed, score_method, thresholds, max_expantion
                )
            else:
                raise ValueError(f"{expansion_method} is not defined.")
            if (
                len(bestEC["EC"]) >= 10
                and bestEC["R"] <= bestBestEC["R"]
                and bestEC["V_S"] <= thresholds["theta_V"]
                and bestEC["Delta_E"] >= thresholds["theta_E"]
            ):
                bestBestEC = bestEC.copy()
                bestBestEC["EC"] = [-1]
            
            scores_dict.pop(seed, None)
            if expansion_method == "ScoreBasedExpansion":
                pop_number = len(bestEC["EC"])
                for node in bestEC["EC"][:pop_number]:
                    scores_dict.pop(node, None)
            else:
                for node in list(bestEC["EC"]):
                    scores_dict.pop(node, None)
            
            
            
                
            end = time.perf_counter()
            time_until_now = f"{end - start:.2f}"
            to_return[checked_for_ABL] = {'bestECforthisseed':bestBestEC.copy(),'timeuntilnow':time_until_now[:]}
            
        pbar.close()
        end = time.perf_counter()
        time_until_now = f"{end - start:.2f}"
        to_return[-1] = {'bestECforthisseed':bestBestEC.copy(),'timeuntilnow':time_until_now[:]}
      
        return to_return
    



    def seed_handler2(
        self,
        score_method,
        expansion_method,
        thresholds,
        seed_ratio,
        extreme_ratio=1,
        degree_ratio=1,
    ):
        assert 0 < seed_ratio <= 1, "not legal ratio"
        assert 0 < extreme_ratio <= 1, "not extreme ratio"
        assert 0 < degree_ratio <= 1, "not degree ratio"
        all_nodes = list(range(self.G.vcount()))
        legit_number = int(len(all_nodes) * seed_ratio)
        legit_nodes = self.get_legit_nodes(
            self.out_degrees, extreme_ratio, degree_ratio
        )
        scores_dict = self.score_function_manager(
            score_method, list(legit_nodes)
        ).copy()
        max_expantion = ceil(10 * (len(self.nodes) ** 0.5) + 1)
        bestBestEC = {"R": np.inf, "EC": [], "V_S": -1, "Delta_E": -1}
        all_Rs = []
        # pbar = tqdm(total=legit_number, desc="Expanding Process")
        start = time.perf_counter()
        while scores_dict:
            # pbar.update(1)
            if legit_number <= 0:
                break
            legit_number -= 1
            seed = max(scores_dict, key=scores_dict.get)
            if expansion_method == "APPR":
                bestEC = self.expansion_APPR(self.G, seed, thresholds, max_expantion)
            elif expansion_method == "ScoreBasedExpansion":
                bestEC = self.ScoreBasedExpansion(
                    self.G, seed, score_method, thresholds, max_expantion
                )
            else:
                raise ValueError(f"{expansion_method} is not defined.")
            if (
                len(bestEC["EC"]) >= 10
                and bestEC["R"] <= bestBestEC["R"]
                and bestEC["V_S"] <= thresholds["theta_V"]
                and bestEC["Delta_E"] >= thresholds["theta_E"]
            ):
                bestBestEC = bestEC.copy()
            scores_dict.pop(seed, None)
            if expansion_method == "ScoreBasedExpansion":
                # pop_number = max(10,int(min(len(bestEC['EC']),sqrt(len(self.nodes)))/2))
                pop_number = len(bestEC["EC"])
                for node in bestEC["EC"][:pop_number]:
                    scores_dict.pop(node, None)
            else:
                for node in list(bestEC["EC"]):
                    scores_dict.pop(node, None)
            end = time.perf_counter()
            exe_time = f"{end - start:.2f}"
            start = time.perf_counter()
            all_Rs.append({"R": bestBestEC["R"], "time": exe_time})
        # pbar.close()
        return all_Rs

def JECHO_main(
    dataset_name,
    dataset_name_prime,
    score_methods,
    expansion_methods,
    pre_path,
    thresholds,
    random_seed=42,
    seed_ratio=None,
):
    G, _ = read_directed_iGraph_from_file(
        filepath=f"{pre_path}{dataset_name}_edges.txt"
    )
    del _
    opinions_dict = pickle.load(open(pre_path + f"{dataset_name_prime}.pkl", "rb"))
    expansion_instance = expansion_class(G, opinions_dict)
    results = {}
    # Resolve seed ratio once per dataset without mutating the argument.
    dataset_seed_ratio = seed_ratio
    if dataset_seed_ratio is None:
        dataset_seed_ratio = 0.005 if "SBM" in dataset_name else 0.001
    for expansion_method, score_method in product(expansion_methods, score_methods):
        expansion_instance.score_bank = None
        random.seed(random_seed)
        if score_method == "JiHom":
            if "SBM" in dataset_name:
                degree_ratio = 0.4
            else:
                degree_ratio = 0.7
            extreme_ratio = 0.1
        elif score_method == "Random":
            degree_ratio = 1
            extreme_ratio = 1
        else:
            degree_ratio = 1
            extreme_ratio = 0.1
        start = time.perf_counter()
        bestEC = expansion_instance.seed_handler(
            score_method,
            expansion_method,
            thresholds,
            dataset_seed_ratio,
            extreme_ratio,
            degree_ratio,
        )
        try:
            bestEC["LenEC"] = len(bestEC["EC"])
            bestEC["EC"] = bestEC["EC"].copy()
        except:
            pass
        for k, v in bestEC.items():
            try:
                bestEC[k] = round(v, 3)
            except:
                pass
        end = time.perf_counter()
        bestEC["time"] = f"{end - start:.2f}"
        results[(expansion_method, score_method)] = bestEC.copy()
    # bestEC["EC"] = ""
    return results


def JECHO_main2(
    dataset_name,
    dataset_name_prime,
    score_methods,
    expansion_methods,
    pre_path,
    thresholds,
    seed_ratios,
    random_seed=42,
): # For ABL
    G, _ = read_directed_iGraph_from_file(
        filepath=f"{pre_path}{dataset_name}_edges.txt"
    )
    del _
    opinions_dict = pickle.load(open(pre_path + f"{dataset_name_prime}.pkl", "rb"))
    expansion_instance = expansion_class(G, opinions_dict)
    results = {}
    for expansion_method, score_method in product(expansion_methods, score_methods):
        print(f"GR file,{expansion_method},{score_method}")
        expansion_instance.score_bank = None
        random.seed(random_seed)
        
        if score_method == "JiHom":
            if "SBM" in dataset_name:
                degree_ratio = 0.4
            else:
                degree_ratio = 0.7
            extreme_ratio = 0.1
        elif score_method == "Random":
            degree_ratio = 1
            extreme_ratio = 1
        else:
            degree_ratio = 1
            extreme_ratio = 0.1
        start = time.perf_counter()

        bestEC = expansion_instance.seed_handler_abl_related(
            score_method,
            expansion_method,
            thresholds,
            seed_ratios,
            extreme_ratio,
            degree_ratio,
        )
        
        end = time.perf_counter()
        bestEC["time"] = f"{end - start:.2f}"
        results[(expansion_method, score_method)] = bestEC.copy()
    
    return results




def main_abl_study(
    dataset_name,
    dataset_name_prime,
    score_methods,
    expansion_methods,
    pre_path,
    thresholds,
    random_seed=42,
):
    pre_path = "../EchoChambersDatasets/"
    G, _ = read_directed_iGraph_from_file(
        filepath=f"{pre_path}{dataset_name}_edges.txt"
    )
    del _
    opinions_dict = pickle.load(open(pre_path + f"{dataset_name_prime}.pkl", "rb"))
    expansion_instance = expansion_class(G, opinions_dict)
    results = {}
    for expansion_method, score_method in product(expansion_methods, score_methods):
        expansion_instance.score_bank = None
        random.seed(random_seed)
        if score_method == "JiHom":
            if "SBM" in dataset_name:
                degree_ratio = 0.4
            else:
                degree_ratio = 0.7
            extreme_ratio = 0.1
        elif score_method == "Random":
            degree_ratio = 1
            extreme_ratio = 1
        else:
            degree_ratio = 1
            extreme_ratio = 0.1
        seed_ratios = [
            0.01
        ]  # list(np.arange(0.001, 0.011, 0.001))# + list(np.arange(0.01, 0.055, 0.01))
        for seed_ratio in tqdm(
            seed_ratios, desc=f"seed ratio , process: {expansion_method} {score_method}"
        ):
            Rs = expansion_instance.seed_handler2(
                score_method,
                expansion_method,
                thresholds,
                seed_ratio,
                extreme_ratio,
                degree_ratio,
            )
            results[(expansion_method, score_method, seed_ratio)] = Rs.copy()
    return results
