import time
import heapq
import numpy as np
import igraph as ig


def qpeeling_1d(graph, opinion, q=1.0, theta=0.0, z_steps=25):
    t0 = time.time()
    n = graph.vcount()
    nodes = list(range(n))

    c = np.array([opinion[v] * q for v in nodes])
    deg = graph.degree()
    deg_max = max(deg)
    c_sorted = np.unique(c)
    delta_min = np.min(np.diff(c_sorted)) if len(c_sorted) > 1 else 1.0
    z_L = 0.0
    z_R = 2.0 * deg_max / delta_min

    best_set = []
    best_density = -1

    for _ in range(z_steps):
        z = (z_L + z_R) / 2.0
        alive = set(nodes)
        cur_deg = {v: deg[v] for v in nodes}
        pq = []
        for v in nodes:
            score = cur_deg[v] + z * (c[v] - theta)
            heapq.heappush(pq, (score, v))
        best_local_set = []
        best_local_density = -1
        total_c = sum(c)
        total_edges = graph.ecount()
        while len(alive) > 1:
            while True:
                _, v = heapq.heappop(pq)
                if v in alive:
                    break
            alive.remove(v)

            total_c -= c[v]
            total_edges -= cur_deg[v]

            for u in graph.neighbors(v):
                if u in alive:
                    cur_deg[u] -= 1
                    score = cur_deg[u] + z * (c[u] - theta)
                    heapq.heappush(pq, (score, u))

            if len(alive) == 0:
                break

            avg_c = total_c / len(alive)
            if avg_c >= theta:
                density = total_edges / len(alive)
                if density > best_local_density:
                    best_local_density = density
                    best_local_set = list(alive)

        if best_local_density >= 0:
            z_R = z
            if best_local_density > best_density:
                best_density = best_local_density
                best_set = best_local_set
        else:
            z_L = z

    runtime = time.time() - t0
    return best_set, runtime


''' ### usage example ###
S_plus,  t_plus  = qpeeling_1d(G, opinion, q=+1, theta=0.0)
S_minus, t_minus = qpeeling_1d(G, opinion, q=-1, theta=0.0)

print("Q-DISCO(+):", len(S_plus), "nodes", "time:", t_plus)
print("Q-DISCO(-):", len(S_minus), "nodes", "time:", t_minus)
'''


import pickle
import random
import time
from math import inf
import numpy as np
from EchoGAE import ECD as GAE_ECD
from SEDA import ECD as SEDA_ECD
from JECHO import JECHO_main
from utils import compute_stats_and_D, read_directed_iGraph_from_file, save_pickle

'''
def SEDA_helper(seed_id, dataset_name, dataset_name_prime, pre_path, thresholds):
    random.seed(seed_id)
    np.random.seed(seed_id)
    G, UG = read_directed_iGraph_from_file(
        filepath=f"{pre_path}{dataset_name}_edges.txt"
    )
    del UG

    opinions = pickle.load(open(pre_path + f"{dataset_name_prime}.pkl", "rb"))

    start = time.perf_counter()
    echo_chambers, _, _ = SEDA_ECD(G, opinions, max_iterations=30)
    bestBestEC = {"R": inf, "EC": [], "V_S": -1, "Delta_E": -1}
    if echo_chambers:
        for chamber in echo_chambers:
            _, V_S, delta_E, R = compute_stats_and_D(G, chamber, opinions)
            if (
                len(chamber) >= 10
                and thresholds["theta_V"] >= V_S
                and thresholds["theta_E"] <= delta_E
                and R <= bestBestEC["R"]
            ):
                bestBestEC = {
                    "R": R,
                    "EC": chamber.copy(),
                    "LenEC": len(chamber),
                    "V_S": V_S,
                    "Delta_E": delta_E,
                }
    end = time.perf_counter()

    bestBestEC["time"] = f"{end - start:.2f}"
    return bestBestEC.copy()

def GAE_helper(seed_id, dataset_name, dataset_name_prime, pre_path, thresholds):
    random.seed(seed_id)
    np.random.seed(seed_id)
    G, UG = read_directed_iGraph_from_file(
        filepath=f"{pre_path}{dataset_name}_edges.txt"
    )
    bestEC, ECs = GAE_ECD(UG, G, dataset_name_prime, pre_path, thresholds)
    return bestEC, ECs

'''

def tester(test_id, dataset_name, suffixes, pre_path, seed_id = 42):
    random.seed(seed_id)
    np.random.seed(seed_id)

    
    if "SBM" in dataset_name:
        dataset_name_plus = f"{test_id}_{dataset_name}"
        thresholds = {"theta_V": 0.05, "theta_E": 0.3, "theta_R": 2}
    else:
        dataset_name_plus = f"{dataset_name}"
        thresholds = {"theta_V": 0.075, "theta_E": 0.3, "theta_R": 2}

    results = {}
    for suffix in suffixes:
        dataset_name_prime = f"{test_id}_{dataset_name}{suffix}"

        G, UG = read_directed_iGraph_from_file(
            filepath=f"{pre_path}{dataset_name_plus}_edges.txt"
        )
       

        opinions = pickle.load(open(pre_path + f"{dataset_name_prime}.pkl", "rb"))
        if isinstance(opinions, list):
            opinions = {i:opinions[i] for i in range(len(opinions))}

        start = time.perf_counter()

        bestset_p,_ = qpeeling_1d(UG, opinions, q=1.0, theta=0.0, z_steps=25)
        bestset_n,_ = qpeeling_1d(UG, opinions, q=-1.0, theta=0.0, z_steps=25)

        bestBestEC = {"R": inf, "EC": [], "V_S": -1, "Delta_E": -1}
        
        _, V_S, delta_E, R = compute_stats_and_D(G, list(bestset_p), opinions)
        if (
            len(list(bestset_p)) >= 10
            and thresholds["theta_V"] >= V_S
            and thresholds["theta_E"] <= delta_E
            and R <= bestBestEC["R"]
        ):
            bestBestEC = {
                "R": R,
                "EC": list(bestset_p).copy(),
                "LenEC": len(list(bestset_p)),
                "V_S": V_S,
                "Delta_E": delta_E,
            }

        if (
            len(list(bestset_n)) >= 10
            and thresholds["theta_V"] >= V_S
            and thresholds["theta_E"] <= delta_E
            and R <= bestBestEC["R"]
        ):
            bestBestEC = {
                "R": R,
                "EC": list(bestset_n).copy(),
                "LenEC": len(list(bestset_n)),
                "V_S": V_S,
                "Delta_E": delta_E,
            }

        end = time.perf_counter()

        bestBestEC["time"] = f"{end - start:.2f}"
        results[suffix] = bestBestEC.copy()
         
    return results.copy()


def main(datasets,test_ids,suffixes, pre_path_results, pre_path_data, seed_id):
    for dataset_name in datasets:
        all_results = {'QPeeling':[]}
        for test_id in test_ids:
            x = tester(test_id,dataset_name,suffixes,pre_path_data,seed_id)
            all_results['QPeeling'].append([dataset_name,test_id,x.copy()])
        save_pickle(pre_path_results, name = f"MainExperiments-{dataset_name}.pkl", obj = all_results)

if __name__ == "__main__":
    pre_path_results = "EchoChambersResultsQPeeling/"
    pre_path_data = "EchoChambersDatasets/"
    
    '''
    suffixes = ["_fj", "_filtering", "_extremes"]
    datasets = [f'SBM_{N}' for N in [5,10,50,100,1000]][:]
    datasets = ["facebook","twitter","git","lastfm","pokec", "soc"]  
    '''
    
    suffixes = ["_fj", "_filtering", "_extremes"]

#    datasets = [f'SBM_{N}' for N in [5,10,50,100]]
#    main(datasets,list(range(10)),suffixes, pre_path_results, pre_path_data, seed_id=42)

#    datasets = ["facebook","lastfm","git","twitter"] 
#    main(datasets,list(range(10)),suffixes, pre_path_results, pre_path_data, seed_id=42)
#
#
    datasets = ["pokec", "soc"] 
    main(datasets,list(range(3)),suffixes, pre_path_results, pre_path_data, seed_id=42)


               