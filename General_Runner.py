import pickle
import random
import time
from math import inf
import numpy as np
from EchoGAE import ECD as GAE_ECD
from SEDA import ECD as SEDA_ECD
from JECHO import JECHO_main,  JECHO_main2

from utils import compute_stats_and_D, read_directed_iGraph_from_file, save_pickle
import multiprocessing as mp
def _run_with_queue(q, func, args):
    try:
        res = func(*args)
        q.put({"status": "ok", "result": res})
    except MemoryError:
        q.put({"status": "out_of_memory"})
    except Exception as e:
        q.put({"status": "error", "error": str(e)})

def isolated_run(func, args, timeout_sec):
    q = mp.Queue()
    p = mp.Process(target=_run_with_queue, args=(q, func, args))
    p.start()
    p.join(timeout_sec)

    if p.is_alive():
        p.terminate()
        return {"status": "time_limit"}

    if q.empty():
        return {"status": "killed_or_oom"}

    return q.get()
    
    
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

def tester(ECD_model, test_id, dataset_name, score_methods = None, expansion_methods = None, suffixes = None, pre_path = "../EchoChambersDatasets/", seed_ratio = None):
    
    seed_id = 42
    
    if "SBM" in dataset_name:
        dataset_name_plus = f"{test_id}_{dataset_name}"
        thresholds = {"theta_V": 0.05, "theta_E": 0.3, "theta_R": 2}
    else:
        dataset_name_plus = f"{dataset_name}"
        thresholds = {"theta_V": 0.075, "theta_E": 0.3, "theta_R": 2}

    results = {}
    for suffix in suffixes:
        dataset_name_prime = f"{test_id}_{dataset_name}{suffix}"

        if ECD_model == 'JECHO':
#            result = JECHO_main(
#                            dataset_name_plus,
#                            dataset_name_prime,
#                            score_methods,
#                            expansion_methods,
#                            pre_path,
#                            thresholds,
#                            seed_id + test_id,
#                            seed_ratio
#                        )
            
            result = isolated_run(
                JECHO_main,
                (
                    dataset_name_plus,
                    dataset_name_prime,
                    score_methods,
                    expansion_methods,
                    pre_path,
                    thresholds,
                    seed_id + test_id,
                    seed_ratio,
                ),
                timeout_sec=12 * 60 * 60,
            )
            
        elif ECD_model == "SEDA":
#            result = SEDA_helper(
#                            seed_id, dataset_name_plus, dataset_name_prime, pre_path, thresholds
#                        )
          result = isolated_run(
              SEDA_helper,
              (seed_id, dataset_name_plus, dataset_name_prime, pre_path, thresholds),
              timeout_sec=12 * 60 * 60,
          )
          
          
        elif ECD_model == "GAE":
#            result,_ = GAE_helper(
#                            seed_id, dataset_name_plus, dataset_name_prime, pre_path, thresholds
#                        )
#                        
            result = isolated_run(
                GAE_helper,
                (seed_id, dataset_name_plus, dataset_name_prime, pre_path, thresholds),
                timeout_sec=12 * 60 * 60,
            )
            
            
        else:
            raise ValueError(f"{ECD_model} is not defined.")
        
#        results[(suffix,ECD_model)] = result.copy()
        results[(suffix,ECD_model)] = result
        
    return results


def tester2(ECD_model, test_id, dataset_name, seed_ratios, score_methods = None, expansion_methods = None, suffixes = None, pre_path = "../EchoChambersDatasets/"): # Foa ABL
    
    seed_id = 42
    if "SBM" in dataset_name:
        dataset_name_plus = f"{test_id}_{dataset_name}"
        thresholds = {"theta_V": 0.05, "theta_E": 0.3, "theta_R": 2}
    else:
        dataset_name_plus = f"{dataset_name}"
        thresholds = {"theta_V": 0.075, "theta_E": 0.3, "theta_R": 2}

    results = {}
    for suffix in suffixes:
        dataset_name_prime = f"{test_id}_{dataset_name}{suffix}"

        if ECD_model == 'JECHO':
            result = JECHO_main2(
                    dataset_name_plus,
                    dataset_name_prime,
                    score_methods,
                    expansion_methods,
                    pre_path,
                    thresholds,
                    seed_ratios,
                    random_seed=seed_id + test_id,
            )
            

        else:
            raise ValueError(f"{ECD_model} is not defined.")
        
        results[(suffix,ECD_model)] = result
        
    return results


def main(datasets,test_ids,ECD_models,score_methods= None, expansion_methods= None, suffixes = None, pre_path_results = '.', pre_path = '.', seed_ratio = None):
    for dataset_name in datasets:
        all_results = {'JECHO':[],'GAE':[],'SEDA':[]}
        for test_id in test_ids:
            for ECD_model in ECD_models:
                x = tester(ECD_model, test_id, dataset_name, score_methods, expansion_methods, suffixes, pre_path, seed_ratio)
                all_results[ECD_model].append([dataset_name,test_id,x.copy()])
        save_pickle(pre_path_results, name = f"MainExperiments-{dataset_name}.pkl", obj = all_results)
        



def main2(datasets,test_ids,ECD_models,seed_ratios,score_methods= None, expansion_methods= None, suffixes = None, pre_path_results = '.', pre_path = '.'): # For ABL studies
    for dataset_name in datasets:
        all_results = {'JECHO':[]}
        for test_id in test_ids:
            for ECD_model in ECD_models:
                x = tester2(ECD_model, test_id, dataset_name, seed_ratios, score_methods, expansion_methods, suffixes, pre_path)
                all_results[ECD_model].append([dataset_name,test_id,x.copy()])
        save_pickle(pre_path_results, name = f"MainExperiments-{dataset_name}.pkl", obj = all_results)