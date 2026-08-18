from General_Runner import main as main_runner
from math import ceil,sqrt
from partition_based_opinion_assignment import (
    check_isolated_nodes, large_make_sbm_graph_precalculated,
    make_sbm_graph_custom_variance, save_edges_to_txt, do_opinion_assignment,SBM_data_generation_API)


def generate_different_graphs(prefix,seed_id,N,pre_path_data,modularity,number_of_testcases):

    if modularity == 1:
        p = 1
        q = 0
    else:
        p = 1e-3
        q = 1e-1 

    sbm_blocks = ceil(sqrt(N*1000))
    flag_isolated_node = True
    datasetname = f"SBM_{N}"

    
    for i in range(number_of_testcases):
        flag_isolated_node = True
        while(flag_isolated_node):
            if N<50:
                G,_,p,q = make_sbm_graph_custom_variance(b = sbm_blocks,p=p,q=q, seed=seed_id+i, modularity_th=modularity)
            else:
                G, _, p, q = large_make_sbm_graph_precalculated(b=sbm_blocks, target_modularity=modularity, seed=seed_id+i)
            
            flag_isolated_node = check_isolated_nodes(G)
            seed_id+=1   

        save_edges_to_txt(G, f"{pre_path_data}{i}_{modularity}_{datasetname}_edges.txt")

        
        do_opinion_assignment(G,pre_path_data,f"{modularity}_{datasetname}", k_filtering = 5 , 
                            k_extreme = 5, pa_filtering = 0.5, pa_extreme = 0.9, 
                            ratio_of_extreme = 0.01, 
                            number_of_filtering_iteration = 5,  seed = seed_id+i,
                            number_of_partitions_for_opinion = int(sbm_blocks * 2.5),
                            sigma_of_mu = 0.2, sigma_of_partition=1, silly_noise_ratio=0.03, prefix=f"{i}")



if __name__ == "__main__":
    pre_path_results = "EchoChambersResultsModulation/"
    pre_path_data = "EchoChambersDatasetsModulation/"
    
    #--------------------#
    ### Possibilities: ###
    #--------------------#
    '''
    score_methods = [
        "JiHom",
        "Entropy",
        "EntropyPure",
        "Hom",
        "APR",
        "CC",
        "Degree",
        "ABS",
        "Random",
    ][:]
    expansion_methods = ["ScoreBasedExpansion", "APPR"][:]
    suffixes = ["_fj", "_filtering", "_extremes"][:]
    '''

    modularity_values = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9][6:]

    
    N = 20
    
    for m in modularity_values:
        generate_different_graphs(prefix = '',seed_id = int(m*100),N = N,pre_path_data = pre_path_data,modularity = m, number_of_testcases=10)
    
    print("Datasets are ready.")

    
    score_methods = ["JiHom",
                    "Entropy",
                    "Hom",
                    "APR",
                    "Degree"]
    
    expansion_methods = ["ScoreBasedExpansion", "APPR"]

    suffixes = ["_fj", "_filtering", "_extremes"][:]
    ECD_models = ['JECHO']


    for m in modularity_values:
        datasets = [f"{m}_SBM_{N}"]


        seed_id = 42

        ECD_models = ['JECHO','SEDA','GAE']

        main_runner(datasets,list(range(10)),ECD_models,score_methods, expansion_methods, suffixes, pre_path_results, pre_path_data)
        