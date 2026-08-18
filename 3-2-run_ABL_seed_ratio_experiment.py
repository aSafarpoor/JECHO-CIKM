from General_Runner import main2 as main_runner


if __name__ == "__main__":
    pre_path_results = "EchoChambersResultsABLSeedRatio/"
    pre_path_data = "EchoChambersDatasets/"
    
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
    datasets = [f'SBM_{N}' for N in [5,10,50,100,1000]][:]
    datasets = ["facebook","twitter","git","lastfm","pokec", "soc"]  
    '''

    score_methods = ["JiHom",
                    "Entropy",
                    "Hom",
                    "APR",
                    "Degree"]
    
    expansion_methods = ["ScoreBasedExpansion"]
    suffixes = ["_fj", "_filtering", "_extremes"][:]
    ECD_models = ['JECHO']

    datasets = [f'SBM_{N}' for N in [5,10,50,100]][3:]
#    datasets = ["facebook","lastfm","git","twitter"][:] 

    seed_ratios = [i/100 for i in list(range(1,5))]
    for dataset in datasets:
        print(dataset)
        main_runner([dataset],list(range(10)),ECD_models,seed_ratios,score_methods, expansion_methods, suffixes, pre_path_results, pre_path_data)
      


               