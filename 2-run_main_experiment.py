from General_Runner import main as main_runner


if __name__ == "__main__":
    pre_path_results = "EchoChambersResultsMain/"
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

    score_methods = ['JiHom']
    expansion_methods = ["ScoreBasedExpansion", "APPR"]
    suffixes = ["_fj", "_filtering", "_extremes"][:]
    ECD_models = ['JECHO','SEDA','GAE']

#    datasets = [f'SBM_{N}' for N in [5,10,50,100]]
#    main_runner(datasets,list(range(10)),ECD_models,score_methods, expansion_methods, suffixes, pre_path_results, pre_path_data)

#    datasets = ["facebook","lastfm","git","twitter"][:] 
#    main_runner(datasets,list(range(10)),ECD_models,score_methods, expansion_methods, suffixes, pre_path_results, pre_path_data)

    
    datasets = ["pokec", "soc"][:] 
    main_runner(datasets,list(range(3)),ECD_models,score_methods, expansion_methods, suffixes, pre_path_results, pre_path_data) 