from partition_based_opinion_assignment import real_world_data_generation_API, SBM_data_generation_API

def simple_helper(datasets,number):
    for dataset_name in datasets:
        print("\n\n", dataset_name, ":\n")
        for test_id in range(number):
            print(f"\n\n test number {test_id} and random seed {seed_id+test_id}")
            if "SBM" in dataset_name:
                N = int(dataset_name[4:])
                SBM_data_generation_API(
                    test_id, seed_id + test_id, dataset_name, N, pre_path
                )
            else:
                real_world_data_generation_API(
                    test_id, seed_id + test_id, dataset_name, pre_path
                )

if __name__ == "__main__":
    seed_id = 42
    pre_path = "EchoChambersDatasets/"
  
#    datasets = [f'SBM_{N}' for N in [5,10,50,100]][:]
#    simple_helper(datasets,10)
#
#    datasets = ["facebook","twitter","git","lastfm"]
#    simple_helper(datasets,10)
#    
    datasets = ["pokec", "soc"][:]  
    simple_helper(datasets,3)