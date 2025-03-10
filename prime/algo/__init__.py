from prime.algo.algo import register_algo_factory_func, algo_name_to_factory_func, algo_factory, Algo, PolicyAlgo, ValueAlgo, PlannerAlgo, HierarchicalAlgo, RolloutPolicy

# note: these imports are needed to register these classes in the global algo registry
from prime.algo.bc import BC, BC_DISCRETE, BC_Gaussian, BC_GMM, BC_VAE, BC_RNN, BC_RNN_GMM
