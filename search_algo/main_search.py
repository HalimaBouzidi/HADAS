import os, sys
os.environ['OMP_NUM_THREADS'] = '1' # speed up
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import yaml
from argparse import ArgumentParser
import numpy as np
from time import time
from multiprocessing import cpu_count
from nsga2 import NSGA2
from optimize import minimize
from pymoo.factory import get_crossover, get_mutation, get_sampling
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from problems.common import build_problem
from problems.search_space import SearchSpace
from external import lhs
from data_export import DataExport
from utils import calc_hypervolume, get_result_dir, setup_logger

def get_args():
    parser = ArgumentParser()

    parser.add_argument('--problem', type=str, default='hw_nas', help='optimization problem')
    parser.add_argument('--algo', type=str, default='nsga2', help='optimization problem')
    parser.add_argument('--pop-init-method', type=str, choices=['nds', 'random', 'lhs'], default='lhs', help='method to init population')
    parser.add_argument('--n-init-sample', type=int, default=10, help='number of initial design samples')
    parser.add_argument('--pop-size', type=int, default=100, help='Population size of the generation')
    parser.add_argument('--nb-gen', type=int, default=5, help='Number of generations')
        
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--subfolder', type=str, default='default',help='subfolder name for storing results')
    parser.add_argument('--exp-name', type=str, default=None, help='custom experiment name')
    parser.add_argument('--log-to-file', default=False, action='store_true', help='log output to file rather than print by stdout')
    parser.add_argument('--search-resume', default=False, help='File that stores the archive of explored configurations')
    parser.add_argument('--offspring', default=False, help='File that stores the archive of explored configurations')
    parser.add_argument('--n-process', type=int, default=1, help='number of processes to be used for parallelization')


    args = parser.parse_args()
    return args

def get_ref(problem_name):

    ref_points = None
    name_ = problem_name
    parm = name_.split('_')
    if(len(parm)>1):
        platform = parm[0]
        cnn = parm[1]

        # load ref_points file
        file = open('../ref_points.csv', 'r')
        csv_reader = csv.reader(file)
        lines = []
        for row in csv_reader:
            if(row[0] == cnn  and row[1] == platform):
                ref_points = np.array([float(row[2]), float(row[3])])
                break
        file.close()
    
    return ref_points

def save_args(args):
    '''
    Save arguments to yaml file
    '''
    result_dir = get_result_dir(args)
    args_path = os.path.join(result_dir, 'args.yml')
    os.makedirs(os.path.dirname(args_path), exist_ok=True)
    with open(args_path, 'w') as f:
        yaml.dump(args, f, default_flow_style=False, sort_keys=False)


class hw_dynamic_nas():

    def __init__(self, kwargs):

        self.problem = kwargs.problem
        self.algo = kwargs.algo
        self.pop_size = kwargs.pop_size
        self.nb_gen = kwargs.nb_gen

        self.n_init_sample = kwargs.n_init_sample 
        self.pop_init_method = kwargs.pop_init_method

        self.seed = kwargs.seed
        self.subfolder = kwargs.subfolder
        self.exp_name = kwargs.exp_name
        self.log_to_file = kwargs.log_to_file
        self.search_resume = kwargs.search_resume

        self.n_var = None
        self.n_obj = None
        self.ref_point = None
        self.logger = None
        self.n_process = kwargs.n_process

    def search(self):
        np.random.seed(self.seed)
        ss = SearchSpace()
        archive = []

        # build problem, get initial samples
        problem, true_pfront, X_init, Y_init = build_problem(self.problem, self.n_var, self.n_obj, self.n_init_sample, self.n_process)
        self.n_var, self.n_obj, self.algo = problem.n_var, problem.n_obj, 'nsga2'

        print(problem)
                
        for member in zip(X_init, Y_init[:, 0], Y_init[:, 1]):
            archive.append(member)

        # get reference point
        if self.ref_point is None:
            self.ref_point = np.array([100, 2000]) # set the worst reference point

        # initialize data exporter
        exporter = DataExport(X_init, Y_init, self)

        # 1) Sampling: population initialization method
        if self.pop_init_method == 'lhs':
            sampling = get_sampling("int_lhs")
        elif self.pop_init_method == 'random':
            sampling = get_sampling("int_random")
        elif self.pop_init_method == 'nds':
            sorted_indices = NonDominatedSorting().do(Y_init)
            sampling = X_init[np.concatenate(sorted_indices)][:self.pop_size]
            if len(sampling) < self.pop_size:
                rest_sampling = lhs(X_init.shape[1], self.pop_size - len(sampling))
                sampling = np.vstack([sampling, rest_sampling])
        else:
            raise NotImplementedError
        
        # 2) Crossover: recombination operation between individuals
        crossover = get_crossover('int_k_point', prob=0.8, n_points=5)

        # 3) Mutation: local neighberhood generation for a single individual real_pm  
        mutation = get_mutation('int_pm', prob=0.2) # low=ss.bounds_arch()[0], up=ss.bounds_arch()[1], )

        # initialize evolutionary algorithm
        ea_algorithm = NSGA2(
            pop_size=self.pop_size,  # initialize with current nd archs
            crossover=crossover,
            mutation=mutation,
            sampling=sampling,
            eliminate_duplicates=True,
            seed=self.seed)
    
        # Run the optimization with the evolutionary algorithm
        res = minimize(problem, ea_algorithm, ('n_gen', self.nb_gen), save_history=True)

        # Get the obtained results (the generated solutions during the search and compute the Pareto front)
        X_history = np.array([algo.pop.get('X') for algo in res.history])
        Y_history = np.array([algo.pop.get('F') for algo in res.history])

        # update data exporter
        for X_next, Y_next in zip(X_history, Y_history):
            exporter.update(X_next, Y_next)

        # export all result to csv
        exporter.write_csvs()
        if true_pfront is not None:
            exporter.write_truefront_csv(true_pfront)

        # statistics
        final_hv = calc_hypervolume(exporter.Y, exporter.ref_point)
        print('========== Result ==========')
        print('Total evaluations: %d, hypervolume: %.4f\n' % (self.nb_gen * self.pop_size, final_hv))

        # close logger
        if self.logger is not None:
            self.logger.close()

def main(args):
    engine = hw_dynamic_nas(args)
    # save arguments and setup logger
    save_args(args)
    engine.logger = setup_logger(args)
    engine.search()
    return

if __name__ == '__main__':

    cfgs = get_args()
    main(cfgs)


# Should think about the evaluation parallelization
# Should think about how to incorporate the second optimization stage