import numpy as np
from pymoo.factory import get_from_list
from problems import *

def get_problem_options():
    problems = [
        ('hw_nas', HW_NAS)
    ]
    return problems

def get_problem(name, platform, *args, d={}, **kwargs):
    args = [platform]
    return get_from_list(get_problem_options(), name.lower(), args, {**d, **kwargs})

def generate_initial_samples(problem, n_sample):
    '''
    Generate feasible initial samples.
    Input:
        problem: the optimization problem
        n_sample: number of initial samples
    Output:
        X, Y: initial samples (design parameters, performances)
    '''

    if(n_sample != 0):
        X_feasible = np.zeros((n_sample, problem.n_var), dtype=object)
        Y_feasible = np.zeros((n_sample, problem.n_obj), dtype=object)

        init_solutions = problem.initialize_pop(n_sample)
        for i,config in enumerate(init_solutions):
            X_feasible[i] = problem.encode_sample(config)
        Y_feasible = problem.evaluate(X_feasible, return_values_of=['F'])
        return X_feasible, Y_feasible
    else:
        return np.array([]), np.array([])

def build_problem(name, n_var, n_obj, n_init_sample, n_process=1):
    '''
    Build optimization problem from name, get initial samples
    Input:
        name: name of the problem
        n_var: number of design variables
        n_obj: number of objectives
        n_init_sample: number of initial samples
        n_process: number of parallel processes
    Output:
        problem: the optimization problem
        X_init, Y_init: initial samples
        pareto_front: the true pareto front of the problem (if defined, otherwise None)
    '''

    # build problem
    try:
        problem = get_problem(name, platform='')
    except:
        raise NotImplementedError('problem not supported yet!')
    try:
        pareto_front = problem.pareto_front()
    except:
        pareto_front = None

    # get initial samples
    X_init, Y_init = generate_initial_samples(problem, n_init_sample)

    return problem, pareto_front, X_init, Y_init