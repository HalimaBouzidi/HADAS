# plot comparison of hypervolume over all runs for all algorithms on all problems

import matplotlib.pyplot as plt
SMALL_SIZE = 8
MEDIUM_SIZE = 12
BIGGER_SIZE = 15
MAX_SIZE=18
plt.rc('font', family='Times New Roman', size=BIGGER_SIZE)
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MAX_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
import os, csv
import numpy as np
import pandas as pd
from argparse import ArgumentParser
from scipy.spatial.distance import cdist


def get_args():
    parser = ArgumentParser()
    parser.add_argument('--n-seed', type=int, default=10, help='number of different seeds')
    parser.add_argument('--subfolder', type=str, default='default', help='subfolder of result')
    parser.add_argument('--savefig', default=False, action='store_true', help='saving instead of showing the plot')
    parser.add_argument('--num-eval', type=int, default=200, help='number of evaluations')
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    problems = ['agx_alexnet', 'agx_mobilenetv2', 'agx_resnet50', 'agx_inceptionv3', 'agx_densenet121', 
    'tx2_alexnet', 'tx2_mobilenetv2', 'tx2_resnet50', 'tx2_inceptionv3', 'tx2_densenet121', 
    'nano_alexnet', 'nano_mobilenetv2', 'nano_resnet50', 'nano_inceptionv3', 'nano_densenet121']
    algos = {'nsga2': 'NSGA-II', 'moead': 'MOEA/D', 'usemo-ei': 'USeMO-EI', 'dgemo': 'DGEMO'}
    algos = {'usemo-ei': 'USeMO-EI', 'dgemo': 'DGEMO'}

    result_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'result')

    n_row, n_col = 3, 5
    fig, axes = plt.subplots(n_row, n_col, figsize=(18, 12))
    n_algo, n_seed, num_eval = len(algos), args.n_seed, args.num_eval
    colors = ['tomato', 'slategray', 'dodgerblue', 'orange', 'mediumaquamarine', 'mediumslateblue']
    
    for pid, problem in enumerate(problems):
        problem_dir = os.path.join(result_dir, problem, args.subfolder)

        # read evaluated samples csvs
        data_list = [[] for _ in range(n_algo)]
        for i, algo in enumerate(algos.keys()):
            for seed in range(n_seed):

                csv_path = f'{problem_dir}/{algo}-{problem}/{seed}/EvaluatedSamples.csv'
                file = open(csv_path, 'r')
                csv_reader = csv.reader(file)
                lines_ = []
                for row in csv_reader:
                    lines_.append(row)
                file.close()

                csv_path = f'{problem_dir}/{algo}-{problem}/{seed}/ParetoFrontApproximation.csv'
                file = open(csv_path , 'r')
                csv_reader = csv.reader(file)
                lines = []
                for row in csv_reader:
                    lines.append(row)
                file.close()

                matching_ = []
                for idx, elem in enumerate(lines):
                    for idx_, elem_ in enumerate(lines_):
                        if(idx!=0 and idx_!=0):
                            if(elem[0]==elem_[0] and float(elem[1]) == float(elem_[1]) and float(elem[2]) == float(elem_[2]) 
                                                and float(elem[3]) == float(elem_[3])):

                                a = np.array([float(elem_[4]), float(elem_[5])])
                                b = np.array([float(elem[5]), float(elem[6])])        
                                matching_.append([np.linalg.norm(a-b), abs(float(elem_[4])-float(elem[5])) / float(elem_[4]) *100, 
                                                                       abs(float(elem_[5])-float(elem[6])) / float(elem_[5]) *100])
                                #matching_.append((abs(a-b)/b*100)[0])
                            
                matching = np.array(matching_)
                data_list[i].append(matching)

        # calculate Euclidean distance (evaluations and estimations)
        for i, algo in enumerate(algos.keys()):
            distance_all = []
            ape_all_l = []
            ape_all_p = []
            for j in range(n_seed):
                distance_all.append(data_list[i][j][:,0].mean(axis=0))
                ape_all_l.append(data_list[i][j][:,1].mean(axis=0))
                ape_all_p.append(data_list[i][j][:,2].mean(axis=0))

            file2 = open("./stats.csv", "a")
            writer = csv.writer(file2)
            writer.writerow([problem, algo, np.array(distance_all).mean(axis=0), np.array(distance_all).std(axis=0),
                                            np.array(ape_all_l).mean(axis=0), np.array(ape_all_l).std(axis=0),
                                            np.array(ape_all_p).mean(axis=0), np.array(ape_all_p).std(axis=0)])
            file2.close()     

if __name__ == '__main__':
    main()
