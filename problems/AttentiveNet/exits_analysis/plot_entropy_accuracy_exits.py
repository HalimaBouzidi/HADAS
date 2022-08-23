import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import csv
import argparse
import shutil

from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA

from scipy.interpolate import make_interp_spline, BSpline

"""
Script to plot ratio of both overall samples and correctly-classified ones over entropy distributions (set according to n_bins). 
"""

parser = argparse.ArgumentParser(description='entropy bins/frequency plot')
parser.add_argument('--dataset', default='tiny_imagenet', type=str)
parser.add_argument('--n_exits', default=16, type=int)
parser.add_argument('--loss', default='kl-div', type=str)
parser.add_argument('--n_bins', default=4, type=int)

args = parser.parse_args()

root = '.'

entropy_file = root +'/' + str(args.dataset) +'/'+ str(args.loss) + '/entropy_table' + '.csv'
df = pd.read_csv(entropy_file, usecols= ['Exit', 'Sample_path', 'Entropy', 'Correct', 'Confidence'])

colors = ['#013A20', '#478C5C', '#BACC81', '#CDD193'] 	#green lemons

local_counts = [0, 1, 0, 0] 

entropy_exits_list = [] 		# max = 5.0484  
correct_exits_list = []
path_exits_list = []
confidence_exits_list = []

for i in range(1, args.n_exits+1):
    df_select = df.loc[df['Exit'] == 'Exit_'+str(i)]
    entropy_exits_list.append(df_select['Entropy'])  
    correct_exits_list.append(df_select['Correct'])
    path_exits_list.append(df_select['Sample_path'])    
    confidence_exits_list.append(df_select['Confidence'])


#************************* Entrpo plotting **************************#
for j in range(args.n_exits):
    entropy_list = entropy_exits_list[j]
    correct_list = correct_exits_list[j]
    max_entropy = max(entropy_list)
    step_size = round((max_entropy - 0)/(args.n_bins),2)

    category = {}
    index_list = []
    for i in range(args.n_bins): 
        index_list.append(i*step_size + step_size)
        category[str(i)] = {'values': [], 'n_correct': 0}

    for entropy, correct in zip(entropy_list, correct_list):
        for i, threshold in enumerate(index_list):
            if entropy <= threshold:
                category[str(i)]['values'].append((entropy, correct))
                if correct:
                    category[str(i)]['n_correct'] += 1
                break

    plt.rcParams["font.family"] = "Times New Roman"
    fig, ax = plt.subplots()

    pos = np.arange(0.2, args.n_bins)

    width = 0.1 # the width of the bars 

    ax.set_xticks(pos+2*(3/4)*width)
    ax.set_xticklabels(index_list, minor=False, fontsize = 11.5)
    # ax.set_yticklabels(fontsize = 12)
    ax.tick_params(axis='y', labelsize=11.5)

    ax.set(xlim=[0, pos[-1]+0.5])
    # ax.set(ylim=ylims)

    # ax.set_xlabel('Action', fontsize = 12)
    ax.set_ylabel('Frequency', fontsize = 12)  

    #ind = np.arange(len(index))  # the x locations for the groups
    edge_hatch = ['oooooo', '......', 'oooo', 'oooooo']
    edge_hatch = ['','','','']

    for key, position in zip(category.keys(), pos):
        ax.bar(position, len(category[key]['values']), width, color=colors[0], label='assigned', hatch = edge_hatch[0], edgecolor = 'black', zorder = 3, alpha=0.5) 
        ax.bar(position, category[key]['n_correct'], width, color=colors[1], label='correct', hatch=edge_hatch[1], edgecolor='black', zorder=3, alpha=1)


    save_path = root +'/' + str(args.dataset) +'/'+ str(args.loss) + '/entropy_exit_' +str(j+1) + '.png'
    plt.savefig(save_path)


accuracy_file = root +'/' + str(args.dataset) +'/'+ str(args.loss) + '/exit_perf' + '.csv'
df = pd.read_csv(accuracy_file, usecols= ['Exit', 'TOP-1', 'TOP-5', 'Loss'])
top1 = df['TOP-1']
top5 = df['TOP-5']
losses = df['Loss']

fig, ax = plt.subplots()
plt.plot(top1, '--bo', label='TOP1 Acc', color='red')
plt.plot(top5, '--bo', label='TOP5 Acc', color='blue')
plt.plot(losses, '--bo', label='Loss', color='green')
plt.legend()
save_path = root +'/' + str(args.dataset) +'/'+ str(args.loss) + '/accuracy_exit_' +str(j+1) + '.png'
plt.savefig(save_path)