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
Set --save_lists flag to save the entropy-categorized dataset into a target directory
"""


parser = argparse.ArgumentParser(description='entropy bins/frequency plot')
parser.add_argument('--dataset', default='tiny_imagenet', type=str)
parser.add_argument('--category', default='train', type=str)
parser.add_argument('--n_bins', default=4, type=int)
parser.add_argument('--save_lists', action='store_true')

args = parser.parse_args()

root = './dataset_entropy_analysis/'
save_root = '/home/mohanadodema/dataset/'

entropy_file = root + str(args.dataset) + '/entropy_table_' + str(args.category) + '.csv'
df = pd.read_csv(entropy_file)

colors = ['#013A20', '#478C5C', '#BACC81', '#CDD193'] 	#green lemons

local_counts = [0, 1, 0, 0] 

entropy_list = df['Entropy'] 		# max = 5.0484  
correct_list = df['Correct']
path_list = df['Sample_path']

max_entropy = max(entropy_list)
step_size = round((max_entropy - 0)/(args.n_bins),2)

category = {}
index_list = []
for i in range(args.n_bins): 
	index_list.append(i*step_size + step_size)
	category[str(i)] = {'values': [], 'n_correct': 0}

for path, entropy, correct in zip(path_list, entropy_list, correct_list):
	for i, threshold in enumerate(index_list):
		# print(threshold)
		if entropy <= threshold:
			category[str(i)]['values'].append((path, entropy, correct))
			if correct:
				category[str(i)]['n_correct'] += 1
			break

for key, value in category.items():
	print(len(category[key]['values']), category[key]['n_correct'])


plt.rcParams["font.family"] = "Times New Roman"
fig, ax = plt.subplots()

pos = np.arange(0.2,args.n_bins)

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

	if args.save_lists:
		for triplet in category[key]['values']:
			path = triplet[0]
			split_path = path.split('/')[-3:]
			target_path = save_root + args.dataset + '_categorized/' + str(args.category) + 'entropy_' + str(key) + '/' + split_path[-3] + '/' + split_path[-2]
			os.makedirs(target_path, exist_ok=True)
			shutil.copy(path, target_path)



# if args.legend is True:
# ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
#          ncol=4, frameon = False, fontsize = 12)
plt.show()
# else:
# 	plt.savefig('C:/Users/Mohanad Odema/Desktop/Resnet' + str(args.arch) + '_' + str(args.trade_off) + '.svg', bbox_inches='tight')

# plt.show()