# Towards Energy-efficient Dynamic Hardware-aware Neural Architecture Search with DVFS

## Overview

This repository contains Python implementation of the algorithm framework for Hardware-aware Dynamic Neural Architecture Search with DVFS. The framework is built upon 
1- DGEMO: Framework for Multi-objective Bayesian Optimization.
2- Pymoo: Framework for Multi-objective Evolutionary Optimization.


## Code Structure

```
baselines/ --- MOO baseline algorithms: NSGA-II
mobo/
 ├── solver/ --- multi-objective solvers
 ├── surrogate_model/ --- surrogate models
 ├── acquisition.py --- acquisition functions
 ├── algorithms.py --- high-level algorithm specifications
 ├── factory.py --- factory for importing different algorithm components
 ├── mobo.py --- main pipeline of multi-objective bayesian optimziation
 ├── selection.py --- selection methods for new samples
 ├── surrogate_problem.py --- multi-objective surrogate problem
 ├── transformation.py --- normalizations on data
 └── utils.py --- utility functions
problems/ --- Hardware-aware NAS problem definition and evaluation
scripts/ --- scripts to run experiments
visualization/ --- performance visualization
main.py --- main execution file for MOBO algorithms
```

## Requirements

- Python version: tested in Python 3.8.10

- Install the software environment in the yaml file *environment.yml*

- Install pygco by pip (which is used by the bayesian optimization DGEMO):

  ```
  pip install pygco
  ```

## Getting Started

- Change the name of the project directory in *cosearch.py"

- Run the script file *script.sh*. The script can run the HW-NAS with one of the optimization algorithms (NSGA-II, DGEMO, USEMO-EI)


## Result

- The optimization results are saved in csv format and the arguments are saved as a yaml file. They are stored under the folder:

```
result/{problem}/{subfolder}/{algo}-{exp-name}/{seed}/
```

*Explanation --- problem: problem name, algo: algorithm name, exp-name: experiment name, seed: random seed used*

- The name of the argument yaml file is `args.yml`.


## References

```
@article{konakovic2020diversity,
  title={Diversity-Guided Multi-Objective Bayesian Optimization With Batch Evaluations},
  author={Konakovic Lukovic, Mina and Tian, Yunsheng and Matusik, Wojciech},
  journal={Advances in Neural Information Processing Systems},
  volume={33},
  year={2020}
}
```

```
@inproceedings{wang2021attentivenas,
  title={Attentivenas: Improving neural architecture search via attentive sampling},
  author={Wang, Dilin and Li, Meng and Gong, Chengyue and Chandra, Vikas},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={6418--6427},
  year={2021}
}
```
