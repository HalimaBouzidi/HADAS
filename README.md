# Towards Energy-efficient Dynamic Hardware-aware Neural Architecture Search with DVFS

## Overview

This repository contains Python implementation of the algorithm framework for Hardware-aware Dynamic Neural Architecture Search with DVFS. The framework is built upon 
- DGEMO: Framework for Multi-objective Bayesian Optimization.
- Pymoo: Framework for Multi-objective Evolutionary Optimization.


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

- Download the pre-trained supernet of AttentiveNAS from [this link](https://drive.google.com/file/d/1cCla-OQNIAn-rjsY2b832DuP59ZKr8uh/view?usp=sharing) and put it in *problems/AttentiveNet/attentive_nas_data/*

- Run the script file *script.sh*. The script can run the HW-NAS with one of the optimization algorithms (NSGA-II, DGEMO, USEMO-EI)


## Result

- The optimization results are saved in csv format and the arguments are saved as a yaml file. They are stored under the folder:

```
result/{problem}/{subfolder}/{algo}-{exp-name}/{seed}/
```

*Explanation --- problem: problem name, algo: algorithm name, exp-name: experiment name, seed: random seed used*

- The name of the argument yaml file is `args.yml`.


## Fine-tuning:
    
    
| Subnet/weights | Cifar10 | Cifar-100 | Tiny-Imagenet | MFLOPs |
|:---:|:---:|:---:|:---:|:---:|
| Min_subnet_Acc | 97.99 | 86.28 | 76.73 | 201 |
| Max_subnet_Acc | 98.58 | 88.43 | 80.54 | 1937 |
| Link to Weights | [Link](https://drive.google.com/drive/folders/1IwvWd8oswS6YBGhu-32YHhXRKqFBnDaF?usp=sharing) | [Link](https://drive.google.com/drive/folders/1hcgG8Jcp_iiJR6ekjlNjdGm7ThEVw-bB?usp=sharing) | [Link](https://drive.google.com/drive/folders/1n3eeX7g8c-MUNWizda3LBZwC5qIN1aKb?usp=sharing) | - |



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
