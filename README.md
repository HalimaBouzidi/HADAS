# HADAS: Hardware-Aware Dynamic Neural Architecture Search for Edge Performance Scaling

## Overview

This repository contains theb Python implementation of HADAS Framework for Hardware-aware Dynamic Neural Architecture Search. The framework is built upon 
- Pymoo: Framework for Multi-objective Evolutionary Optimization.


## Code Structure

```
baselines/ --- MOO baseline algorithms: NSGA-II
optim/
 ├── solver/ --- multi-objective solver
 ├── algorithms.py --- high-level algorithm specifications
 ├── factory.py --- factory for importing different algorithm components
 ├── selection.py --- selection methods for new populations
 └── utils.py --- utility functions
problems/ --- Hardware-aware NAS problem definition and evaluation
scripts/ --- scripts to run experiments
visualization/ --- performance visualization
main.py --- main execution file for HADAS optimization algorithm
```

## Requirements

- Python version: tested in Python 3.8.10
- Install the software environment in the yaml file *environment.yml*

## Getting Started

- Download the pre-trained supernet of AttentiveNAS from [this link](https://drive.google.com/file/d/1cCla-OQNIAn-rjsY2b832DuP59ZKr8uh/view?usp=sharing) and put it in *problems/AttentiveNet/attentive_nas_data/*

- Run the script file *script.sh*. The script can run the optimization process for OOE+IOE or IOE for selected backbones


## Result

- The optimization results are saved in csv format and the arguments are saved as a yaml file. They are stored under the folder:

```
results/{exp-name}/{seed}
```

*Explanation --- optim-level: optimization-level, backbone: backbone name, exp-name: experiment name, seed: random seed used*

- The name of the argument yaml file is `args.yml`.


## Fine-tuning:
    
| Subnet/weights | Cifar10 | Cifar-100 | Tiny-Imagenet | MFLOPs |
|:---:|:---:|:---:|:---:|:---:|
| Min_subnet_Acc | 97.99 | 86.28 | 76.73 | 201 |
| Max_subnet_Acc | 98.58 | 88.43 | 80.54 | 1937 |
| Link to Weights | [Link](https://drive.google.com/drive/folders/1IwvWd8oswS6YBGhu-32YHhXRKqFBnDaF?usp=sharing) | [Link](https://drive.google.com/drive/folders/1hcgG8Jcp_iiJR6ekjlNjdGm7ThEVw-bB?usp=sharing) | [Link](https://drive.google.com/drive/folders/1n3eeX7g8c-MUNWizda3LBZwC5qIN1aKb?usp=sharing) | - |



## References

```
@inproceedings{wang2021attentivenas,
  title={Attentivenas: Improving neural architecture search via attentive sampling},
  author={Wang, Dilin and Li, Meng and Gong, Chengyue and Chandra, Vikas},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={6418--6427},
  year={2021}
}
```
