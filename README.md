# HADAS: Hardware-Aware Dynamic Neural Architecture Search for Edge Performance Scaling

## Overview

This repository contains the Python implementation of HADAS Framework for Hardware-aware Dynamic Neural Architecture Search. The framework is built upon:
- AttentiveNAS: Framwork for Neural Architecture Search
- Pymoo: Framework for Multi-objective Evolutionary Optimization.

For more details, please refer to our paper [HADAS: Hardware-Aware Dynamic Neural Architecture Search for Edge Performance Scaling](https://arxiv.org/pdf/????) by Halima Bouzidi, Mohanad Odema, Hamza Ouarnouhgi, Mohammad Abdullah Al-Faruque, and Smail Niar.

If you find this implementation helpful, please consider citing our work:

```BibTex
@article{????,
  title={HADAS: Hardware-Aware Dynamic Neural Architecture Search for Edge Performance Scaling},
  author={Halima, Bouzidi and Mohanad, Odema and Hamza, Ouarnouhgi and Mohammad, Abdullah Al-Faruque and Smail, Niar},
  journal={arXiv preprint arXiv:????},
  year={2022}
}
```

## Code Structure

```
search_space/
 ├── nas_search_space.py --- Search space definition for the NAS process (OOE)
 └── eex_dvfs_search_space --- Search space definition for EEx_DVFS (IOE)
search_algo/
 ├── hadas_search.py --- OOE + IOE search algorithms (evolutionary-based)
 ├── utils_opt.py --- Selection functions for the optimization algorithms (based on non-dominated sorting and crowding distance)
 └── utils_eval.py --- utility functions to remotely run the evaluation and read/write results
supernet/
 ├── AttentiveNAS/ --- Contains the original scripts from AttentiveNAS for supernet specifications + our scripts to transform backbones to dynamic neural networks with multiple exits
 ├── utils/ --- Contains essential scripts to manage distributed training/evaluation over multiple GPUs
 ├── parellel_evaluation_exits.py --- Main script to train, evaluate multi-exits backbones in parallel
 └── parellel_evaluation_nas.py --- Main script to evaluate backbone accuracy in parallel
```

## Requirements

- Python version: tested in Python 3.8.10
- Install the software environment in the yaml file *environment.yml*

## Getting Started

- Download one of our pre-trained supernets of AttentiveNAS and put it in *supernet/AttentiveNet/attentive_nas_data/*

- Run the script file *script.sh*. The script can run the optimization process for OOE+IOE or IOE for a selected backbone(s)


## Evaluation

## Pretrained AttentiveNAS supernets on multiple datasets:
    
| Subnet/weights | Cifar10 | Cifar-100 | Tiny-Imagenet | MFLOPs |
|:---:|:---:|:---:|:---:|:---:|
| Min_subnet_Acc | 97.99 | 86.28 | 76.73 | 201 |
| Max_subnet_Acc | 98.58 | 88.43 | 80.54 | 1937 |
| Link to Weights | [Link](https://drive.google.com/drive/folders/1IwvWd8oswS6YBGhu-32YHhXRKqFBnDaF?usp=sharing) | [Link](https://drive.google.com/drive/folders/1hcgG8Jcp_iiJR6ekjlNjdGm7ThEVw-bB?usp=sharing) | [Link](https://drive.google.com/drive/folders/1n3eeX7g8c-MUNWizda3LBZwC5qIN1aKb?usp=sharing) | - |



## Acknowledgements

[Pymoo](https://github.com/anyoptimization/pymoo)

[AttentiveNAS](https://github.com/facebookresearch/AttentiveNAS)

