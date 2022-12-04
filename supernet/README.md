# Towards Energy-efficient Dynamic Hardware-aware Neural Architecture Search with DVFS

## Overview

This section of the repository contains the original codebase for the AttentiveNAS supernet and its evaluations. Additional scripts explained below target entropy analysis for the predicted outputs from the trained AttentiveNets on CIFAR10 and tiny-imagenet. 


## Additional Scripts 

Example to evaluate subnet's performance on a dataset with entropy grouping as follows.

```
python test_for_clustering.py --config-file ./configs/eval_attentive_nas_models_tiny.yml --model a1
```

Example to sweep over temperature values to identify the one that gives minimal Expected Calibration Error (ECE) (Once identified, you need to apply it manually to the outputs of a model -- see 'temp_scale_and_softmax()' in 'calibration.py' for reference)
```
python test_for_calibration.py --config-file ./configs/eval_attentive_nas_models_tiny.yml --model a1 --temp_step 1 --n_bins 5 --temp_min=1 --temp_max=100 --loader train
```

Example to evaluate at a specific temperature value
```
python test_for_calibration.py --config-file ./configs/eval_attentive_nas_models_tiny.yml --model a1 --temp_min=2 --loader train -single
```

To visualize how predictions are distributed over entropy scores use the following script.
-- n_bins flag is for setting the number of entropy groups
-- save_lists flag is for restructuring the dataset into entropy groups (currently working with tiny_imagenet)

```
python dataset_entropy_analysis/plot_entropy_accuracy.py --dataset cifar10 --n_bins 4 --category val --save_lists
```


To evaluate the exit architectures, download the pretrained weights from [this link](https://drive.google.com/drive/folders/1YKvht2ROO6gjlTHmlE-pf65A4X1q3W0m?usp=sharing) and put them in *problems/AttentiveNet/saved_models*