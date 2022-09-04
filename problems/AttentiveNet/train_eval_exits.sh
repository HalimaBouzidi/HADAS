#!/bin/bash

script_1="python3 /home/hbouzidi/hbouzidi/AttentiveNAS/train_exit_blocks.py \
--config-file /home/hbouzidi/hbouzidi/AttentiveNAS/configs/train_attentive_nas_eex_models_cifar.yml --model a0 --gpu 0";

script_2="python3 /home/hbouzidi/hbouzidi/AttentiveNAS/test_for_exits.py \
--config-file /home/hbouzidi/hbouzidi/AttentiveNAS/configs/eval_attentive_nas_eex_models_cifar.yml --model a0 --gpu 0";

script_3="python /home/hbouzidi/hbouzidi/AttentiveNAS/test_for_exits_calibration.py \
--config-file /home/hbouzidi/hbouzidi/AttentiveNAS/configs/eval_attentive_nas_eex_models_cifar.yml --model a0 --gpu 1 \
--temp_step 1 --n_bins 5 --temp_min=1 --temp_max=10 --loader train";

$script_1 ;  

$script_2 ;  

$script_3 ;  