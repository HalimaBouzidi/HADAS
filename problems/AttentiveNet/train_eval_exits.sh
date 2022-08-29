#!/bin/bash

script_1="python3 /home/hbouzidi/hbouzidi/AttentiveNAS/train_exit_blocks.py \
--config-file /home/hbouzidi/hbouzidi/AttentiveNAS/configs_attentivenet/train_attentive_nas_eex_models_cifar.yml --model a0 --gpu 0";

script_2="python3 /home/hbouzidi/hbouzidi/AttentiveNAS/test_for_exits.py \
--config-file /home/hbouzidi/hbouzidi/AttentiveNAS/configs_attentivenet/eval_attentive_nas_eex_models_cifar.yml --model a0 --gpu 0";

$script_1 ;  

$script_2 ;  