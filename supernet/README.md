# HADAS: Hardware-Aware Dynamic Neural Architecture Search for Edge Performance Scaling

## Overview

This section of the repository contains the original codebase for the AttentiveNAS supernet and its evaluations. Additional scripts explained below target Accuracy evaluation and Early-exit training and scores assessment:


## Additional Scripts 

Example to run parallel accuracy evaluation for multiple backbones (sampled from OOE):

```
python3 parellel_evaluation_nas.py --machine-rank 0 --num-machines 16 --dist-url tcp://node_addrs:port \
--config-file ./config.yml --seed {seed}"
```

Example to run parallel training/evaluation for the multi-exit backbone (sampled from IOE):

```
python3 parellel_evaluation_exits.py --machine-rank 0 --num-machines 16 --dist-url tcp://node_addrs:port \
--config-file config.yml --seed {seed}"
```

