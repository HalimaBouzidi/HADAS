import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed

import utils.comm as comm
from .imagenet_eval import validate_one_subnet

def validate(
    subnets_to_be_evaluated,
    train_loader, 
    val_loader, 
    model, 
    args, 
    bn_calibration=True,
):
    supernet = model.module \
        if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model

    results = []
    with torch.no_grad():
        for idx, net_id in enumerate(subnets_to_be_evaluated, start=0):

            supernet.set_active_subnet(
                subnets_to_be_evaluated[net_id]['resolution'],
                subnets_to_be_evaluated[net_id]['width'],
                subnets_to_be_evaluated[net_id]['depth'],
                subnets_to_be_evaluated[net_id]['kernel_size'],
                subnets_to_be_evaluated[net_id]['expand_ratio'],
            )

            subnet = supernet.get_active_subnet()
            subnet_cfg = supernet.get_active_subnet_settings()
            subnet.cuda(args.gpu)

            if bn_calibration:
                subnet.eval()
                subnet.reset_running_stats_for_calibration()

                # estimate running mean and running statistics
                for batch_idx, (images, _, _) in enumerate(train_loader):
                    if batch_idx >= args.post_bn_calibration_batch_num:
                        break
                    if getattr(args, 'use_clean_images_for_subnet_training', False):
                        _, images = images
                    images = images.cuda(args.gpu, non_blocking=True)
                    subnet(images)  #forward only

            err1 = 100 - validate_one_subnet(val_loader, subnet, args)

            summary = str({
                        'net_id': net_id,
                        'mode': 'evaluate',
                        'acc1': err1,
                        **subnet_cfg
            })

            if args.distributed and getattr(args, 'distributed_val', True):
                results += [summary]
            else:
                group = comm.reduce_eval_results(summary, args.gpu)
                results += group
    
    return results
