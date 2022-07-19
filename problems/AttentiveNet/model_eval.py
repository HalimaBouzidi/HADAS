import torch
from . import models
from .utils.config import setup
from .utils.flops_counter import count_net_flops_and_params
from .evaluate.imagenet_eval import validate_one_subnet
from .data.data_loader import build_data_loader

def subnet_acc_eval(arch, train_loader, val_loader, args):
    
    criterion = torch.nn.CrossEntropyLoss().cuda()

    args.__dict__['active_subnet'] = args.__dict__['pareto_models']['a0']
    for i in range(len(arch['kernel_size'])):
        arch['kernel_size'][i] = int(arch['kernel_size'][i])
    args.__dict__['active_subnet']['resolution'] = arch['resolution']
    args.__dict__['active_subnet']['width'] = arch['width']
    args.__dict__['active_subnet']['kernel_size'] = arch['kernel_size']
    args.__dict__['active_subnet']['expand_ratio'] = arch['expand_ratio']
    args.__dict__['active_subnet']['depth'] = arch['depth']

    ## init static attentivenas model with weights inherited from the supernet 
    subnet = models.model_factory.create_model(args)
    subnet.to(args.gpu)
    subnet.eval()

    # Batch norm parameters calibration following Slimmable (https://arxiv.org/abs/1903.05134)
    with torch.no_grad():
        subnet.reset_running_stats_for_calibration()
        for batch_idx, (images, _) in enumerate(train_loader):
            if batch_idx >= args.post_bn_calibration_batch_num:
                break
            images = images.cuda(args.gpu, non_blocking=True)
            subnet(images) 

    torch.cuda.empty_cache()

    subnet.eval()
    with torch.no_grad():
        acc1, acc5, loss = validate_one_subnet(val_loader, subnet, criterion, args)

    torch.cuda.empty_cache()

    return acc1

def subnet_flops_eval(arch, args):
    
    args.__dict__['active_subnet'] = args.__dict__['pareto_models']['a0']
    for i in range(len(arch['kernel_size'])):
        arch['kernel_size'][i] = int(arch['kernel_size'][i])
    args.__dict__['active_subnet']['resolution'] = arch['resolution']
    args.__dict__['active_subnet']['width'] = arch['width']
    args.__dict__['active_subnet']['kernel_size'] = arch['kernel_size']
    args.__dict__['active_subnet']['expand_ratio'] = arch['expand_ratio']
    args.__dict__['active_subnet']['depth'] = arch['depth']

    ## init static attentivenas model with weights inherited from the supernet 
    subnet = models.model_factory.create_model(args)
    subnet.to(args.gpu)
    subnet.eval()

    # compute flops
    if getattr(subnet, 'module', None):
        resolution = subnet.module.resolution
    else:
        resolution = subnet.resolution
    data_shape = (1, 3, resolution, resolution)

    flops, params = count_net_flops_and_params(subnet, data_shape)

    torch.cuda.empty_cache()

    return flops

if __name__ == '__main__':

    args = setup('./configs/eval_attentive_nas_models.yml')
    train_loader, val_loader, train_sampler = build_data_loader(args)

    arch = {'resolution': 224, 
    'width': [16, 24, 24, 40, 72, 128, 192, 224, 1984], 
    'depth': [1, 5, 4, 5, 3, 8, 2], 
    'kernel_size': [3, 5, 5, 5, 3, 5, 5], 
    'expand_ratio': [1, 5, 6, 5, 6, 6, 6]}

    subnet_acc_eval(arch, train_loader, val_loader, args)
    subnet_flops_eval(arch, args)
    