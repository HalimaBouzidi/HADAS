import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed

import numpy as np
from utils.progress import ProgressMeter, accuracy_exits, entropy_exits

def log_helper(summary, logger=None):
    if logger:
        logger.info(summary)
    else:
        print(summary)

def validate_eex_subnet(
    val_loader,
    subnet,
    criterion,
    n_exits,
    args, 
    logger=None
):
    losses = [0 for i in range(n_exits+1)]
    top1 = [0 for i in range(n_exits+1)]
    top5 = [0 for i in range(n_exits+1)]
    conf_list_exits = [[] for i in range(n_exits+1)]
    path_list_exits = [[] for i in range(n_exits+1)]
    entropy_list_exits = [[] for i in range(n_exits+1)]
    correct_list_exits = [[] for i in range(n_exits+1)]
    progress = ProgressMeter(
                len(val_loader),
                [losses, top1, top5],
                prefix='Test: ')

    log_helper('evaluating...', logger)   
    
    #evaluation
    subnet.cuda(args.gpu)
    subnet.eval() # freeze again all running stats
   
    for batch_idx, (images, target) in enumerate(val_loader):     
    # path for CIFAR datasets is the index

        images = images.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)
        val_len = len(val_loader)
        batch_size = images.size(0)

        # compute output
        output, conf, exits = subnet(images)
        batch_entropy = entropy_exits(output, exits)

        # measure accuracy 
        acc, correct = accuracy_exits(output, target, topk=(1, 5), n_exits=exits, per_sample=True)
                
        for i, elem in enumerate(acc, start=0):
            top1[i] += elem[0].item()
            top5[i] += elem[1].item()
            
        for i in range(exits+1):
            losses[i] += criterion(output[i], target).item()   
            conf_list_exits[i].extend(conf[i])
            entropy_list_exits[i].extend(batch_entropy[i])
            #path_list_exits[i].extend(path)
            correct_list_exits[i].extend(correct[i][0]) 

    avg_top1 = [elem/val_len for elem in top1]
    avg_top5 = [elem/val_len for elem in top5]
    avg_losses = [elem/val_len for elem in losses]

    return avg_top1, avg_top5, avg_losses, conf_list_exits, path_list_exits, entropy_list_exits, correct_list_exits

def validate_eex(
    train_loader, 
    val_loader, 
    model, 
    criterion, 
    n_exits, 
    args,
    logger,
    bn_calibration=True,
):
    results = []
    with torch.no_grad():

            model.eval()
            
            if bn_calibration:
                model.eval()
                model.reset_running_stats_for_calibration()

                # estimate running mean and running statistics
                logger.info('Calibrating bn running statistics')
                for batch_idx, (images, _) in enumerate(train_loader):
                    if batch_idx >= args.post_bn_calibration_batch_num:
                        break
                    if getattr(args, 'use_clean_images_for_subnet_training', False):
                        _, images = images
                    images = images.cuda(args.gpu, non_blocking=True)
                    model(images)  #forward only

            acc1, acc5, loss, _, _, _, _ = validate_eex_subnet(val_loader, model, criterion, n_exits, args, logger)

            acc1 = np.mean(acc1)
            acc5 = np.mean(acc5)
            loss = np.mean(loss)
            
            summary = str({
                        'mode': 'evaluate',
                        'epoch': getattr(args, 'curr_epoch', -1),
                        'acc1': acc1,
                        'acc5': acc5,
                        'loss': loss,
            })

            logger.info(summary)
            results += [summary]

    return results
