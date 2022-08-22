import numpy as np
import torch
import torch.nn.functional as F

class CumulativeLossExits(torch.nn.modules.loss._Loss):
    def forward(self, output, num_ee, target):
        
        pred_loss = 0
        for i in range(num_ee):
            pred_loss += F.nll_loss(output[i].log(), target)
        cum_loss = pred_loss

        return cum_loss/num_ee

class CumulativeKLDivergenceExits(torch.nn.modules.loss._Loss):
    def forward(self, pred, soft_logits, num_ee, final_target=None, reduction='mean', temperature=1., alpha=0.9):
        pred_loss = 0
        for i in range(num_ee):
            output = pred[i]
            target = final_target
            output, soft_logits = output/ temperature, soft_logits / temperature
            soft_target_prob = torch.nn.functional.softmax(soft_logits, dim=1)
            output_log_prob = torch.nn.functional.log_softmax(output, dim=1)
            kd_loss = -torch.sum(soft_target_prob * output_log_prob, dim=1)
            if target is not None:
                n_class = output.size(1)
                target = torch.zeros_like(output).scatter(1, target.view(-1, 1), 1)
                target = target.unsqueeze(1)
                output_log_prob = output_log_prob.unsqueeze(2)
                ce_loss = -torch.bmm(target, output_log_prob).squeeze()
                loss = alpha*temperature* temperature*kd_loss + (1.0-alpha)*ce_loss
            else:
                loss = kd_loss 
                        
            if reduction == 'mean':
                pred_loss += loss.mean()
            elif reduction == 'sum':
                pred_loss += loss.sum()
        
        cum_loss = pred_loss/num_ee

        return cum_loss