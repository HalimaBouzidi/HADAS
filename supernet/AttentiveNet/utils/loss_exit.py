import numpy as np
import torch
import torch.nn.functional as F

class CumulativeCrossEntropyExits(torch.nn.modules.loss._Loss):
    def forward(self, output, num_ee, target):
        
        pred_loss = 0
        for i in range(num_ee):
            criterion = torch.nn.CrossEntropyLoss().cuda()
            pred_loss += criterion(output[i].log(), target)
        cum_loss = pred_loss

        return cum_loss/num_ee

class CumulativeLossExits(torch.nn.modules.loss._Loss):
    def forward(self, output, num_ee, target):
        
        pred_loss = 0
        for i in range(num_ee):
            pred_loss += F.nll_loss(output[i].log(), target)
        cum_loss = pred_loss

        return cum_loss/num_ee

class CumulativeKLDivergenceExits(torch.nn.modules.loss._Loss):
    def forward(self, pred_exits, num_ee, final_target=None, temperature=1.):
        distill_loss = 0
        classif_loss = 0
        for i in range(num_ee): # student exits 
            s_output = pred_exits[i]
            t = 0
            kd_loss = 0
            # Compute the distillation loss for each exits and its older teachers
            for j in range(num_ee, num_ee+1): # teacher exits 
                t_output = pred_exits[j]
                s_output, t_output = F.softmax(s_output, dim=1)/temperature, F.softmax(t_output, dim=1) / temperature # temperature scaling 
                loss = -temperature*temperature*torch.sum(t_output * s_output.log(), dim=1)
                kd_loss += loss.mean()
                t += 1

            distill_loss += kd_loss / t

            # Compute the classification loss (nll or Cross-Entropy) for all exits and target labels
            target = final_target
            classif_loss += F.nll_loss(F.softmax(pred_exits[i]).log(), target)

        #print(distill_loss, classif_loss)
        cum_loss = (distill_loss + classif_loss) / num_ee

        return cum_loss

class CumulativeKLDivergenceExitsOld(torch.nn.modules.loss._Loss):
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