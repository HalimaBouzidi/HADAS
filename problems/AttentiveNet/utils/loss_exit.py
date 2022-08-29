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

class CumulativeNLLExits(torch.nn.modules.loss._Loss):
    def forward(self, output, num_ee, target):
        
        pred_loss = 0
        for i in range(num_ee):
            pred_loss += F.nll_loss(output[i].log(), target)
        cum_loss = pred_loss

        return cum_loss/num_ee

# Implementation of the loss function proposed in: 
# https://openaccess.thecvf.com/content_ICCV_2019/papers/Phuong_Distillation-Based_Training_for_Multi-Exit_Architectures_ICCV_2019_paper.pdf

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
                s_output, t_output = F.softmax(s_output, dim=1)/ temperature, F.softmax(t_output, dim=1) / temperature # temperature scaling 
                loss = -temperature*temperature*torch.sum(t_output * s_output.log(), dim=1)
                kd_loss += loss.mean()
                t += 1

            distill_loss += kd_loss / t

            # Compute the classification loss (nll or Cross-Entropy) for all exits and target labels
            target = final_target
            classif_loss += F.nll_loss(pred_exits[i].log(), target)

        #print(distill_loss, classif_loss)
        cum_loss = (distill_loss + classif_loss) / num_ee

        return cum_loss