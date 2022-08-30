# Class for confiednece calibration with Calibration-related fns
# bin is the bin number, bin_data is a nested list containing the true class labels and the associated confidences for each prediction
# TODO: MAKE SURE INPUT AND OUTPUT DATA PASSED ON TO THE CALIBRATE FUNCTION ARE FROM THE DATALOADER CALL IN VALIDATE
# I can do it like the imagenet eval where I keep passing on batches, and the class keeps on updating
# You need to show how you added NamedImageFOlder for H

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class temp_estimator():
    def __init__(self, model, dataloader, temp_step=5, n_bins=5, temp_min=0, temp_max=5000):
        self.model = model
        self.dataloader = dataloader
        self.global_ECE = np.inf
        self.best_temp = 1
        self.temp_min = temp_min
        self.temp_max = temp_max
        self.ECE_list = []       # if needed for plot
        self.temp_list = []
        self.temp_step = temp_step
        self.n_bins = n_bins
        self.bin_dict = {}
        self.bin_step = 1.0 / self.n_bins 
        self.total_samples = 0

    def infer_for_calibration(self, temp):
        with torch.no_grad():
            for batch_idx, (images, targets, path) in enumerate(self.dataloader):
                # 'path' for CIFAR datasets is the index
                images = images.cuda(0, non_blocking=True)
                targets = targets.cuda(0, non_blocking=True)   # class number directly (not one hot vector)
                self.total_samples += len(images)
                outputs = self.model(images)        # probabilities (#samples*#classes)
                outputs = self.temp_scale_and_softmax(outputs, temp)        # scaled softmax  
                self.assign_to_bins(outputs, targets, path) 
                for key in self.bin_dict:
                    print(batch_idx, len(self.bin_dict[key]['bin_true']))

    def evaluate(self):
        # evaluating for a single temperature value
        if self.temp_min == 0:
            raise ValueError("Set temp_min argument to a value other than 0 for evaluation.")
        self.reset_bin_dict()
        self.total_samples = 0
        self.infer_for_calibration(self.temp_min)
        ECE = round(self.Exp_Cal_Error(), 3)
        return self.temp_min, ECE

    def sweep(self, temp1_start=True):
        # evaluating for multiple candidatetemperatures at once
        if temp1_start:
            self.reset_bin_dict()
            self.total_samples = 0
            self.infer_for_calibration(1)
            # lazy coding
            ECE = round(self.Exp_Cal_Error(), 3)
            self.ECE_list.append(ECE)
            self.temp_list.append(1)
            if ECE < self.global_ECE:
                self.global_ECE = ECE
                self.best_temp = 1
        for temp in np.arange(self.temp_min, self.temp_max, self.temp_step):
            print(f"Temperature Value: {temp}")
            self.reset_bin_dict()
            self.total_samples = 0
            if temp == 0:
                continue
            else:
                self.infer_for_calibration(temp)
            # print(f"Bin sizes for temp {temp} are: {len(bin_0_true)}, {len(bin_1_true)}, {len(bin_2_true)}, {len(bin_3_true)}, {len(bin_4_true)}")
            ECE = round(self.Exp_Cal_Error(), 3)
            self.ECE_list.append(ECE)
            self.temp_list.append(temp)
            if ECE < self.global_ECE:
                print(f"New best ECE: {ECE} @temp: {temp}")
                self.global_ECE = ECE
                self.best_temp = temp
        return self.best_temp, self.global_ECE

    def temp_scale_and_softmax(self, logits, temp):
        scaled_outputs = (1.0/temp)*logits
        return F.softmax(scaled_outputs)

    def assign_to_bins(self, outputs, targets, paths):
        for output, target, path in zip(outputs, targets, paths): 
            for i in range(0, self.n_bins):         # iterate over bins
                bin_th = min(round(self.bin_step*i + self.bin_step,2), 1.0) 
                if torch.max(output).item() < bin_th:         # largest probability meets the threshold
                    self.bin_dict['bin'+str(i)+'(<'+str(bin_th)+')']['bin_true'].append(target.item())
                    self.bin_dict['bin'+str(i)+'(<'+str(bin_th)+')']['bin_conf'].append(torch.max(output).item())
                    self.bin_dict['bin'+str(i)+'(<'+str(bin_th)+')']['bin_pred'].append(torch.argmax(output).item())
                    break

    def reset_bin_dict(self):
        self.bin_dict = {}
        for i in range(0, self.n_bins):
            bin_th = min(round(self.bin_step*i + self.bin_step,2), 1.0)         # The min is when small fractions exceed 1.0
            self.bin_dict['bin'+str(i)+'(<'+str(bin_th)+')'] = {'bin_true':[], 'bin_conf':[], 'bin_pred':[]}     # Each bin is also a dictionary to store labels & predictions (the keys)

    def Bin_accuracy(self, bin_data):
        # Number of Correctly classified samples over the total
        targets = bin_data['bin_true']
        predictions = bin_data['bin_pred']
        # The following in case the targets/predictions are in a hot vector format
        # predicted_labels = []
        # true_labels = []
        # for (prediction, target) in zip(predictions, targets):
        #     predicted_labels.append(np.argmax(prediction)) 
        #     true_labels.append(np.argmax(target))
        # assert(len(true_labels) == len(predicted_labels)) 
        correct_predictions = 0 
        for (prediction, target) in zip(predictions, targets):
            if prediction == target:
                correct_predictions += 1
        acc = correct_predictions/(len(targets)+10e-14)
        return acc

    def Bin_confidence(self, bin_data):
        # The average confidence experienced in this bin
        probabilities = bin_data['bin_conf']
        # confidences = []
        # for probability in probabilities:
        #     confidences.append(np.max(prediction))
        # assert(len(confidences) == len(probabilities))
        avg_confidence = (sum(probabilities))/(len(probabilities)+10e-14)
        return avg_confidence

    def Exp_Cal_Error(self):
        # Average expected Calibration error across all bins
        ECE = 0
        for i, bin_key in enumerate(self.bin_dict.keys()):
            bin_acc = round(self.Bin_accuracy(self.bin_dict[bin_key]),3)
            bin_conf = round(self.Bin_confidence(self.bin_dict[bin_key]),3)
            bin_weight = len(self.bin_dict[bin_key]['bin_true'])/self.total_samples
            diff = abs(bin_acc - bin_conf)
            ECE += bin_weight*diff
            print(f"{i}: {bin_key}, Samples: {len(self.bin_dict[bin_key]['bin_conf'])}, accuracy: {bin_acc}, avg_conf: {bin_conf}")
        return round(ECE,3)