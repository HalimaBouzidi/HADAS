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
        self.global_ECE = np.Inf()
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
                targets = targets.cuda(0, non_blocking=True)
                self.total_samples += len(images)
                outputs = self.model(images)        # probabilities
                outputs = self.temp_scale_and_softmax(outputs, temp)        # scaled softmax
                self.assign_to_bins(outputs, targets, path) # keep extending lists for the final ECE computation

    def evaluate(self, temp=self.temp_min):
        # evaluating for a single temperature value
        self.reset_bin_dict()
        self.total_samples = 0
        self.infer_for_calibration(temp)
        ECE = round(self.Exp_Cal_Error(), 3)
        return temp, ECE

    def sweep(self, temp1_start=True):
        # evaluating for multiple candidatetemperatures at once
        if temp1_start:
            self.reset_bin_dict()
            self.total_samples = 0
            self.infer_for_calibration(1)
        for temp in range(self.temp_min, self.temp_max, self.temp_step):
            self.reset_bin_dict()
            self.total_samples = 0
            if temp <= 1 and temp1_start:
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
                if np.max(output) < bin_th:         # largest probability meets the threshold
                    self.bin_dict['bin'+str(i)+'(<'+str(bin_th)+')']['bin_true'].extend(target)
                    self.bin_dict['bin'+str(i)+'(<'+str(bin_th)+')']['bin_pred'].extend(output)
                    self.bin_dict['bin'+str(i)+'(<'+str(bin_th)+')']['bin_path'].extend(path)
                    break

    def reset_bin_dict(self):
        self.bin_dict = {}
        for i in range(0, self.n_bins):
            bin_th = min(round(self.bin_step*i + self.bin_step,2), 1.0)         # The min is when small fractions exceed 1.0
            self.bin_dict['bin'+str(i)+'(<'+str(bin_th)+')'] = {'bin_true':[], 'bin_pred':[], 'bin_path':[]}     # Each bin is also a dictionary to store labels & predictions (the keys)

    def Bin_accuracy(self, bin_data):
        # Number of Correctly classified samples over the total
        targets = bin_data['bin_true']
        predictions = bin_data['bin_pred']
        predicted_labels = []
        true_labels = []
        for (prediction, target) in zip(predictions, targets):
            predicted_labels.append(np.argmax(prediction)) 
            true_labels.append(np.argmax(target))
        assert(len(true_labels) == len(predicted_labels)) 
        i = 0
        correct_predictions = 0 
        while i<len(true_labels):
            if true_labels[i] == predicted_labels[i]:
                correct_predictions += 1
            i += 1
        acc = correct_predictions/(len(true_labels)+10e-14)
        return acc

    def Bin_confidence(self, bin_data):
        # The average confidence experienced in this bin
        predictions = bin_data['bin_pred']
        confidences = []
        for prediction in predictions:
            confidences.append(np.max(prediction))
        assert(len(confidences) == len(predictions))
        avg_confidence = (np.sum(confidences))/(len(predictions)+10e-14)
        return avg_confidence

    def Exp_Cal_Error(self):
        # Average expected Calibration error across all bins
        ECE = 0
        for bin_key in self.bin_dict.keys():
            bin_acc = Bin_accuracy(bin_dict[bin_key])
            bin_conf = Bin_confidence(bin_dict[bin_key])
            bin_weight = len(bin_dict[bin_key]['bin_true'])/self.total_samples
            diff = abs(bin_acc - bin_conf)
            ECE += bin_weight*diff
        return ECE