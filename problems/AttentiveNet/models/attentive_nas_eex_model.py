import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import math

from .modules.exit_block import ExitBlock
from .modules.nn_base import MyNetwork

class AttentiveNasEExModel(MyNetwork):

    def __init__(self, first_conv, blocks, num_blocks, last_conv, classifier, confidence, resolution, 
                        block_ee, num_ee, use_v3_head=True):

        
        super(AttentiveNasEExModel, self).__init__()

        self.layers = nn.ModuleList()
        self.stages = nn.ModuleList()
        self.exits = nn.ModuleList()
        self.num_ee = num_ee
        self.block_ee = block_ee
        self.exit_threshold = 0.5
        self.num_classes = classifier.out_features

        self.first_conv = first_conv

        for i in range(num_blocks):
            self.layers.append(blocks[i])
            if(i in block_ee): # the block has been chosen for exit
                in_features = blocks[i].mobile_inverted_conv.out_channels
                self.add_exit_block(in_features)

        self.last_conv = last_conv

        self.classifier = classifier
        self.confidence = confidence

        self.stages.append(nn.Sequential(*self.layers))

        self.resolution = resolution 
        self.use_v3_head = use_v3_head

    def add_exit_block(self, in_features):
        self.stages.append(nn.Sequential(*self.layers))
        self.exits.append(ExitBlock(in_features, self.num_classes))
        self.layers = nn.ModuleList()

    def forward(self, x):
        preds, confs = [], []

        # resize input to target resolution first
        if x.size(-1) != self.resolution:
            x = torch.nn.functional.interpolate(x, size=self.resolution, mode='bicubic')

        x = self.first_conv(x)

        for idx, exitblock in enumerate(self.exits):
            x = self.stages[idx](x)
            pred, conf = exitblock(x)
            if not self.training and conf.item() > self.exit_threshold:
                return pred, idx
            preds.append(pred)
            confs.append(conf)

        x = self.stages[-1](x)
        x = self.last_conv(x)

        if not self.use_v3_head:
            x = x.mean(3, keepdim=True).mean(2, keepdim=True)  # global average pooling

        x = torch.squeeze(x)
        pred = self.classifier(x)
        conf = self.confidence(x)

        if not self.training:
            return pred, len(self.exits), 1.0

        preds.append(pred)
        confs.append(conf)

        return preds, confs

        
    @property
    def module_str(self):
        _str = self.first_conv.module_str + '\n'
        for block in self.blocks:
            _str += block.module_str + '\n'
        #_str += self.last_conv.module_str + '\n'
        _str += self.classifier.module_str
        return _str

    @property
    def config(self):
        return {
            'name': AttentiveNasEExModel.__name__,
            'bn': self.get_bn_param(),
            'first_conv': self.first_conv.config,
            'blocks': [
                block.config for block in self.blocks
            ],
            #'last_conv': self.last_conv.config,
            'classifier': self.classifier.config,
            'resolution': self.resolution
        }


    def weight_initialization(self):
        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    @staticmethod
    def build_from_config(config):
        raise NotImplementedError


    def reset_running_stats_for_calibration(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.SyncBatchNorm):
                m.training = True
                m.momentum = None # cumulative moving average
                m.reset_running_stats()

