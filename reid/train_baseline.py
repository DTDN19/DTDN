from __future__ import print_function, absolute_import
import time

import torch
import torch.nn.functional as F
from torch.autograd import Variable

from .evaluation_metrics import accuracy
from .loss import TripletLoss
from .utils.meters import AverageMeter
import pdb
import random
import numpy as np
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter


class BaseTrainer(object):
    def __init__(self, model, criterion, criterion_trip=None, InvNet=None):
        super(BaseTrainer, self).__init__()
        self.model = model
        self.criterion = criterion
        self.criterion_trip = criterion_trip
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.write = SummaryWriter(log_dir="./")
        self.InvNet = InvNet

    def train(self, epoch, data_loader, optimizer, print_freq=1):
        self.model[0].train()
        self.model[1].train()
        self.model[2].train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses_IDE_s = AverageMeter()
        losses_IDE_t = AverageMeter()
        losses_neightboor = AverageMeter()
        losses_agreement = AverageMeter()


        # To make sure the longest loader is consumed, we cycle the other one.

        
        src_loader, tgt_loader = data_loader
        from itertools import cycle, tee
        if len(src_loader) < len(tgt_loader):
            src_loader = cycle(src_loader)
        elif len(src_loader) > len(tgt_loader):
            tgt_loader = cycle(tgt_loader)
        src_loader, src_pad = tee(src_loader)
        tgt_loader, tgt_pad = tee(tgt_loader)

        end = time.time()
        for i, src_inputs in enumerate(src_loader):
            data_time.update(time.time() - end)

            inputs_source, pids_source, pindexs_source = self._parse_data(src_inputs)
            # print (pindexs_target)


            # print(inputs_source.shape, pids_source.shape, pindexs_source.shape)
            if inputs_source.size(0) < 128:
                new_inputs = next(src_pad)
                x, y, z = self._parse_data(new_inputs)
                inputs_source = torch.cat([inputs_source, x])[:128]
                pids_source = torch.cat([pids_source, y])[:128]
                pindexs_source = torch.cat([pindexs_source, z])[:128]


            loss_sc_sa = self._forward([inputs_source], pids_source, epoch)
            loss = loss_sc_sa
            
            # from IPython import embed; embed()
            optimizer[0].zero_grad()
            optimizer[1].zero_grad()
            optimizer[2].zero_grad()
            loss.backward()
            optimizer[2].step()
            optimizer[1].step()
            optimizer[0].step()
                
            losses_IDE_s.update(loss_sc_sa.item(), pids_source.size(0))

            batch_time.update(time.time() - end)
            end = time.time()


            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}] \t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'task_1 {:.3f} ({:.3f})\t'
                      .format(epoch, i + 1, max(map(len, data_loader)),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses_IDE_s.val, losses_IDE_s.avg))

    def _parse_data(self, inputs):
        raise NotImplementedError

    def _forward(self, inputs, targets):
        raise NotImplementedError


class Trainer(BaseTrainer):
    def _parse_data(self, inputs):
        imgs, _, pids, _, pindexs = inputs
        inputs = imgs.to(self.device)
        pids = pids.to(self.device)
        pindexs = pindexs.to(self.device)

        return inputs, pids, pindexs

    def _forward(self, inputs, targets, epoch, update_only=False):
        outputs_source,_ = self.model[1](inputs[0])
        loss_sc_sa = self.criterion[0](outputs_source, targets)
        return loss_sc_sa
