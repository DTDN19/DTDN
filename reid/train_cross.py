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

        end = time.time()
        for i, src_inputs in enumerate(src_loader):
            data_time.update(time.time() - end)

            inputs_source, pids_source, pindexs_source = self._parse_data(src_inputs)
            # print (pindexs_target)


            # print(inputs_source.shape, pids_source.shape, pindexs_source.shape)
            if inputs_source.size(0) < 64:
                new_inputs = next(src_pad)
                x, y, z = self._parse_data(new_inputs)
                inputs_source = torch.cat([inputs_source, x])[:64]
                pids_source = torch.cat([pids_source, y])[:64]
                pindexs_source = torch.cat([pindexs_source, z])[:64]


            loss_sc_sa, loss_neightboor, loss_query = self._forward([inputs_source], pids_source, epoch)
            if epoch == 0:
                loss = loss_sc_sa + 0* loss_neightboor +  0 * loss_query
            else:
                loss = loss_sc_sa + 0* loss_neightboor +  loss_query

            
            # from IPython import embed; embed()
            optimizer[0].zero_grad()
            optimizer[1].zero_grad()
            optimizer[2].zero_grad()
            loss.backward()
            optimizer[2].step()
            optimizer[1].step()
            optimizer[0].step()
                
            losses_IDE_s.update(loss_sc_sa.item(), pids_source.size(0))
            losses_IDE_t.update(loss_query.item(),pids_source.size(0))
            batch_time.update(time.time() - end)
            end = time.time()


            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}] '
                    'Time {:.3f} ({:.3f}) '
                    'Data {:.3f} ({:.3f}) '
                    'task_1 {:.3f} ({:.3f}) '
                    'task_2 {:.3f} ({:.3f})'
                    .format(epoch, i + 1, max(map(len, data_loader)),
                            batch_time.val, batch_time.avg,
                            data_time.val, data_time.avg,
                            losses_IDE_s.val, losses_IDE_s.avg,
                            losses_IDE_t.val, losses_IDE_t.avg))

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
        targets_gallery, targets_query = targets[0::2], targets[1::2]
        inputs_gallery, inputs_query = inputs[0][0::2], inputs[0][1::2]

        outputs_gallery, _ = self.model[1](inputs_gallery)
        outputs_query, feature_query = self.model[1](inputs_query)

        outputs_source, _ = self.model[1](inputs[0])
        loss_sc_sa = self.criterion[0](outputs_source, targets)
        
        # loss_sc_sa = self.criterion[0](outputs_query, targets_query)


        # gallery
        gallery_memory = self.InvNet.em.clone()
        # gallery_memory = torch.zeros(self.InvNet.em.shape).cuda()
        _, tgt_feature = self.model[1](inputs[0][0::2], tgt_output_feature='pool5')

        for i,t in enumerate(targets_gallery):
            gallery_memory[t] = tgt_feature[i]

        loss_query = self.criterion[0](feature_query.mm(gallery_memory.t()), targets_query)     
        target_loss = self.InvNet(tgt_feature, targets_gallery, epoch=epoch)
        # gallery_memory = self.InvNet.em

        return loss_sc_sa,target_loss, loss_query
