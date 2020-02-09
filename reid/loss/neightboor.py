import torch
import torch.nn.functional as F
from torch import nn, autograd
from torch.autograd import Variable, Function
import numpy as np
import math


<<<<<<< HEAD
=======
class ExemplarMemory(Function):
    def __init__(self, em, alpha=0.01):
        super(ExemplarMemory, self).__init__()
        self.em = em
        self.alpha = alpha

    def forward(self, inputs, targets):
        self.save_for_backward(inputs, targets)
        outputs = inputs.mm(self.em.t())
        return outputs

    def backward(self, grad_outputs):
        inputs, targets = self.saved_tensors
        grad_inputs = None
        if self.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(self.em)
        for x, y in zip(inputs, targets):
            self.em[y] = self.alpha * self.em[y] + (1. - self.alpha) * x
            self.em[y] /= self.em[y].norm()
        return grad_inputs, None

>>>>>>> f0906cafd587b9f863e29ed0904c7c6f81d0db32
class InvNet(nn.Module):
    def __init__(self, num_features, num_classes, beta=0.05, knn=6, alpha=0.01):
        super(InvNet, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_features = num_features
        self.num_classes = num_classes
<<<<<<< HEAD
        self.alpha = alpha  
        self.beta = beta  
        self.knn = knn  

        self.em = nn.Parameter(torch.zeros(num_classes, num_features), requires_grad=False)
        self.untouched_targets = set(range(num_classes))

    def forward(self, inputs, label, epoch=None, fnames_target=None):
        alpha = self.alpha * epoch
        tgt_feature = inputs.mm(self.em.t())
        tgt_feature /= self.beta

        loss = self.smooth_loss(tgt_feature, label)

        for x, y in zip(inputs, label):
            self.em.data[y] = alpha * self.em.data[y]  + (1. - alpha) * x.data
            self.em.data[y] /= self.em.data[y].norm()
        return loss

    def smooth_loss(self, tgt_feature, tgt_label):
=======
        self.alpha = alpha  # Memory update rate
        self.beta = beta  # Temperature fact
        self.knn = knn  # Knn for neighborhood invariance

        # Exemplar memory
        self.em = nn.Parameter(torch.zeros(num_classes, num_features))
        self.untouched_targets = set(range(num_classes))

    def forward(self, tgt_feature, tgt_label, epoch=None, fnames_target=None):
        '''
        tgt_feature: [128, 2048], each t's 2048-d feature
        tgt_label: [128], each t's label
        '''
        # x, y, fnames = tgt_feature, tgt_label, fnames_target
        alpha = self.alpha * epoch
        tgt_feature = ExemplarMemory(self.em, alpha=alpha)(tgt_feature, tgt_label)
        tgt_feature /= self.beta

        loss = self.smooth_loss(tgt_feature, tgt_label)
        return loss

    def smooth_loss(self, tgt_feature, tgt_label):
        '''
        tgt_feature: [128, 16522], similarity of batch & targets
        tgt_label: see forward
        '''
>>>>>>> f0906cafd587b9f863e29ed0904c7c6f81d0db32
        mask = self.smooth_hot(tgt_feature.detach().clone(), tgt_label.detach().clone(), self.knn)
        outputs = F.log_softmax(tgt_feature, dim=1)
        loss = - (mask * outputs)
        loss = loss.sum(dim=1)
        loss = loss.mean(dim=0)
        return loss

    def smooth_hot(self, tgt_feature, targets, k=6):
<<<<<<< HEAD
        mask = torch.zeros(tgt_feature.size()).to(self.device)

        k=7

        _, topk = tgt_feature.topk(k, dim=1)
        mask.scatter_(1, topk, 2)
=======
        '''
        see smooth_loss
        '''
        mask = torch.zeros(tgt_feature.size()).to(self.device)
        # d-m:k=6
        # m-d:k=8
        _, topk = tgt_feature.topk(k, dim=1)
        mask.scatter_(1, topk, 2)
        # for f, m, t in zip(tgt_feature, mask, topk):
            # m[t] = 6 * F.normalize(f[t], dim=0)
        
        # mask.scatter_(1, topk[:2], 2)
        # mask.scatter_(1, topk[2:4], 1.0)
        # mask.scatter_(1, topk[4:], 0.5)
>>>>>>> f0906cafd587b9f863e29ed0904c7c6f81d0db32

        index_2d = targets[..., None]
        mask.scatter_(1, index_2d, 3)

        return mask
