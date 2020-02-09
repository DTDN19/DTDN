<<<<<<< HEAD
import torch
import torch.nn.functional as F
from torch import nn, autograd
from torch.autograd import Variable, Function
import numpy as np
import math

TARGET = dict()

class InvNet_office(nn.Module):
    def __init__(self, num_features, num_classes, beta=0.05, knn=6, alpha=0.01):
        super(InvNet_office, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_features = num_features
        self.num_classes = num_classes
        self.alpha = alpha  
        self.beta = beta 
        self.knn = knn  

        self.em = nn.Parameter(torch.zeros(num_classes, num_features), requires_grad=False)
        self.untouched_targets = set(range(num_classes))

    def forward(self, inputs, label, epoch=None, fnames_target=None, predict_class=None, model=None):
        alpha = self.alpha * epoch
        if model is not None:
            tgt_feature = inputs.mm(self.em.t())
            tgt_class = model.module.classifier(self.em)
        else:
            tgt_feature = inputs.mm(self.em.t())
            tgt_class = None
        tgt_feature /= self.beta

        loss = self.smooth_loss(tgt_feature, tgt_label,predict_class,tgt_class)

        for x, y in zip(inputs, label):
            self.em.data[y] = alpha * self.em.data[y]  + (1. - alpha) * x.data
            self.em.data[y] /= self.em.data[y].norm()
            
        return loss

    def smooth_loss(self, tgt_feature, tgt_label,predict_class, tgt_class):
        '''
        tgt_feature: [128, 16522], similarity of batch & targets
        tgt_label: see forward
        '''        
        if tgt_class is not None:
            outputs_class = torch.abs(predict_class.argmax(dim=1).unsqueeze(1).repeat(1,tgt_class.shape[0])-tgt_class.argmax(dim=1))
            # from IPython import embed; embed();exit(0)
            outputs_class[outputs_class != 0] = 1
            outputs_class = 1.0 - outputs_class
        else:
            outputs_class = None
        mask = self.smooth_hot(tgt_feature.detach().clone(), tgt_label.detach().clone(), outputs_class, predict_class,self.knn)

        outputs = F.log_softmax(tgt_feature, dim=1)
        loss = - (mask * outputs)
        loss = loss.sum(dim=1)
        loss = loss.mean(dim=0)
        return loss

    def smooth_hot(self, tgt_feature, targets, outputs_class,predict_class, k=6):
        '''
        see smooth_loss
        '''

        mask = torch.zeros(tgt_feature.size()).to(self.device)
        if outputs_class is not None:
            print (tgt_feature.min())
            tgt_mask = tgt_feature.detach().clone()
            tgt_mask[outputs_class[:,0],outputs_class[:,1]] = -1e5
            _, topk = tgt_mask.topk(k, dim=1)
        else:
            _, topk = tgt_feature.topk(k, dim=1)
        mask.scatter_(1, topk, 1.0)

        index_2d = targets[..., None]
        mask.scatter_(1, index_2d, 2.0)

        return mask
=======
import torch
import torch.nn.functional as F
from torch import nn, autograd
from torch.autograd import Variable, Function
import numpy as np
import math

TARGET = dict()


class ExemplarMemory(Function):
    def __init__(self, em, alpha=0.01):
        super(ExemplarMemory, self).__init__()
        self.em = em
        self.alpha = alpha

    def forward(self, inputs, targets, classification=False):
        self.save_for_backward(inputs, targets)
        # if classification:
        # return self.em
        outputs = inputs.mm(self.em.t())
        return outputs

    def backward(self, grad_outputs):
        inputs, targets = self.saved_tensors
        grad_inputs = None
        if self.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(self.em)
            # grad_inputs = inputs
        for x, y in zip(inputs, targets):
            self.em[y] = self.alpha * self.em[y] + (1. - self.alpha) * x
            self.em[y] /= self.em[y].norm()
        return grad_inputs, None

class InvNet_office(nn.Module):
    def __init__(self, num_features, num_classes, beta=0.05, knn=6, alpha=0.01):
        super(InvNet_office, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_features = num_features
        self.num_classes = num_classes
        self.alpha = alpha  # Memory update rate
        self.beta = beta  # Temperature fact
        self.knn = knn  # Knn for neighborhood invariance

        # Exemplar memory
        self.em = nn.Parameter(torch.zeros(num_classes, num_features))
        self.untouched_targets = set(range(num_classes))

    def forward(self, tgt_feature, tgt_label, epoch=None, fnames_target=None, predict_class=None, model=None):
        '''
        tgt_feature: [128, 2048], each t's 2048-d feature
        tgt_label: [128], each t's label
        '''
        # x, y, fnames = tgt_feature, tgt_label, fnames_target
        alpha = self.alpha * epoch
        if model is not None:
            EM  = ExemplarMemory(self.em, alpha=alpha)            
            tgt_feature = EM(tgt_feature, tgt_label)
            tgt_class = model.module.classifier(EM.em)
        else:
            tgt_feature = ExemplarMemory(self.em, alpha=alpha)(tgt_feature, tgt_label)
            tgt_class = None
        tgt_feature /= self.beta

        loss = self.smooth_loss(tgt_feature, tgt_label,predict_class,tgt_class)
        return loss

    def smooth_loss(self, tgt_feature, tgt_label,predict_class, tgt_class):
        '''
        tgt_feature: [128, 16522], similarity of batch & targets
        tgt_label: see forward
        '''        
        if tgt_class is not None:
            # outputs_class = (predict_class.argmax(dim=1).unsqueeze(1).float()).mm(tgt_class.argmax(dim=1).unsqueeze(1).float().t())
            # outputs_class = predict_class.mm(tgt_class.t())
            # outputs_class = F.log_softmax(outputs_class, dim=1)
            # print (outputs_class.shape)
            outputs_class = torch.abs(predict_class.argmax(dim=1).unsqueeze(1).repeat(1,tgt_class.shape[0])-tgt_class.argmax(dim=1))
            # from IPython import embed; embed();exit(0)
            outputs_class[outputs_class != 0] = 1
            outputs_class = 1.0 - outputs_class

            # print (outputs_class.shape)

            # import numpy as np
            # predict_target = torch.from_numpy(np.repeat(predict_class.cpu().numpy(),self.knn)).cuda()
            # print (predict_target.shape)
        else:
            outputs_class = None
        mask = self.smooth_hot(tgt_feature.detach().clone(), tgt_label.detach().clone(), outputs_class, predict_class,self.knn)

        outputs = F.log_softmax(tgt_feature, dim=1)
        loss = - (mask * outputs)
        loss = loss.sum(dim=1)
        loss = loss.mean(dim=0)
        return loss

    def smooth_hot(self, tgt_feature, targets, outputs_class,predict_class, k=6):
        '''
        see smooth_loss
        '''

        mask = torch.zeros(tgt_feature.size()).to(self.device)
        if outputs_class is not None:
            print (tgt_feature.min())
            tgt_mask = tgt_feature.detach().clone()
            tgt_mask[outputs_class[:,0],outputs_class[:,1]] = -1e5
            _, topk = tgt_mask.topk(k, dim=1)
        else:
            _, topk = tgt_feature.topk(k, dim=1)
        # for i in range(len(topk)):
        #     print (targets[i], ":", topk[i])
        # mask.scatter_(1, topk, 1.0/k) #GY: 1.0 -> 1.0/k
        mask.scatter_(1, topk, 1.0/k) #GY: 1.0 -> 1.0/k

        
        # if predict_class is not None:
        #     for i,c in enumerate(predict_class):
        #         if c ==10:
        #             mask[i] *= 0

        index_2d = targets[..., None]
        mask.scatter_(1, index_2d, 1.0)
        print("mask:", mask.sum())


        # for i in range(len(targets)):
        #     if TARGET[targets[i].item()]== 10:
        #         print(predict_class[i])
        # # test knn
        # for i in targets:
        #     print (TARGET[i.item()])
        # from collections import defaultdict
        # a = defaultdict(list)
        # for sim, label in zip(tgt_mask, targets):
        #     assert len(sim) == len(TARGET)
        #     label = TARGET[label.item()]
        #     _, topk = sim.topk(k)
        #     a[label].append(sum([TARGET[i.item()] == label for i in topk]))
        # for k, v in a.items():
        #     print(k, v)
        # from IPython import embed; embed();exit(0)
        

        return mask
>>>>>>> f0906cafd587b9f863e29ed0904c7c6f81d0db32
