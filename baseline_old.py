from __future__ import print_function, absolute_import
import argparse
import os.path as osp

import numpy as np
import sys
import torch

from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader

from reid.datasets.domain_adaptation import DA
from reid import models
from reid.loss import TripletLoss, InvNet
from reid.trainers import Trainer
from reid.evaluators import Evaluator
from reid.utils.data import transforms as T
from reid.utils.data.preprocessor import Preprocessor
#from reid.utils.data.sampler import RandomIdentitySampler
from reid.utils.logging import Logger
from reid.utils.serialization import load_checkpoint, save_checkpoint

from reid.utils.CrossEntropyLabelSmooth import CrossEntropyLabelSmooth


def get_data(data_dir, source, target, height, width, batch_size, re=0, workers=8):

    dataset = DA(data_dir, source, target)

    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    num_classes = dataset.num_train_ids
    target_num_classes = dataset.target_pindex

    train_transformer = T.Compose([
        T.Resize((height, width)),
        T.RandomCrop((256,128)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        normalizer,
        T.RandomErasing(EPSILON=re),
    ])
    # train_transformer = T.Compose([
    #     T.RandomSizedRectCrop(height, width),
    #     T.RandomHorizontalFlip(),
    #     T.ToTensor(),
    #     normalizer,
    #     T.RandomErasing(EPSILON=re),
    # ])

    test_transformer = T.Compose([
        T.Resize((height, width), interpolation=3),
        T.ToTensor(),
        normalizer,
    ])

    source_train_loader = DataLoader(
        Preprocessor(dataset.source_train, root=osp.join(dataset.source_images_dir, dataset.source_train_path),
                     transform=train_transformer),
        batch_size=batch_size, num_workers=workers,
        #sampler=RandomIdentitySampler(dataset.source_train, 2),
        shuffle=True, pin_memory=True, drop_last=True)
    target_train_loader = DataLoader(
        Preprocessor(dataset.target_train, root=osp.join(dataset.target_images_dir, dataset.target_train_path),
                     transform=train_transformer),
        batch_size=batch_size, num_workers=workers,
        #sampler=RandomIdentitySampler(dataset.source_train, 2),
        shuffle=True, pin_memory=True, drop_last=True)
    query_loader = DataLoader(
        Preprocessor(dataset.query,
                     root=osp.join(dataset.target_images_dir, dataset.query_path), transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)
    gallery_loader = DataLoader(
        Preprocessor(dataset.gallery,
                     root=osp.join(dataset.target_images_dir, dataset.gallery_path), transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    return dataset, num_classes, target_num_classes, source_train_loader, target_train_loader, query_loader, gallery_loader

def cross_entropy_with_label_smoothing(pred, target, label_smoothing=0.1):
    logsoftmax = nn.logsoftmax()
    num_classes = pred.size(1)
    target = torch.unsqueeze(target,1)
    soft_target = torch.zeros_like(pred)
    soft_target.scatter_(1, target, 1)
    soft_target = soft_target * (1 - label_smoothing) + label_smoothing / num_classes
    return torch.mean(torch.sum(-soft_target * logsoftmax(pred),1))

def main(args):
    cudnn.benchmark = True

    # vehicle:
    # target_num_classes = 37778
    # target_num_classes = 113346
    # dukenum:
    # target_num_classes = 16522 
    # # marketnum:
    # target_num_classes = 12937
    # target-feature-memory


    # Redirect print to both console and log file
    if not args.evaluate:
        sys.stdout = Logger(osp.join(args.logs_dir, 'log.txt'))

    # Create data loaders
    dataset, num_classes, target_num_classes, source_train_loader, target_train_loader, query_loader, gallery_loader = \
        get_data(args.data_dir, args.source, args.target, args.height,
                 args.width, args.batch_size, args.re, args.workers)
    print ("num_classses:", num_classes)
    print ("target_num_classes:", target_num_classes)

    # target_num_classes = 12936

    # Create model
    Encode, IdeNet, AttentionMask = models.create(args.arch, num_features=args.features,
                          dropout=args.dropout, num_classes=num_classes, cut_layer='layer3', target_num=target_num_classes)

    # Load from checkpoint
    start_epoch = 0
    if args.resume:
        checkpoint = load_checkpoint(args.resume)
        Encode.load_state_dict(checkpoint['Encode'])
        AttentionMask.load_state_dict(checkpoint['AttentionMask'])
        IdeNet.load_state_dict(checkpoint['IdeNet'])
        start_epoch = checkpoint['epoch']
        print("=> Start epoch {} "
              .format(start_epoch))

    Encode = nn.DataParallel(Encode).cuda()
    IdeNet = nn.DataParallel(IdeNet).cuda()
    AttentionMask = nn.DataParallel(AttentionMask).cuda()

    model = [Encode, IdeNet, AttentionMask]

    # Evaluator
    evaluator = Evaluator([Encode, IdeNet, AttentionMask])
    if args.evaluate:
        print("Test:")
        evaluator.evaluate(query_loader, gallery_loader, dataset.query, dataset.gallery, args.output_feature, args.rerank, save_dir=None, querydataset=dataset.query, gallerydataset=dataset.gallery)
        # evaluator.evaluate(query_loader, gallery_loader, dataset.query, dataset.gallery, args.output_feature, args.rerank)
        # from reid.utils.tensor_saver import SAVER
        # SAVER.save()
        return

    # Criterion
    criterion = []
    criterion.append(nn.CrossEntropyLoss().cuda())
    # criterion.append(CrossEntropyLabelSmooth(num_classes).cuda())
    criterion.append(TripletLoss(margin=args.margin).cuda())
    # criterion = cross_entropy_with_label_smoothing().cuda()

    # Optimizer Encode
    base_param_ids = set(map(id, Encode.module.base.parameters()))
    new_params = [p for p in Encode.parameters() if
                    id(p) not in base_param_ids]
    param_groups = [
        {'params': Encode.module.base.parameters(), 'lr_mult': 0.1},
        {'params': new_params, 'lr_mult': 1.0}
    ]
    optimizer_Encode = torch.optim.SGD(param_groups, lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                nesterov=True)

    # Optimizer Ide
    base_param_ids = set(map(id, IdeNet.module.base.parameters()))
    new_params = [p for p in IdeNet.parameters() if
                    id(p) not in base_param_ids]
    param_groups = [
        {'params': IdeNet.module.base.parameters(), 'lr_mult': 0.1},
        {'params': new_params, 'lr_mult': 1.0}
    ] 
    optimizer_Ide = torch.optim.SGD(param_groups, lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                nesterov=True)
    param_groups = [
        {'params':AttentionMask.module.parameters(), 'lr_mult':1.0},
    ]
    optimizer_Att = torch.optim.SGD(param_groups, lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                nesterov=True)
    if target_num_classes > 0:
            # targetMemory = torch.zeros((target_num_classes, args.features)).cuda()
            invNet = InvNet(args.features, target_num_classes, beta=0.05, knn=6, alpha=0.01)
            invNet.cuda()

    # Trainer
    trainer = Trainer([Encode, IdeNet, AttentionMask], criterion, InvNet=invNet)

    # Schedule learning rate
    def adjust_lr(epoch):
        step_size = 10
        lr = args.lr * (0.1 ** (epoch // step_size))
        print("lr",lr)
        # if epoch <= 5:
        #     lr = 0.08
        # elif epoch <= 10:
        #     lr = 0.0008
        # elif epoch <= 15:
        #     lr = 0.0001
        # else:
        #     lr = 0.00001
        # lr = args.lr * 0.9 ** epoch
        for g in optimizer_Encode.param_groups:
            g['lr'] = lr * g.get('lr_mult', 1)
        for g in optimizer_Ide.param_groups:
            g['lr'] = lr * g.get('lr_mult', 1)
        for g in optimizer_Att.param_groups:
            g['lr'] = lr * g.get('lr_mult', 1)

    tmp=best=0
    # Start training
    for epoch in range(start_epoch, args.epochs):
        adjust_lr(epoch)
        trainer.train(epoch, [source_train_loader, target_train_loader], [optimizer_Encode, optimizer_Ide, optimizer_Att], target_num_classes=target_num_classes)

        save_checkpoint({
            'Encode': Encode.module.state_dict(),
            'IdeNet': IdeNet.module.state_dict(),
            'AttentionMask': AttentionMask.module.state_dict(),
            'InvNet': invNet.state_dict(), 
            'epoch': epoch + 1,
        }, fpath=osp.join(args.logs_dir, 'checkpoint.pth.tar'))

        # from reid.utils.tensor_saver import SAVER
        # SAVER.save()
        # raise Exception('Saver finished.')
        # evaluator = Evaluator(model)
        if epoch % 1 == 0:tmp=evaluator.evaluate(query_loader, gallery_loader, dataset.query, dataset.gallery, args.output_feature, args.rerank)
        if(tmp>best):
            save_checkpoint({
            'Encode': Encode.module.state_dict(),
            'IdeNet': IdeNet.module.state_dict(),
            'AttentionMask': AttentionMask.module.state_dict(),
            'InvNet': invNet.state_dict(), 
            'epoch': epoch + 1,
            }, fpath=osp.join(args.logs_dir, 'best_checkpoint.pth.tar'))
            best=tmp
        print('Best Rank-1:{:.1f}%'.format(best*100))

        print('\n * Finished epoch {:3d} \n'.
              format(epoch))
        

    # Final test
    print('Test with best model:')
    evaluator = Evaluator(model)
    evaluator.evaluate(query_loader, gallery_loader, dataset.query, dataset.gallery, args.output_feature, args.rerank)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="baseline")
    # source
    parser.add_argument('-s', '--source', type=str, default='market1501',
                        choices=['market1501_', 'DukeMTMC-reID_', 'msmt', 'cuhk03_detected', 'VeRi', 'VehicleID_V1.0'])
    # target
    parser.add_argument('-t', '--target', type=str, default='market1501',
                        choices=['market1501_', 'DukeMTMC-reID_', 'msmt', 'viper', 'VeRi', 'VehicleID_V1.0'])
    # images
    parser.add_argument('-b', '--batch-size', type=int, default=128, help="batch size for source")
    parser.add_argument('-j', '--workers', type=int, default=8)
    parser.add_argument('--height', type=int, default=256,
                        help="input height, default: 256")
    parser.add_argument('--width', type=int, default=128,
                        help="input width, default: 128")
    # model
    parser.add_argument('-a', '--arch', type=str, default='resnet50',
                        choices=models.names())
    parser.add_argument('--features', type=int, default=1024)
    parser.add_argument('--dropout', type=float, default=0.5)
    # optimizer
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    # training configs
    parser.add_argument('--resume', type=str, default='', metavar='PATH')
    parser.add_argument('--evaluate', action='store_true',
                        help="evaluation only")
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--print-freq', type=int, default=1)
    # metric learning
    parser.add_argument('--dist-metric', type=str, default='euclidean')
    # misc
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'data'))
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'logs'))
    parser.add_argument('--output_feature', type=str, default='pool5')
    #random erasing
    parser.add_argument('--re', type=float, default=0)
    #  perform re-ranking
    parser.add_argument('--rerank', action='store_true', help="perform re-ranking")

    parser.add_argument('--lambda-adv', type=float, default=0.015)

    parser.add_argument('--lambda-trip', type=float, default=0.03)

    parser.add_argument('--margin', type=float, default=0.7)

    

    main(parser.parse_args())
