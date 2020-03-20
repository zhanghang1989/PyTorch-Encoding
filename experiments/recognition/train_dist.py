##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Hang Zhang
## Email: zhanghang0704@gmail.com
## Copyright (c) 2020
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree 
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

from __future__ import print_function
import os
import time
import argparse
from tqdm import tqdm
import torch.multiprocessing as mp
try:
    from mpi4py import MPI
except ImportError:
    logging.info('mpi4py is not installed. Use "pip install --no-cache mpi4py" to install')
    MPI = None

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
from torch.multiprocessing import Process

import encoding

class Options():
    def __init__(self):
        # Training settings
        parser = argparse.ArgumentParser(description='Deep Encoding')
        parser.add_argument('--dataset', type=str, default='cifar10',
                            help='training dataset (default: cifar10)')
        parser.add_argument('--base-size', type=int, default=256,
                            help='base image size')
        parser.add_argument('--crop-size', type=int, default=224,
                            help='crop image size')
        # model params 
        parser.add_argument('--model', type=str, default='densenet',
                            help='network model type (default: densenet)')
        parser.add_argument('--pretrained', action='store_true', 
                            default=False, help='load pretrianed mode')
        parser.add_argument('--rectify', action='store_true', 
                            default=False, help='rectify convolution')
        # training hyper params
        parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                            help='batch size for training (default: 128)')
        parser.add_argument('--test-batch-size', type=int, default=256, metavar='N',
                            help='batch size for testing (default: 256)')
        parser.add_argument('--epochs', type=int, default=120, metavar='N',
                            help='number of epochs to train (default: 600)')
        parser.add_argument('--start_epoch', type=int, default=1, 
                            metavar='N', help='the epoch number to start (default: 1)')
        parser.add_argument('--workers', type=int, default=8,
                            metavar='N', help='dataloader threads')
        # lr setting
        parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                            help='learning rate (default: 0.1)')
        parser.add_argument('--lr-scheduler', type=str, default='cos', 
                            help='learning rate scheduler (default: cos)')
        # optimizer
        parser.add_argument('--momentum', type=float, default=0.9, 
                            metavar='M', help='SGD momentum (default: 0.9)')
        parser.add_argument('--weight-decay', type=float, default=1e-4, 
                            metavar ='M', help='SGD weight decay (default: 1e-4)')
        # cuda, seed and logging
        parser.add_argument('--seed', type=int, default=1, metavar='S',
                            help='random seed (default: 1)')
        # checking point
        parser.add_argument('--resume', type=str, default=None,
                            help='put the path to resuming file if needed')
        parser.add_argument('--checkname', type=str, default='default',
                            help='set the checkpoint name')
        # distributed
        parser.add_argument('--world-size', default=1, type=int,
                            help='number of nodes for distributed training')
        parser.add_argument('--rank', default=0, type=int,
                            help='node rank for distributed training')
        parser.add_argument('--dist-url', default='tcp://localhost:23456', type=str,
                            help='url used to set up distributed training')
        parser.add_argument('--dist-backend', default='nccl', type=str,
                            help='distributed backend')
        self.parser = parser

    def parse(self):
        args = self.parser.parse_args()
        return args

def main():
    args = Options().parse()
    ngpus_per_node = torch.cuda.device_count()
    args.world_size = ngpus_per_node * args.world_size
    args.lr = args.lr * args.world_size
    mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))

# global variable
best_pred = 0.0
acclist_train = []
acclist_val = []

def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu
    args.rank = args.rank * ngpus_per_node + gpu
    print('rank: {} / {}'.format(args.rank, args.world_size))
    dist.init_process_group(backend=args.dist_backend,
                            init_method=args.dist_url,
                            world_size=args.world_size,
                            rank=args.rank)
    torch.cuda.set_device(args.gpu)
    # init the args
    global best_pred, acclist_train, acclist_val

    if args.gpu == 0:
        print(args)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    # init dataloader
    transform_train, transform_val = encoding.transforms.get_transform(args.dataset, args.base_size, args.crop_size)
    trainset = encoding.datasets.get_dataset(args.dataset, root=os.path.expanduser('~/.encoding/data'),
                                             transform=transform_train, train=True, download=True)
    valset = encoding.datasets.get_dataset(args.dataset, root=os.path.expanduser('~/.encoding/data'),
                                           transform=transform_val, train=False, download=True)

    train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True,
        sampler=train_sampler)

    val_sampler = torch.utils.data.distributed.DistributedSampler(valset)
    val_loader = torch.utils.data.DataLoader(
        valset, batch_size=args.test_batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True,
        sampler=val_sampler)
    
    # init the model
    model_kwargs = {'pretrained': args.pretrained}
    if args.rectify:
        from encoding.nn import RFConv2d
        #model_kwargs['conv_layer'] = RFConv2d
        model_kwargs['use_rfconv'] = True

    model = encoding.models.get_model(args.model, **model_kwargs)
    if args.gpu == 0:
        print(model)

    criterion = nn.CrossEntropyLoss()

    model.cuda(args.gpu)
    criterion.cuda(args.gpu)
    model = DistributedDataParallel(model, device_ids=[args.gpu])

    # criterion and optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    # check point
    if args.resume is not None:
        if os.path.isfile(args.resume):
            if args.gpu == 0:
                print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch'] + 1 if args.start_epoch == 1 else args.start_epoch
            best_pred = checkpoint['best_pred']
            acclist_train = checkpoint['acclist_train']
            acclist_val = checkpoint['acclist_val']
            model.module.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            if args.gpu == 0:
                print("=> loaded checkpoint '{}' (epoch {})"
                .format(args.resume, checkpoint['epoch']))
        else:
            raise RuntimeError ("=> no resume checkpoint found at '{}'".\
                format(args.resume))
    scheduler = encoding.utils.LR_Scheduler(args.lr_scheduler, args.lr, args.epochs,
                                            len(train_loader))
    def train(epoch):
        train_sampler.set_epoch(epoch)
        model.train()
        losses = AverageMeter()
        top1 = AverageMeter()
        global best_pred, acclist_train
        for batch_idx, (data, target) in enumerate(train_loader):
            scheduler(optimizer, batch_idx, epoch, best_pred)
            data, target = data.cuda(args.gpu), target.cuda(args.gpu)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            acc1 = accuracy(output, target, topk=(1,))
            top1.update(acc1[0], data.size(0))
            losses.update(loss.item(), data.size(0))
            if batch_idx % 100 == 0 and args.gpu == 0:
                print('Batch: %d| Loss: %.3f | Top1: %.3f'%(batch_idx, losses.avg, top1.avg))

        acclist_train += [top1.avg]

    def validate(epoch):
        model.eval()
        top1 = AverageMeter()
        top5 = AverageMeter()
        global best_pred, acclist_train, acclist_val
        is_best = False
        for batch_idx, (data, target) in enumerate(val_loader):
            data, target = data.cuda(args.gpu), target.cuda(args.gpu)
            with torch.no_grad():
                output = model(data)
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                top1.update(acc1[0], data.size(0))
                top5.update(acc5[0], data.size(0))

        if MPI is not None:
            comm = MPI.COMM_WORLD
            res1 = comm.gather(top1.avg, root=0)
            res2 = comm.gather(top5.avg, root=0)
            if args.gpu == 0:
                top1_acc = sum(res1) / len(res1)
                top5_acc = sum(res2) / len(res2)
                print('Validation: Top1: %.3f | Top5: %.3f'%(top1_acc, top5_acc))

        # save checkpoint
        acclist_val += [top1.avg]
        if top1.avg > best_pred:
            best_pred = top1.avg 
            is_best = True
        encoding.utils.save_checkpoint({
            'epoch': epoch,
            'state_dict': model.module.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_pred': best_pred,
            'acclist_train':acclist_train,
            'acclist_val':acclist_val,
            }, args=args, is_best=is_best)

    for epoch in range(args.start_epoch, args.epochs + 1):
        tic = time.time()
        train(epoch)
        validate(epoch)
        elapsed = time.time() - tic
        if args.gpu == 0:
            print(f'Epoch: {epoch}, Time cost: {elapsed}')

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

if __name__ == "__main__":
    main()

