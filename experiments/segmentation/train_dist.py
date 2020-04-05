##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Hang Zhang
## Email: zhanghang0704@gmail.com
## Copyright (c) 2020
##
## LICENSE file in the root directory of this source tree 
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import os
import time
import argparse
import numpy as np
from tqdm import tqdm

import torch
from torch.utils import data
import torch.multiprocessing as mp
import torchvision.transforms as transform
from torch.nn.parallel.scatter_gather import gather
from torch.nn.parallel import DistributedDataParallel

import encoding.utils as utils
from encoding.nn import SegmentationLosses, DistSyncBatchNorm

from encoding.datasets import get_dataset
from encoding.models import get_segmentation_model


class Options():
    def __init__(self):
        parser = argparse.ArgumentParser(description='PyTorch \
            Segmentation')
        # model and dataset 
        parser.add_argument('--model', type=str, default='encnet',
                            help='model name (default: encnet)')
        parser.add_argument('--backbone', type=str, default='resnet50',
                            help='backbone name (default: resnet50)')
        parser.add_argument('--dataset', type=str, default='ade20k',
                            help='dataset name (default: pascal12)')
        parser.add_argument('--workers', type=int, default=16,
                            metavar='N', help='dataloader threads')
        parser.add_argument('--base-size', type=int, default=520,
                            help='base image size')
        parser.add_argument('--crop-size', type=int, default=480,
                            help='crop image size')
        parser.add_argument('--train-split', type=str, default='train',
                            help='dataset train split (default: train)')
        # training hyper params
        parser.add_argument('--aux', action='store_true', default= False,
                            help='Auxilary Loss')
        parser.add_argument('--aux-weight', type=float, default=0.2,
                            help='Auxilary loss weight (default: 0.2)')
        parser.add_argument('--se-loss', action='store_true', default= False,
                            help='Semantic Encoding Loss SE-loss')
        parser.add_argument('--se-weight', type=float, default=0.2,
                            help='SE-loss weight (default: 0.2)')
        parser.add_argument('--epochs', type=int, default=None, metavar='N',
                            help='number of epochs to train (default: auto)')
        parser.add_argument('--start_epoch', type=int, default=0,
                            metavar='N', help='start epochs (default:0)')
        parser.add_argument('--batch-size', type=int, default=16,
                            metavar='N', help='input batch size for \
                            training (default: auto)')
        parser.add_argument('--test-batch-size', type=int, default=16,
                            metavar='N', help='input batch size for \
                            testing (default: same as batch size)')
        # optimizer params
        parser.add_argument('--lr', type=float, default=None, metavar='LR',
                            help='learning rate (default: auto)')
        parser.add_argument('--lr-scheduler', type=str, default='poly',
                            help='learning rate scheduler (default: poly)')
        parser.add_argument('--momentum', type=float, default=0.9,
                            metavar='M', help='momentum (default: 0.9)')
        parser.add_argument('--weight-decay', type=float, default=1e-4,
                            metavar='M', help='w-decay (default: 1e-4)')
        # cuda, seed and logging
        parser.add_argument('--seed', type=int, default=1, metavar='S',
                            help='random seed (default: 1)')
        # checking point
        parser.add_argument('--resume', type=str, default=None,
                            help='put the path to resuming file if needed')
        parser.add_argument('--checkname', type=str, default='default',
                            help='set the checkpoint name')
        parser.add_argument('--model-zoo', type=str, default=None,
                            help='evaluating on model zoo model')
        # finetuning pre-trained models
        parser.add_argument('--ft', action='store_true', default= False,
                            help='finetuning on a different dataset')
        # evaluation option
        parser.add_argument('--eval', action='store_true', default= False,
                            help='evaluating mIoU')
        parser.add_argument('--test-val', action='store_true', default= False,
                            help='generate masks on val set')
        # test option
        parser.add_argument('--test-folder', type=str, default=None,
                            help='path to test image folder')
        # the parser
        self.parser = parser

    def parse(self):
        args = self.parser.parse_args()
        # default settings for epochs, batch_size and lr
        if args.epochs is None:
            epoches = {
                'coco': 30,
                'pascal_aug': 80,
                'pascal_voc': 50,
                'pcontext': 80,
                'ade20k': 180,
                'citys': 240,
            }
            args.epochs = epoches[args.dataset.lower()]
        if args.lr is None:
            lrs = {
                'coco': 0.004,
                'pascal_aug': 0.001,
                'pascal_voc': 0.0001,
                'pcontext': 0.001,
                'ade20k': 0.004,
                'citys': 0.004,
            }
            args.lr = lrs[args.dataset.lower()] / 16 * args.batch_size
        print(args)
        return args

def main():
    args = Options().parse()
    ngpus_per_node = torch.cuda.device_count()
    args.world_size = ngpus_per_node * args.world_size
    args.lr = args.lr * args.world_size
    mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))

def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu
    args.rank = args.rank * ngpus_per_node + gpu
    print('rank: {} / {}'.format(args.rank, args.world_size))
    dist.init_process_group(backend=args.dist_backend,
                            init_method=args.dist_url,
                            world_size=args.world_size,
                            rank=args.rank)
    torch.cuda.set_device(args.gpu)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    # data transforms
    input_transform = transform.Compose([
        transform.ToTensor(),
        transform.Normalize([.485, .456, .406], [.229, .224, .225])])
    # dataset
    data_kwargs = {'transform': input_transform, 'base_size': args.base_size,
                   'crop_size': args.crop_size}
    trainset = get_dataset(args.dataset, split=args.train_split, mode='train', **data_kwargs)
    testset = get_dataset(args.dataset, split='val', mode ='val', **data_kwargs)
    train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
    val_sampler = torch.utils.data.distributed.DistributedSampler(valset, shuffle=False)
    # dataloader
    kwargs = {'batch_size': args.batch_size, 'num_workers': args.workers, 'pin_memory': True}
    trainloader = data.DataLoader(trainset, sampler=train_sampler, **kwargs)
    valloader = data.DataLoader(testset, sampler=val_sampler, **kwargs)
    nclass = trainset.num_class
    # model
    model = get_segmentation_model(args.model, dataset=args.dataset,
                                   backbone = args.backbone, aux = args.aux,
                                   se_loss = args.se_loss, norm_layer=DistSyncBatchNorm,
                                   base_size=args.base_size, crop_size=args.crop_size)
    if args.gpu == 0:
        print(model)
    # optimizer using different LR
    params_list = [{'params': model.pretrained.parameters(), 'lr': args.lr},]
    if hasattr(model, 'head'):
        params_list.append({'params': model.head.parameters(), 'lr': args.lr*10})
    if hasattr(model, 'auxlayer'):
        params_list.append({'params': model.auxlayer.parameters(), 'lr': args.lr*10})
    optimizer = torch.optim.SGD(params_list,
                                lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    # criterions
    criterion = SegmentationLosses(se_loss=args.se_loss,
                                   aux=args.aux,
                                   nclass=nclass, 
                                   se_weight=args.se_weight,
                                   aux_weight=args.aux_weight)
    # distributed data parallel
    model.cuda(args.gpu)
    criterion.cuda(args.gpu)
    model = DistributedDataParallel(model, device_ids=[args.gpu])

    # resuming checkpoint
    if args.resume is not None:
        if not os.path.isfile(args.resume):
            raise RuntimeError("=> no checkpoint found at '{}'" .format(args.resume))
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        model.module.load_state_dict(checkpoint['state_dict'])
        if not args.ft:
            optimizer.load_state_dict(checkpoint['optimizer'])
        best_pred = checkpoint['best_pred']
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(args.resume, checkpoint['epoch']))
    # clear start epoch if fine-tuning
    if args.ft:
        args.start_epoch = 0

    # lr scheduler
    scheduler = utils.LR_Scheduler_Head(args.lr_scheduler, args.lr,
                                        args.epochs, len(trainloader))
    best_pred = 0.0

    def training(epoch):
        train_loss = 0.0
        model.train()
        for i, (image, target) in enumerate(trainloader):
            scheduler(optimizer, i, epoch, best_pred)
            optimizer.zero_grad()
            outputs = model(image)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            if i % 50 == 0 and args.gpu == 0:
                print('Train loss: %.3f' % (train_loss / (i + 1)))

    def validation(epoch):
        # Fast test during the training
        def eval_batch(model, image, target):
            outputs = model(image)
            outputs = gather(outputs, 0, dim=0)
            pred = outputs[0]
            target = target.cuda(args.gpu)
            correct, labeled = utils.batch_pix_accuracy(pred.data, target)
            inter, union = utils.batch_intersection_union(pred.data, target, nclass)
            return correct, labeled, inter, union

        is_best = False
        model.eval()
        total_inter, total_union, total_correct, total_label = 0, 0, 0, 0
        for i, (image, target) in enumerate(valloader):
            with torch.no_grad():
                correct, labeled, inter, union = eval_batch(model, image, target)

            total_correct += correct
            total_label += labeled
            total_inter += inter
            total_union += union
            pixAcc = 1.0 * total_correct / (np.spacing(1) + total_label)
            IoU = 1.0 * total_inter / (np.spacing(1) + total_union)
            mIoU = IoU.mean()
            if i % 50 == 0 and args.gpu == 0:
                print('pixAcc: %.3f, mIoU: %.3f' % (pixAcc, mIoU))

        new_pred = (pixAcc + mIoU)/2
        if new_pred > best_pred:
            is_best = True
            best_pred = new_pred
        utils.save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.module.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_pred': best_pred,
        }, args, is_best)

    if args.gpu == 0:
        print('Starting Epoch:', trainer.args.start_epoch)
        print('Total Epoches:', trainer.args.epochs)

    for epoch in range(trainer.args.start_epoch, trainer.args.epochs):
        tic = time.time()
        training(epoch)
        if epoch % 10 == 0:
            validation(epoch)
        elapsed = time.time() - tic
        if args.gpu == 0:
            print(f'Epoch: {epoch}, Time cost: {elapsed}')

    validation(epoch)

if __name__ == "__main__":
    main()
