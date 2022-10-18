import argparse
import os
import shutil
import time
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.optim
import math
import numpy as np
import random
from numpy.linalg import norm, svd

import VGG16 as vgg

# used for logging to TensorBoard
# from tensorboard_logger import configure, log_value

parser = argparse.ArgumentParser(description='PyTorch DenseNet Training')
parser.add_argument('--epochs', default=30, type=int,
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    help='mini-batch size (default: 64)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    help='initial learning rate')
parser.add_argument('--lr_decay_step', default='10, 20', type=str,
                    help='learning rate decay step')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    help='print frequency (default: 10)')
parser.add_argument('--no-augment', dest='augment', action='store_false',
                    help='whether to use standard augmentation (default: True)')
parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--name', default='VGG_16', type=str,
                    help='name of experiment')
parser.add_argument('--tensorboard',
                    help='Log progress to TensorBoard', action='store_true')
parser.add_argument('--save-every', dest='save_every',
                    help='Saves checkpoints at every specified number of epochs',
                    type=int, default=10)
parser.set_defaults(augment=True)


use_gpu = torch.cuda.is_available()
device = torch.device("cuda:0" if use_gpu else "cpu")

best_prec1 = 0

def main():
    global args, best_prec1
    args = parser.parse_args()

    # Data loading code
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    
    if args.augment:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
            ])
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            normalize,
            ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize
        ])

    kwargs = {'num_workers': 8, 'pin_memory': True}
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./data', train=True, download=True,
                         transform=transform_train),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    val_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./data', train=False, transform=transform_test),
        batch_size=args.batch_size, shuffle=True, **kwargs)

    # create model
    model = vgg.VGG()

    # get the number of model parameters
    print('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))
    
    # for training on multiple GPUs. 
    # Use CUDA_VISIBLE_DEVICES=0,1 to specify which GPUs to use
    # model = torch.nn.DataParallel(model).cuda()
    model = model.cuda()

    # load pretrained model
    checkpoint = torch.load('./model-12/model.th')
    model.load_state_dict(checkpoint['state_dict'])

    criterion = nn.CrossEntropyLoss().cuda()
    print('The accuracy of the original DNN: \n')
    ac_ori = validate(val_loader, model, criterion)

    layer_prune = [2, 5, 9, 12, 16, 19, 22, 26, 29, 32, 36, 39, 42]

    num_prune = 13

    indices_pruned = {}
    # for i in range(len(layer_prune)):
##########################################################################################
    for i in range(num_prune):
        # load pruned channel
        indices_pruned['pruned_filter' + str(i)] = load_pruning_file(int(i+1))
        # pruning
        pruning(model, layer_prune[i], indices_pruned['pruned_filter' + str(i)])
 ##########################################################################################

    print('The accuracy of the pruned DNN without fine-tuning: \n')
    prec1 = validate(val_loader, model, criterion)
    is_best = prec1 > best_prec1
    best_prec1 = max(prec1, best_prec1)

    file_name = open('./VGG_acc-' + str(num_prune) + '.txt', mode='w')
    if not os.path.exists('./model-' + str(num_prune)):
        os.mkdir('./model-' + str(num_prune))
    args.save_dir = './model-' + str(num_prune)

    save_checkpoint({
        'state_dict': model.state_dict(),
        'best_prec1': best_prec1,
    }, is_best, filename=os.path.join(args.save_dir, 'model.th'))

    # # optionally resume from a checkpoint
    # if args.resume:
    #     if os.path.isfile(args.resume):
    #         print("=> loading checkpoint '{}'".format(args.resume))
    #         checkpoint = torch.load(args.resume)
    #         args.start_epoch = checkpoint['epoch']
    #         best_prec1 = checkpoint['best_prec1']
    #         model.load_state_dict(checkpoint['state_dict'])
    #         print("=> loaded checkpoint '{}' (epoch {})"
    #               .format(args.resume, checkpoint['epoch']))
    #     else:
    #         print("=> no checkpoint found at '{}'".format(args.resume))
    #
    cudnn.benchmark = True

    # define loss function (criterion) and pptimizer
    # criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                nesterov=True,
                                weight_decay=args.weight_decay)

    lr_decay_step = list(map(int, args.lr_decay_step.split(',')))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_decay_step, gamma=0.1)


    for epoch in range(args.start_epoch, args.epochs):
        # adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        print('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
        train(train_loader, model, criterion, optimizer, epoch, layer_prune, indices_pruned)
        scheduler.step()

##########################################################################################
        for i in range(num_prune):
            pruning(model, layer_prune[i], indices_pruned['pruned_filter' + str(i)])
##########################################################################################
        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion)

        # torch.save(model.state_dict(), './model-'+str(num_prune)+'/VGG_' + str(epoch + 1) + '.pth')
        file_name.write('{} \n'.format(prec1))

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1_tmp = best_prec1
        best_prec1 = max(prec1, best_prec1)

        if epoch > 0 and epoch % args.save_every == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
            }, is_best, filename=os.path.join(args.save_dir, 'checkpoint.th'))

        if best_prec1 > best_prec1_tmp:
            save_checkpoint({
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
            }, is_best, filename=os.path.join(args.save_dir, 'model.th'))

    print('Best accuracy: ', best_prec1)
    print('ori_Top1 - curr_Top1: ', (ac_ori - best_prec1))

# filter-wise pruning
def pruning(model, layer, pruned_filter):
    for k, m in enumerate(model.modules()):
        if (k == layer) & isinstance(m, nn.Conv2d):
            # print('Pruning the {}-th layer'.format(k))
            weight_copy = m.weight.data.clone()
            mask = torch.ones((weight_copy.size()[0], weight_copy.size()[1], weight_copy.size()[2],
                               weight_copy.size()[3])).float().to(device)
            for pruned_index in pruned_filter:
                mask[pruned_index] = torch.zeros(
                    (weight_copy.size()[1], weight_copy.size()[2], weight_copy.size()[3])).abs().float().to(device)
            m.weight.data.mul_(mask)

# channel-wise pruning
# def pruning(model, layer, pruned_channel):
#     for k, m in enumerate(model.modules()):
#         if (k == layer) & isinstance(m, nn.Conv2d):
#             weight_copy = m.weight.data.clone()
#             mask = torch.ones((weight_copy.size()[0], weight_copy.size()[1], weight_copy.size()[2],
#                                weight_copy.size()[3])).float().to(device)
#
#             for pruned_index in pruned_channel:
#
#                 pruned_in = (pruned_index + 1) % weight_copy.size()[1]
#                 pruned_out = math.floor((pruned_index + 1) / weight_copy.size()[1])
#                 if pruned_in == 0:
#                     pruned_in = weight_copy.size()[1] - 1
#                     pruned_out = pruned_out -1
#                 else:
#                     pruned_in = pruned_in - 1
#                     pruned_out = pruned_out + 1 -1
#
#                 mask[pruned_out][pruned_in] = torch.zeros(
#                     (weight_copy.size()[2], weight_copy.size()[3])).abs().float().to(device)
#             m.weight.data.mul_(mask)

def train(train_loader, model, criterion, optimizer, epoch, layer_prune, indices_pruned):
    """Train for one epoch on the training set"""
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda()
        input = input.cuda()
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target, topk=(1,))[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1))

def validate(val_loader, model, criterion):
    """Perform validation on the validation set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    # top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda()
        input = input.cuda()
        with torch.no_grad():
            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        # top5.update(prec5.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'.format(
                      i, len(val_loader), batch_time=batch_time, loss=losses,
                      top1=top1))

    print(' * Prec@1 \033[31m {top1.avg:.3f} \033[0m \t'.format(top1=top1))
    return top1.avg

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """
        Save the training model
        """
    torch.save(state, filename)

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

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def load_pruning_file(layer_num):
    f = open('./pruned_label/FM' + str(layer_num) + '.txt', 'r')
    pruned_filter_str = f.readlines()
    pruned_filter_str = [x.strip() for x in pruned_filter_str if x.strip() != '']
    f.close()
    pruned_filter = []
    for n in pruned_filter_str:
        pruned_filter.append(int(n) - 1)
    return pruned_filter

if __name__ == '__main__':
    main()