import torch
import torchvision
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.models as models
import math
import numpy as np
import shutil
import argparse
import torch.backends.cudnn as cudnn
import time
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3,4,5,6,7"

use_gpu = torch.cuda.is_available()
device = torch.device("cuda:0" if use_gpu else "cpu")

parser = argparse.ArgumentParser(description='Propert ResNets for ImageNet in pytorch')

parser.add_argument('-j', '--workers', default=40, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=150, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=50, type=int,
                    metavar='N', help='print frequency (default: 50)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--save-dir', dest='save_dir',
                    help='The directory used to save the trained models',
                    default='save_temp', type=str)
parser.add_argument('--save-every', dest='save_every',
                    help='Saves checkpoints at every specified number of epochs',
                    type=int, default=10)
best_prec1 = 0
best_prec5 = 0

def main():
    global args, best_prec1, best_prec5
    args = parser.parse_args()

    # Check the save_dir exists or not
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    model = models.resnet50(pretrained=False)
    model = torch.nn.DataParallel(model)
    model.to(device)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            best_prec5 = checkpoint['best_prec5']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.evaluate, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True


    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = torchvision.datasets.ImageFolder(root='/data/dataset/ILSVRC2012_pytorch/train',
                                                     transform=transforms.Compose([
                                                         transforms.RandomResizedCrop(224),
                                                         transforms.RandomHorizontalFlip(),
                                                         transforms.ToTensor(),
                                                         normalize]))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=512, shuffle=True, num_workers=32, pin_memory=True)

    val_dataset = torchvision.datasets.ImageFolder(root='/data/dataset/ILSVRC2012_pytorch/val',
                                                    transform=transforms.Compose([
                                                        transforms.Resize(256),
                                                        transforms.CenterCrop(224),
                                                        transforms.ToTensor(),
                                                        normalize]))
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                              batch_size=512, shuffle=True, num_workers=32, pin_memory=True)

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)

    # pruning
    num_prune = 16


    layer_index = [8, 10, 12, 19, 21, 23, 27, 29, 31,
                   36, 38, 40, 47, 49, 51, 55, 57, 59, 63, 65, 67,
                   72, 74, 76, 83, 85, 87, 91, 93, 95, 99, 101, 103, 107, 109, 111, 115, 117, 119,
                   124, 126, 128, 135, 137, 139, 143, 145, 147]
    indices_pruned = {}
    kk = 0
    for i in range(num_prune):
        indices_pruned['pruned_filter' + str(3*i)] = load_pruning_file(int((3*i)+1))
        indices_pruned['pruned_filter' + str(3*i+1)] = load_pruning_file(int((3*i)+2))
        indices_pruned['pruned_filter' + str(3*i+2)] = load_pruning_file(int((3*i)+3))

        kk = kk + 1
        pruning(model, layer_index[int(3*i)], indices_pruned['pruned_filter' + str(3*i)], kk)
        pruning(model, layer_index[int(3*i+1)], indices_pruned['pruned_filter' + str(3*i+1)], kk)
        pruning(model, layer_index[int(3*i+2)], indices_pruned['pruned_filter' + str(3*i+2)], kk)



    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80, 120], gamma=0.1)

    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    file_name = open('./res50_acc-' +str(num_prune)+ '.txt', mode='w')
    if not os.path.exists('./model-' +str(num_prune)):
        os.mkdir('./model-' +str(num_prune))
    args.save_dir = './model-' +str(num_prune)

    # file_name = open('./res50_acc-' + str(num_prune) + '(2).txt', mode='w')
    # if not os.path.exists('./model-' + str(num_prune) + '(2)'):
    #     os.mkdir('./model-' + str(num_prune) + '(2)')

    for epoch in range(args.start_epoch, args.epochs):
        print('epoch {}'.format(epoch + 1))
        print('*' * 10)

        # train for one epoch
        print('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
        train(train_loader, model, criterion, optimizer, epoch)
        lr_scheduler.step()


        ################################################################
        #################################################################
        kk = 0 # count times
        for i in range(num_prune):
            kk = kk + 1
            pruning(model, layer_index[int(3*i)], indices_pruned['pruned_filter' + str(3*i)], kk)
            pruning(model, layer_index[int(3*i+1)], indices_pruned['pruned_filter' + str(3*i+1)], kk)
            pruning(model, layer_index[int(3*i+2)], indices_pruned['pruned_filter' + str(3*i+2)], kk)
        ################################################################

        # evaluate on validation set
        prec1, prec5 = validate(val_loader, model, criterion)

        # torch.save(model.state_dict(), './model-' +str(num_prune)+ '/res50_'+ str(epoch + 1) +'.pth', _use_new_zipfile_serialization=False)
        file_name.write('{} {}\n'.format(prec1, prec5))

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1_tmp = best_prec1
        best_prec1 = max(prec1, best_prec1)

        best_prec5 = max(prec5, best_prec5)

        if epoch > 0 and epoch % args.save_every == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'best_prec5': best_prec5,
            }, is_best, filename=os.path.join(args.save_dir, 'checkpoint.th'))

        if best_prec1 > best_prec1_tmp:
            save_checkpoint({
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'best_prec5': best_prec5,
            }, is_best, filename=os.path.join(args.save_dir, 'model.th'))

    print('Best Top-1 accuracy: {:.2f}, Top-5 accuracy: {:.2f} '.format(best_prec1, best_prec5))
    # print('ori_Top1-curTop1: ', (prec1_ori-best_prec1))


def train(train_loader, model, criterion, optimizer, epoch):
    """
        Run one train epoch
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda()
        input_var = input.cuda()
        target_var = target
        # if args.half:
        #     input_var = input_var.half()

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output = output.float()
        loss = loss.float()
        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1,5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@1 {top5.val:.3f} ({top5.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses, top1=top1, top5=top5))

def validate(val_loader, model, criterion):
    """
    Run evaluation
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda()
            input_var = input.cuda()
            target_var = target.cuda()

            # if args.half:
            #     input_var = input_var.half()

            # compute output
            output = model(input_var)
            loss = criterion(output, target_var)

            output = output.float()
            loss = loss.float()

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1,5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.4f} ({top1.avg:.4f})\t'
                      'Prec@5 {top5.val:.4f} ({top5.avg:.4f})\t'.format(
                          i, len(val_loader), batch_time=batch_time, loss=losses,
                          top1=top1, top5=top5))

    print(' * Prec@1 {top1.avg:.2f}\t'
          ' * Prec@5 {top5.avg:.2f}\t'
          .format(top1=top1, top5=top5))

    return top1.avg, top5.avg

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
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def pruning(model, layer, pruned_filter, kk):
    for k, m in enumerate(model.modules()):
        if (k == layer) & isinstance(m, nn.Conv2d):
            # print('Pruning the {}-th layer ....'.format(kk))
            # print('pruning: ......')
            weight_copy = m.weight.data.clone()
            mask = torch.ones((weight_copy.size()[0], weight_copy.size()[1], weight_copy.size()[2], weight_copy.size()[3])).float().to(device)
            for pruned_index in pruned_filter:
                mask[pruned_index] = torch.zeros((weight_copy.size()[1], weight_copy.size()[2], weight_copy.size()[3])).abs().float().to(device)
            m.weight.data.mul_(mask)

# channel pruning
# def pruning(model, layer, pruned_channel, kk):
#     for k, m in enumerate(model.modules()):
#         if (k == layer) & isinstance(m, nn.Conv2d):
#             print('Pruning the {}-th layer:'.format(kk))
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

def load_pruning_file(layer_num):
    f = open('../resnet50/pruned_label/FM' + str(layer_num) + '.txt', 'r')
    pruned_filter_str = f.readlines()
    pruned_filter_str = [x.strip() for x in pruned_filter_str if x.strip() != '']
    f.close()
    pruned_filter = []
    for n in pruned_filter_str:
        pruned_filter.append(int(n) - 1)
    return pruned_filter

if __name__ == '__main__':
    main()