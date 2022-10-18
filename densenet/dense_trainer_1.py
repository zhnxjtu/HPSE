import argparse
import os
import shutil
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import math

import densenet as dn

# used for logging to TensorBoard
# from tensorboard_logger import configure, log_value

parser = argparse.ArgumentParser(description='PyTorch DenseNet Training')
parser.add_argument('--epochs', default=100, type=int,
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    help='mini-batch size (default: 64)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    help='print frequency (default: 10)')
parser.add_argument('--layers', default=40, type=int,
                    help='total number of layers (default: 100)')
parser.add_argument('--growth', default=12, type=int,
                    help='number of new channels per layer (default: 12)')
parser.add_argument('--droprate', default=0, type=float,
                    help='dropout probability (default: 0.0)')
parser.add_argument('--no-augment', dest='augment', action='store_false',
                    help='whether to use standard augmentation (default: True)')
parser.add_argument('--reduce', default=1, type=float,
                    help='compression rate in transition stage (default: 0.5)')
parser.add_argument('--no-bottleneck', dest='bottleneck', action='store_false',
                    help='To not use bottleneck block')
parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--name', default='DenseNet_BC_40_12', type=str,
                    help='name of experiment')
parser.add_argument('--tensorboard',
                    help='Log progress to TensorBoard', action='store_true')
parser.add_argument('--save-every', dest='save_every',
                    help='Saves checkpoints at every specified number of epochs', type=int, default=10)
parser.set_defaults(bottleneck=True)
parser.set_defaults(augment=True)

use_gpu = torch.cuda.is_available()
device = torch.device("cuda:0" if use_gpu else "cpu")

best_prec1 = 0


def main():
    global args, best_prec1
    args = parser.parse_args()
    # if args.tensorboard: configure("runs/%s"%(args.name))

    # Data loading code
    normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x / 255.0 for x in [63.0, 62.1, 66.7]])

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

    kwargs = {'num_workers': 1, 'pin_memory': True}
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./data', train=True, download=True,
                         transform=transform_train), batch_size=args.batch_size, shuffle=True, **kwargs)
    val_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./data', train=False,
                         transform=transform_test), batch_size=args.batch_size, shuffle=True, **kwargs)

    # create model
    args.layers = 40
    args.growth = 12
    args.reduce = 1.0
    args.bottleneck = False
    model = dn.DenseNet3(args.layers, 10, args.growth, reduction=args.reduce,
                         bottleneck=args.bottleneck, dropRate=args.droprate)

    # get the number of model parameters
    print('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))

    # for training on multiple GPUs.
    # Use CUDA_VISIBLE_DEVICES=0,1 to specify which GPUs to use
    # model = torch.nn.DataParallel(model).cuda()
    model = model.cuda()

    #############################  open when prune the first basic block ###################
    pretrained_dict = torch.load('../2.Pre-trained_Model/densenet_ori.pth')
    model_dict = model.state_dict()
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    # ############################  open when prune the second basic block ###################
    # checkpoint = torch.load('./model-12/model.pth')
    # model.load_state_dict(checkpoint['state_dict'])

    criterion = nn.CrossEntropyLoss().cuda()

    print('The accuracy of the original DNN: \n')
    ac_ori = validate(val_loader, model, criterion)

    ##########################################################################################
    num_prune = 12                                                 # change 24 when prune the second block

    layer_prune = [7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51,
                   61, 65, 69, 73, 77, 81, 85, 89, 93, 97, 101, 105,
                   115, 119, 123, 127, 131, 135, 139, 143, 147, 151, 155, 159]
    indices_pruned = {}
    for i in range(num_prune):
        # load pruned channel
        indices_pruned['pruned_filter' + str(i)] = load_pruning_file(int(i + 1))
        # pruning
        pruning(model, layer_prune[int(i)], indices_pruned['pruned_filter' + str(i)])


    print('The accuracy of the pruned DNN without fine-tuning: \n')
    validate(val_loader, model, criterion)
    ##########################################################################################

    cudnn.benchmark = True

    # define loss function (criterion) and pptimizer

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                nesterov=True,
                                weight_decay=args.weight_decay)

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                        milestones=[30, 80],
                                                        last_epoch=args.start_epoch - 1)

    ############################################################################################################################
    file_name = open('./dense_acc-' + str(num_prune) + '.txt', mode='w')
    if not os.path.exists('./model-' + str(num_prune)):
        os.mkdir('./model-' + str(num_prune))
    args.save_dir = './model-' +str(num_prune)

    for epoch in range(args.start_epoch, args.epochs):
        # adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)
        lr_scheduler.step()

        ################################################################
        for i in range(num_prune):
            pruning(model, layer_prune[int(i)], indices_pruned['pruned_filter' + str(i)])
        ################################################################

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion)

        file_name.write('{} \n'.format(prec1))

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1_tmp = best_prec1
        best_prec1 = max(prec1, best_prec1)

        if epoch >0 and epoch % args.save_every == 0:
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


def train(train_loader, model, criterion, optimizer, epoch):
    """Train for one epoch on the training set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
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
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                loss=losses, top1=top1))
    # log to TensorBoard
    # if args.tensorboard:
    #     log_value('train_loss', losses.avg, epoch)
    #     log_value('train_acc', top1.avg, epoch)


def validate(val_loader, model, criterion):
    """Perform validation on the validation set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda()
        input = input.cuda()
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target, topk=(1,))[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                i, len(val_loader), batch_time=batch_time, loss=losses,
                top1=top1))

    print(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))
    # log to TensorBoard
    # if args.tensorboard:
    #     log_value('val_loss', losses.avg, epoch)
    #     log_value('val_acc', top1.avg, epoch)
    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """Saves checkpoint to disk"""
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

def load_pruning_file(layer_num):
    f = open('./pruned_label/FM' + str(layer_num) + '.txt', 'r')
    # f = open('./4.Pruned_Label/pruned_label_dense_0.40/FM' + str(layer_num) + '.txt', 'r')
    pruned_filter_str = f.readlines()
    pruned_filter_str = [x.strip() for x in pruned_filter_str if x.strip() != '']
    f.close()
    pruned_filter = []
    for n in pruned_filter_str:
        pruned_filter.append(int(n) - 1)
    return pruned_filter


if __name__ == '__main__':
    main()
