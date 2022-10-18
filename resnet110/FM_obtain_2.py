from PIL import Image
import matplotlib.pyplot as plt
import torch
import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import imageio
import os
import math

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import math
import resnet
import argparse

model_names = sorted(name for name in resnet.__dict__
                     if name.islower() and not name.startswith("__")
                     and name.startswith("resnet")
                     and callable(resnet.__dict__[name]))

print(model_names)

parser = argparse.ArgumentParser(description='Propert ResNets for CIFAR10 in pytorch')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet110',
                    choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) +
                    ' (default: resnet32)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--half', dest='half', action='store_true',
                    help='use half-precision(16-bit) ')
parser.add_argument('--save-dir', dest='save_dir',
                    help='The directory used to save the trained models',
                    default='save_temp', type=str)
parser.add_argument('--save-every', dest='save_every',
                    help='Saves checkpoints at every specified number of epochs',
                    type=int, default=10)


def get_image_info(image_dir):
    image_info = Image.open(image_dir).convert('RGB')

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    image_info = transform(image_info)
    image_info = image_info.unsqueeze(0)
    return image_info


if __name__ == '__main__':
    for num_fig in range(1, 501):
        image_dir = '../1.Image/Image_CIFAR10/' + str(num_fig) + '.png'

        global args
        args = parser.parse_args()
        model = resnet.__dict__[args.arch]()

        use_gpu = torch.cuda.is_available()
        device = torch.device("cuda:0" if use_gpu else "cpu")
        model = nn.DataParallel(model)
        model.to(device)

        k = 0
        index_dic = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

        for layer_num in range(37, 73):

            if layer_num == 37:

                checkpoint = torch.load('./model-18/model.th')
                model.load_state_dict(checkpoint['state_dict'])
                model.eval()

                image_info = get_image_info(image_dir)
                image_info = image_info.to(device)

                x = image_info

                #########################################################
                with torch.no_grad():
                    for index, layer in model.module._modules.items():
                        x = layer(x)
                        if index == 'layer1':
                            break

                    for index, layer in model.module.layer2[0]._modules.items():
                        x = layer(x)
                        if index == 'conv1':
                            print(index)
                            break

                feature_map = x

                if not os.path.exists('./FM'):
                    os.mkdir('./FM')

                if not os.path.exists('./FM/FM-' + str(layer_num)):
                    os.mkdir('./FM/FM-' + str(layer_num))

                if not os.path.exists('./FM/FM-' + str(layer_num) + '/' + str(num_fig)):
                    os.mkdir('./FM/FM-' + str(layer_num) + '/' + str(num_fig))

                for b_num in range(feature_map.shape[1]):
                    np.savetxt('./FM/FM-' + str(layer_num) + '/' + str(num_fig) + '/conv' + str(layer_num) + '_' + str(
                        b_num + 1) + '.csv', feature_map[0, b_num, :, :].cpu().numpy(), delimiter=',')

            if layer_num == 38:

                checkpoint = torch.load('./model-18/model.th')
                model.load_state_dict(checkpoint['state_dict'])
                model.eval()

                image_info = get_image_info(image_dir)
                image_info = image_info.to(device)

                x = image_info

                #########################################################
                with torch.no_grad():
                    for index, layer in model.module._modules.items():
                        x = layer(x)
                        if index == 'layer1':
                            break

                    for index, layer in model.module.layer2[0]._modules.items():
                        x = layer(x)
                        if index == 'conv2':
                            print(index)
                            break

                feature_map = x

                if not os.path.exists('./FM'):
                    os.mkdir('./FM')

                if not os.path.exists('./FM/FM-' + str(layer_num)):
                    os.mkdir('./FM/FM-' + str(layer_num))

                if not os.path.exists('./FM/FM-' + str(layer_num) + '/' + str(num_fig)):
                    os.mkdir('./FM/FM-' + str(layer_num) + '/' + str(num_fig))

                for b_num in range(feature_map.shape[1]):
                    np.savetxt('./FM/FM-' + str(layer_num) + '/' + str(num_fig) + '/conv' + str(layer_num) + '_' + str(
                        b_num + 1) + '.csv', feature_map[0, b_num, :, :].cpu().numpy(), delimiter=',')

            if layer_num > 38:

                if layer_num % 2 != 0:
                    index_conv = 'conv1'
                else:
                    index_conv = 'conv2'

                checkpoint = torch.load('./model-18/model.th')
                model.load_state_dict(checkpoint['state_dict'])
                model.eval()

                image_info = get_image_info(image_dir)
                image_info = image_info.to(device)

                x = image_info

                ########################################################
                with torch.no_grad():
                    for index, layer in model.module._modules.items():
                        x = layer(x)
                        if index == 'layer1':
                            print('Model path (image_num({}), layer_num({})): {}\t'.format(num_fig, layer_num, index),
                                  end='')
                            break

                    for index, layer in model.module.layer2._modules.items():
                        x = layer(x)
                        if index == str(index_dic[k]):
                            print('--- {}\t'.format(index), end='')
                            break

                    for index, layer in model.module.layer2[index_dic[k] + 1]._modules.items():
                        x = layer(x)
                        if index == index_conv:
                            print('--- {}\t'.format(index_dic[k] + 1), end='')
                            print('--- {}\t'.format(index))
                            break

                if index_conv == 'conv2':
                    k = k + 1

                feature_map = x

                if not os.path.exists('./FM'):
                    os.mkdir('./FM')

                if not os.path.exists('./FM/FM-' + str(layer_num)):
                    os.mkdir('./FM/FM-' + str(layer_num))

                if not os.path.exists('./FM/FM-' + str(layer_num) + '/' + str(num_fig)):
                    os.mkdir('./FM/FM-' + str(layer_num) + '/' + str(num_fig))

                for b_num in range(feature_map.shape[1]):
                    np.savetxt('./FM/FM-' + str(layer_num) + '/' + str(num_fig) + '/conv' + str(layer_num) + '_' + str(
                        b_num + 1) + '.csv', feature_map[0, b_num, :, :].cpu().numpy(), delimiter=',')