from PIL import Image
import matplotlib.pyplot as plt
import torch
import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import imageio
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import argparse
import densenet as dn

# used for logging to TensorBoard
# from tensorboard_logger import configure, log_value

parser = argparse.ArgumentParser(description='PyTorch DenseNet Training')
parser.add_argument('--layers', default=40, type=int,
                    help='total number of layers (default: 100)')
parser.add_argument('--growth', default=12, type=int,
                    help='number of new channels per layer (default: 12)')
parser.add_argument('--droprate', default=0, type=float,
                    help='dropout probability (default: 0.0)')
parser.add_argument('--no-augment', dest='augment', action='store_false',
                    help='whether to use standard augmentation (default: True)')
parser.add_argument('--reduce', default=1.0, type=float,
                    help='compression rate in transition stage (default: 0.5)')
parser.add_argument('--no-bottleneck', dest='bottleneck', action='store_false',
                    help='To not use bottleneck block')
parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--name', default='DenseNet_40_12', type=str,
                    help='name of experiment')
parser.add_argument('--tensorboard',
                    help='Log progress to TensorBoard', action='store_true')
parser.set_defaults(bottleneck=True)
parser.set_defaults(augment=True)
args = parser.parse_args()

def get_image_info(image_dir):
    image_info = Image.open(image_dir).convert('RGB')

    normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x / 255.0 for x in [63.0, 62.1, 66.7]])

    transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
            ])

    image_info = transform(image_info)
    image_info = image_info.unsqueeze(0)
    return  image_info


if __name__ == '__main__':
    for num_fig in range(1,501):
        image_dir = '../1.Image/Image_CIFAR10/' + str(num_fig) + '.png'

        args.layers = 40
        args.growth = 12
        args.reduce = 1.0
        args.bottleneck = False


        k = 0
        for layer_num in range(13,25):

            index_dic = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

            if layer_num == 13:

                model = dn.DenseNet3(args.layers, 10, args.growth, reduction=args.reduce,
                                     bottleneck=args.bottleneck, dropRate=args.droprate)

                checkpoint = torch.load('./model-12/model.th')
                model.load_state_dict(checkpoint['state_dict'])

                use_gpu = torch.cuda.is_available()
                device = torch.device("cuda:0" if use_gpu else "cpu")
                model = nn.DataParallel(model)
                model.to(device)
                model.eval()

                image_info = get_image_info(image_dir)
                image_info = image_info.to(device)
                x = image_info

                #########################################################
                with torch.no_grad():
                    for index, layer in model.module._modules.items():
                        x = layer(x)
                        if index == 'trans1':
                            break

                    for index, layer in model.module.block2.layer[0]._modules.items():
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
                        np.savetxt(
                            './FM/FM-' + str(layer_num) + '/' + str(num_fig) + '/conv' + str(layer_num) + '_' + str(
                                b_num + 1) + '.csv', feature_map[0, b_num, :, :].cpu().numpy(), delimiter=',')

            if layer_num > 13:

                model = dn.DenseNet3(args.layers, 10, args.growth, reduction=args.reduce,
                                     bottleneck=args.bottleneck, dropRate=args.droprate)

                checkpoint = torch.load('./model-12/model.th')
                model.load_state_dict(checkpoint['state_dict'])

                use_gpu = torch.cuda.is_available()
                device = torch.device("cuda:0" if use_gpu else "cpu")
                model = nn.DataParallel(model)
                model.to(device)
                model.eval()

                image_info = get_image_info(image_dir)
                image_info = image_info.to(device)
                x = image_info

                #########################################################
                with torch.no_grad():
                    for index, layer in model.module._modules.items():
                        x = layer(x)
                        if index == 'trans1':
                            print('Model path (image_num({}), layer_num({})): {}\t'.format(num_fig, layer_num, index), end='')
                            break

                    for index, layer in model.module.block2.layer._modules.items():
                        x = layer(x)
                        if index == str(index_dic[k]):
                            print('--- {}\t'.format(index), end='')
                            break

                    for index, layer in model.module.block2.layer[index_dic[k] + 1]._modules.items():
                        x = layer(x)
                        if index == 'conv1':
                            print('--- {}\t'.format(index_dic[k] + 1), end='')
                            print('--- {}\t'.format(index))
                            break

                    k = k + 1

                    feature_map = x

                    if not os.path.exists('./FM/FM-' + str(layer_num)):
                        os.mkdir('./FM/FM-' + str(layer_num))

                    if not os.path.exists('./FM/FM-' + str(layer_num) + '/' + str(num_fig)):
                        os.mkdir('./FM/FM-' + str(layer_num) + '/' + str(num_fig))

                    for b_num in range(feature_map.shape[1]):
                        np.savetxt(
                            './FM/FM-' + str(layer_num) + '/' + str(num_fig) + '/conv' + str(layer_num) + '_' + str(
                                b_num + 1) + '.csv', feature_map[0, b_num, :, :].cpu().numpy(), delimiter=',')

        #
        # layer_num = 24
        # pretrained_dict = torch.load('./model-12/dense_85.pth')
        #
        # model_dict = model.state_dict()
        # model_dict.update(pretrained_dict)
        # model.load_state_dict(model_dict)
        #
        # use_gpu = torch.cuda.is_available()
        # device = torch.device("cuda:0" if use_gpu else "cpu")
        # model = nn.DataParallel(model)
        # model.to(device)
        # model.eval()
        #
        # image_info = get_image_info(image_dir)
        # image_info = image_info.to(device)
        # x = image_info
        #
        # #########################################################
        # with torch.no_grad():
        #     for index, layer in model.module._modules.items():
        #         x = layer(x)
        #         if index == 'trans1':
        #             break
        #
        #     for index, layer in model.module.block2.layer._modules.items():
        #         x = layer(x)
        #         if index == '10':
        #             break
        #
        #     for index, layer in model.module.block2.layer[11]._modules.items():
        #         x = layer(x)
        #         if index == 'conv1':
        #             print(index)
        #             break
        #
        #     feature_map = x
        #
        #     if not os.path.exists('./FM/FM-' + str(layer_num)):
        #         os.mkdir('./FM/FM-' + str(layer_num))
        #
        #     if not os.path.exists('./FM/FM-' + str(layer_num) + '/' + str(num_fig)):
        #         os.mkdir('./FM/FM-' + str(layer_num) + '/' + str(num_fig))
        #
        #     for b_num in range(feature_map.shape[1]):
        #         np.savetxt('./FM/FM-' + str(layer_num) + '/' + str(num_fig) + '/conv' + str(layer_num) + '_' + str(
        #             b_num + 1) + '.csv', feature_map[0, b_num, :, :].cpu().numpy(), delimiter=',')