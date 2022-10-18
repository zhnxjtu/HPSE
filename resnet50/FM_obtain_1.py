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

def get_image_info(image_dir):
    image_info = Image.open(image_dir).convert('RGB')

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    transform = transforms.Compose([transforms.RandomResizedCrop(224),
                                    # transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    normalize])

    image_info = transform(image_info)
    image_info = image_info.unsqueeze(0)
    return  image_info


if __name__ == '__main__':
    for num_fig in range(1,201):
        image_dir = '../1.Image/Image_ImageNet/'+ str(num_fig) +'.png'

        k = 0
        index_dic = [0, 1]
        for layer_num in range(1, 10):

            if layer_num == 1:

                model = models.resnet50(pretrained=False)
                pretrained_dict = torch.load('./resnet50-19c8e357.pth')
                model_dict = model.state_dict()
                model_dict.update(pretrained_dict)
                model.load_state_dict(model_dict)

                use_gpu = torch.cuda.is_available()
                device = torch.device("cuda:0" if use_gpu else "cpu")
                model = torch.nn.DataParallel(model)
                model.to(device)
                model.eval()

                image_info = get_image_info(image_dir)
                image_info = image_info.to(device)
                x = image_info

                with torch.no_grad():
                    for index, layer in model.module._modules.items():
                        x = layer(x)
                        if index == 'maxpool':
                            break

                    for index, layer in model.module.layer1[0]._modules.items():
                        x = layer(x)
                        if index == 'conv1':
                            print('Now index is: ', index)
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

            if layer_num == 2:

                model = models.resnet50(pretrained=False)
                pretrained_dict = torch.load('./resnet50-19c8e357.pth')
                model_dict = model.state_dict()
                model_dict.update(pretrained_dict)
                model.load_state_dict(model_dict)

                use_gpu = torch.cuda.is_available()
                device = torch.device("cuda:0" if use_gpu else "cpu")
                model = torch.nn.DataParallel(model)
                model.to(device)
                model.eval()

                image_info = get_image_info(image_dir)
                image_info = image_info.to(device)
                x = image_info

                with torch.no_grad():
                    for index, layer in model.module._modules.items():
                        x = layer(x)
                        if index == 'maxpool':
                            break

                    for index, layer in model.module.layer1[0]._modules.items():
                        x = layer(x)
                        if index == 'conv2':
                            print('Now index is: ', index)
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

            if layer_num == 3:

                model = models.resnet50(pretrained=False)
                pretrained_dict = torch.load('./resnet50-19c8e357.pth')
                model_dict = model.state_dict()
                model_dict.update(pretrained_dict)
                model.load_state_dict(model_dict)

                use_gpu = torch.cuda.is_available()
                device = torch.device("cuda:0" if use_gpu else "cpu")
                model = torch.nn.DataParallel(model)
                model.to(device)
                model.eval()

                image_info = get_image_info(image_dir)
                image_info = image_info.to(device)
                x = image_info

                with torch.no_grad():
                    for index, layer in model.module._modules.items():
                        x = layer(x)
                        if index == 'maxpool':
                            break

                    for index, layer in model.module.layer1[0]._modules.items():
                        x = layer(x)
                        if index == 'conv3':
                            print('Now index is: ', index)
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

            if layer_num > 3:

                if layer_num % 3 == 1:
                    index_conv = 'conv1'
                elif layer_num % 3 == 2:
                    index_conv = 'conv2'
                elif layer_num % 3 == 0:
                    index_conv = 'conv3'

                model = models.resnet50(pretrained=False)
                pretrained_dict = torch.load('./resnet50-19c8e357.pth')
                model_dict = model.state_dict()
                model_dict.update(pretrained_dict)
                model.load_state_dict(model_dict)

                use_gpu = torch.cuda.is_available()
                device = torch.device("cuda:0" if use_gpu else "cpu")
                model = torch.nn.DataParallel(model)
                model.to(device)
                model.eval()

                image_info = get_image_info(image_dir)
                image_info = image_info.to(device)
                x = image_info

                with torch.no_grad():
                    for index, layer in model.module._modules.items():
                        x = layer(x)
                        if index == 'maxpool':
                            print('Model path (image_num({}), layer_num({})): {}\t'.format(num_fig, layer_num, index),
                                  end='')
                            break

                    for index, layer in model.module.layer1._modules.items():
                        x = layer(x)
                        if index == str(index_dic[k]):
                            print('--- {}\t'.format(index), end='')
                            break

                    for index, layer in model.module.layer1[index_dic[k] + 1]._modules.items():
                        x = layer(x)
                        if index == index_conv:
                            print('--- {}\t'.format(index_dic[k] + 1), end='')
                            print('--- {}\t'.format(index))
                            break

                if index_conv == 'conv3':
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