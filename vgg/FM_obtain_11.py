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

import VGG16 as vgg

def get_image_info(image_dir):
    image_info = Image.open(image_dir).convert('RGB')

    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

    transform = transforms.Compose([
        transforms.RandomCrop(32),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    image_info = transform(image_info)
    image_info = image_info.unsqueeze(0)
    return  image_info

def get_k_layer_feature_map(feature_extractor, k, x):
    with torch.no_grad():
        for index, layer in enumerate(feature_extractor):
            x = layer(x)
            if k == index:
                return x

if __name__ == '__main__':
    for num_fig in range(1,501):
        image_dir = '../1.Image/Image_CIFAR10/' + str(num_fig) + '.png'

        layer_num = 11
        conv_index = [0, 3, 7, 10, 14, 17, 20, 24, 27, 30, 34, 37, 40]

        model = vgg.VGG()

        checkpoint = torch.load('./model-10/model.th')
        model.load_state_dict(checkpoint['state_dict'])

        use_gpu = torch.cuda.is_available()
        device = torch.device("cuda:0" if use_gpu else "cpu")
        model = nn.DataParallel(model)
        model.to(device)
        model.eval()

        image_info = get_image_info(image_dir)
        image_info = image_info.to(device)
        feature_extractor = model.module.features

        feature_map = get_k_layer_feature_map(feature_extractor, conv_index[layer_num-1], image_info)

        if not os.path.exists('./FM'):
            os.mkdir('./FM')

        if not os.path.exists('./FM/FM-' + str(layer_num)):
            os.mkdir('./FM/FM-' + str(layer_num))

        if not os.path.exists('./FM/FM-' + str(layer_num) + '/' + str(num_fig)):
            os.mkdir('./FM/FM-' + str(layer_num) + '/' + str(num_fig))

        for b_num in range(feature_map.shape[1]):
            np.savetxt('./FM/FM-' + str(layer_num) + '/' + str(num_fig) + '/conv' + str(layer_num) + '_' + str(
                b_num + 1) + '.csv', feature_map[0, b_num, :, :].cpu().numpy(), delimiter=',')
