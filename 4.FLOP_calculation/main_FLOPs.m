clc
clear all
close all

% calculate the FLOP and parameters of DenseNet
name_floder_den = ['../4.Pruned_Label/pruned_label_dense_0.40'];
% name_floder_den = ['../4.Pruned_Label/pruned_label_dense_0.50'];
[parameter_rate_den, FLOPs_rate_den] = FLOPs_densenet(name_floder_den);

% calculate the FLOP and parameters of ResNet-56
name_floder_res56 = ['../4.Pruned_Label/pruned_label_res56_0.35'];
% name_floder_res56 = ['../4.Pruned_Label/pruned_label_res56_0.53'];
[parameter_rate_res56, FLOPs_rate_res56] = FLOPs_res56(name_floder_res56);

% calculate the FLOP and parameters of ResNet-110
name_floder_res110 = ['../4.Pruned_Label/pruned_label_res110_0.40'];
% name_floder_res110 = ['../4.Pruned_Label/pruned_label_res110_0.50'];
[parameter_rate_res110, FLOPs_rate_res110] = FLOPs_res110(name_floder_res110);

% calculate the FLOP and parameters of VGG-16
% name_floder_vgg = ['../4.Pruned_Label/pruned_label_vgg_0.25'];
name_floder_vgg = ['../4.Pruned_Label/pruned_label_vgg_0.45'];
% name_floder_vgg = ['../4.Pruned_Label/pruned_label_vgg_0.60'];
[parameter_rate_vgg, FLOPs_rate_vgg] = FLOPs_vgg(name_floder_vgg);

% calculate the FLOP and parameters of ResNet-50
name_floder_res50 = ['../4.Pruned_Label/pruned_label_res50_0.30'];
% name_floder_res50 = ['../4.Pruned_Label/pruned_label_res50_0.37'];
[parameter_rate_res50, FLOPs_rate_res50] = FLOPs_res50(name_floder_res50);