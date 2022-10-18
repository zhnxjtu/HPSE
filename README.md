# HPSE

*************************************************************************************
*This is the instruction of the code for "Hierarchical Model Compression via Shape-Edge Representation of Feature Map — an Enlightenment from the Primate Visual System".
*************************************************************************************

Contents:

    (1) In the "1.Image" folder, we give the image samples used in the paper.

    (2) In the "2.Pre-trained_Model" folder, we give the pre-trained model of different DNNs.

    (3) In the "3.Pruned_Model" folder, we demonstrate the different model after pruning.

    (4) In the "4.Pruned_Label" folder, we show the removed labels of different DNNs.

    (5) In the "5.FLOP_calculation" folder, we give the code to calculate the FOLPs and parameters of pruned model.

    (6) In other folders, we give the code to prune different models in two manners (whether to train the network from scratch).


Note: 

    - The "2.Pre-trained_Model" folder and "3.Pruned_Model" folder give the baseline model and pruned model of DenseNet, ResNet-56 and ResNet-110, respectively. 
*************************************************************************************

Pruning process:


1. DenseNet: in the "densenet" folder.

    (1) Run "FM_obtain_1.py" to get the feature map of the conv layers in the 1-st basic block (the 1-st to 12-th conv layers).

    (2) Run "main_layer_1.m" to get the label of the removed filter in the 1-st basic block.
         - the code in Line 7 is used to change the compression rate, i.e., \Delta in the main paper.

    (3) Run "dense_trainer_1.py" to restore the accuracy of the pruned model (the filters in the 1-st basic block are removed). 

    (4) Run "FM_obtain_2.py" to get the feature map of the conv layers in the 2-nd basic block (the 13-th to 24-th conv layers).

    (5) Run "main_layer_2.m" to get the label of the removed filter in the 2-nd basic block.
         - the code in Line 7 is used to change the compression rate, i.e., \Delta in the main paper.

    (6) Run "dense_trainer_2.py" to restore the accuracy of the pruned model (the filters in the 1-st and 2-nd basic block are removed). 

    - In the "densenet_scratch" folder, run "dense_trainer_scratch.py" to prune the DenseNet from scratch.
*********************************

2. ResNet-56: in the "resnet50" folder.

    (1) Run "FM_obtain_1.py" to get the feature map of the conv layers in the 1-st basic block (the 1-st to 9-th residual blocks, i.e., the 1-st to 18-th conv layers).

    (2) Run "main_layer_1.m" to get the label of the removed filter in the 1-st basic block.
         - the code in Line 7 is used to change the compression rate, i.e., \Delta in the main paper.

    (3) Run "res56_trainer_1.py" to restore the accuracy of the pruned model (the filters in the 1-st basic block are removed). 

    (4) Run "FM_obtain_2.py" to get the feature map of the conv layers in the 2-nd basic block (the 9-th to 18-th residual blocks, i.e., the 19-th to 36-th conv layers).

    (5) Run "main_layer_2.m" to get the label of the removed filter in the 2-nd basic block.
         - the code in Line 7 is used to change the compression rate, i.e., \Delta in the main paper.

    (6) Run "res56_trainer_2.py" to restore the accuracy of the pruned model (the filters in the 1-st and 2-nd basic block are removed). 

    (7) Run "FM_obtain_3.py" to get the feature map of the conv layers in the 3-rd basic block (the 19-th to 27-th residual blocks, i.e., the 37-th to 54-th conv layers).

    (8) Run "main_layer_3.m" to get the label of the removed filter in the 3-rd basic block.
         - the code in Line 7 is used to change the compression rate, i.e., \Delta in the main paper.

    (9) Run "res56_trainer_3.py" to restore the accuracy of the pruned model (all the redundant filters are removed). 

    - In the "resnet56_sctatch" folder, run "res56_trainer_scratch.py" to prune the ResNet from scratch.
*********************************

3. ResNet-110: in "resnet110" folder.

    (1) Run "FM_obtain_1.py" to get the feature map of the conv layers in the 1-st basic block (the 1-st to 18-th residual blocks, i.e., the 1-st to 36-th conv layers).

    (2) Run "main_layer_1.m" to get the label of the removed filter in the 1-st basic block.
         - the code in Line 7 is used to change the compression rate, i.e., \Delta in the main paper.

    (3) Run "res110_trainer_1.py" to restore the accuracy of the pruned model (the filters in the 1-st basic block are removed). 

    (4) Run "FM_obtain_2.py" to get the feature map of the conv layers in the 2-nd basic block (the 19-th to 36-th residual blocks, i.e., the 37-th to 72-th conv layers).

    (5) Run "main_layer_2.m" to get the label of the removed filter in the 2-nd basic block.
         - the code in Line 7 is used to change the compression rate, i.e., \Delta in the main paper.

    (6) Run "res110_trainer_2.py" to restore the accuracy of the pruned model (the filters in the 1-st and 2-nd basic block are removed). 

    (7) Run "FM_obtain_3.py" to get the feature map of the conv layers in the 3-rd basic block (the 37-th to 54-th residual blocks, i.e., the 73-th to 108-th conv layers).

    (8) Run "main_layer_3.m" to get the label of the removed filter in the 3-rd basic block.
         - the code in Line 7 is used to change the compression rate, i.e., \Delta in the main paper.

    (9) Run "res110_trainer_3.py" to restore the accuracy of the pruned model (all the redundant filters are removed). 

    - In the "resnet110_sctatch" folder, run "res110_trainer_scratch.py" to prune the ResNet from scratch.
*********************************

4. Calculate the parameters and FLOPs

    - In the "5.FLOP_calculation" folder, run "main_FLOPs.m" to get the parameters and FLOPs of pruned DNN.
*************************************************************************************

Supplementary note:

    - The pruning steps of VGG-16 and ResNet-50 are similar to the above steps.
 
This paper is accepted by IEEE Transaction on Multimedia
