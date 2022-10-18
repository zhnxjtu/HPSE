function [L, H] = LH_decompose(I)
% HPSE: Hierarchical Pruning via Shape-Edge Representation of Feature Map --- an Enlightenment from the primate visual system
% -----------------------------------------------------------------------
%                                INPUTS
% I  : Input feature maps.
% -----------------------------------------------------------------------
%                                OUTPUTS
% H1 : Output edge feature maps, which is transformed along the x-axis
% H2 : Output edge feature maps, which is transformed along the y-axis
% H  : Output edge feature maps
% L1 : Output shape feature maps, which is transformed along the x-axis
% L2 : Output shape feature maps, which is transformed along the y-axis
% L  : Output shape feature maps
% -----------------------------------------------------------------------

    H1 = conv2(I,[-1,1],'same');
    H2 = conv2(I',[-1,1],'same')';
    H = (H1.^2+H2.^2).^(1/2);


    L1 = conv2(I,[1,1],'same');
    L2 = conv2(I',[1,1],'same')';
    L = (L1.^2+L2.^2).^(1/2);

end