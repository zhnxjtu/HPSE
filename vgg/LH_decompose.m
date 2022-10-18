% clc;
% clear all;
% close all
function [L, H] = LH_decompose(I)

% I = csvread('./conv0_4.csv');

% new_x = size(I,1)-1;
% new_y = size(I,2)-1;

H1 = conv2(I,[-1,1],'same');
H2 = conv2(I',[-1,1],'same')';
H = (H1.^2+H2.^2).^(1/2);

% H_new = H(2:new_x,2:new_y);

% figure
% colormap jet
% imagesc(H(2:size(H-1),2:size(H-1)))

L1 = conv2(I,[1,1],'same');
L2 = conv2(I',[1,1],'same')';
L = (L1.^2+L2.^2).^(1/2);
% L_new = L(2:new_x,2:new_y);

% figure
% colormap jet
% imagesc(L(2:size(L-1),2:size(L-1)))

% % figure
% % h = imhist(L,256);
% % h0 = h/(sum(h));
% % h1=h0(1:1:256);
% % horz=1:1:256;
% % stem(horz,h1,'fill')
% % axis([0 255 0 0.1]) %其中参数可调整 

% lower = min(min(L(2:new_x,2:new_y)));
% upper = min(min(L(2:new_x,2:new_y))) + 100*(max(max(L(2:new_x,2:new_y)))-min(min(L(2:new_x,2:new_y))))/255;

% for ii = 2:new_x
%     for jj = 2:new_y
%         if L(ii,jj)>=lower && L(ii,jj)<upper
%             L_new(ii-1,jj-1) = 0;
%         end
%     end
% end


% % figure
% % h = imhist(L,256);
% % h0 = h/(sum(h));
% % h1=h0(1:1:256);
% % horz=1:1:256;
% % stem(horz,h1,'fill')
% % axis([0 255 0 0.1]) %其中参数可调整 
% 
% figure
% colormap jet
% imagesc(L(2:size(L)-1,2:size(L)-1))

end