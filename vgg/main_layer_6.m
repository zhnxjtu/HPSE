clc;
clear all;
close all

imag_num = 500;

for num_of_imag = 1:imag_num

    eta = 0.25;             % compression ratio

    num_of_layer = 6;

    % num of filter
    if (1 <= num_of_layer) && (num_of_layer <=2)
        num_of_filter = 64;
        k = 4;
    elseif (3 <= num_of_layer) && (num_of_layer <= 4)
        num_of_filter = 128;
        k = 8;
    elseif (5 <= num_of_layer) && (num_of_layer <= 7)
        num_of_filter = 256;
        k = 16;
    elseif (8 <= num_of_layer) && (num_of_layer <= 10)
        num_of_filter = 512;
        k = 32;
    elseif (11 <= num_of_layer) && (num_of_layer <= 13)
        num_of_filter = 512;
        k = 32;
    end

    for i = 1:num_of_filter
        I = csvread(['./FM/FM-' num2str(num_of_layer) '/' num2str(num_of_imag) '/conv' num2str(num_of_layer) '_' num2str(i) '.csv']);
        [L(i,:,:), H(i,:,:)] = LH_decompose(I);
    end

    L_re = reshape(L,num_of_filter,size(L,2)*size(L,3));
    index_cluster(num_of_imag, :) = kmeans(L_re,k);

    comp = [];
    for kk = 1:k
        eval(['ind' num2str(kk) '= find(index_cluster(num_of_imag, :) == kk);']);
        eval(['comp' num2str(kk) '= sum(sum(L_re(ind' num2str(kk) ',:))/length(ind' num2str(kk) '));']);
        eval(['comp = [comp, comp' num2str(kk) '];']);
    end

    [~, comp_ind] = sort(comp);

    for kk = 1:k
        eval(['index_cluster_new(num_of_imag, ind' num2str(comp_ind(kk)) ') = kk;']);
    end

    for j = 1:num_of_filter
        s_H(num_of_imag, j) = sum(sum(H(j,:,:)));
    end

end

ave_H = 1/imag_num * sum(s_H);

for clu_col = 1:size(index_cluster_new,2)
    index_cluster_final(clu_col) = mode(index_cluster_new(:, clu_col));
end

for clu_val = 1:k
    
    index_L = find(index_cluster_final' == clu_val);
    
    [~, index_H] = sort(ave_H(index_L));
    
    eval(['pruned_label' num2str(clu_val) '= index_L(index_H);']);
    
    clear index_L
    clear index_H
end

removed_num = eta*num_of_filter;
pruned_label = [];
for ii = 1:k
    eval(['par(i)= length(pruned_label' num2str(ii) ')/num_of_filter;']);
    eval(['pruned_label = [pruned_label;pruned_label' num2str(ii) '(1:floor(par(i)*removed_num))];']);
end

if ~exist('./pruned_label/', 'dir')
        mkdir('./pruned_label')
end

filename = fopen(['./pruned_label/FM' num2str(num_of_layer) '.txt'], 'wt');
fprintf(filename, '%g\n', eval('pruned_label'));

