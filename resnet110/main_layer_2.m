clc;
clear all;
close all

for num_of_layer = 37:72
    
    eta = 0.5;             % compression ratio
    imag_num = 500;
    
    % num of filter
    if (1 <= num_of_layer) && (num_of_layer <=36)
        num_of_filter = 16;
        k = 1;
    elseif (37 <= num_of_layer) && (num_of_layer <= 72)
        num_of_filter = 32;
        k = 2;
    elseif (73 <= num_of_layer) && (num_of_layer <= 108)
        num_of_filter = 64;
        k = 4;
    end
    
    for num_of_imag = 1:imag_num
        
        for i = 1:num_of_filter
            I = csvread(['./FM/FM-' num2str(num_of_layer) '/' num2str(num_of_imag) '/conv' num2str(num_of_layer) '_' num2str(i) '.csv']);
            [L(i,:,:), H(i,:,:)] = LH_decompose(I);
        end
        
        L_re = reshape(L,num_of_filter,size(L,2)*size(L,3));
        index_cluster(num_of_imag, :) = kmeans(L_re,k);

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% make the smaller to 1 %%%%%%%%
        ind1 = find(index_cluster(num_of_imag, :) == 1);
        ind2 = find(index_cluster(num_of_imag, :) == 2);
        comp1 = sum(sum(L_re(ind1,:))/length(ind1));
        comp2 = sum(sum(L_re(ind2,:))/length(ind2));
        [~,comp_ind] = sort([comp1,comp2]);
        eval(['index_cluster_new(num_of_imag, ind' num2str(comp_ind(1)) ') = 1;']);
        eval(['index_cluster_new(num_of_imag, ind' num2str(comp_ind(2)) ') = 2;']);
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        for j = 1:num_of_filter
            s_H(num_of_imag, j) = sum(sum(H(j,:,:)));
        end
        
    end
    
    ave_H = 1/imag_num * sum(s_H); 
    
       
    for clu_col = 1:size(index_cluster_new,2)  
        index_cluster_final(clu_col) = mode(index_cluster_new(:,clu_col));
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
    
    clear all
  
end
