clear all;
close all;
clc;
load('data.mat');
load('label.mat');
train = reshape(imageTrain,[784,5000])/255;
test = reshape(imageTest,[784,500])/255;

%% Part 1
% compute sample mean:
train_mean = mean(train,2); % get 784*1 mean of all 5000 imageTrain
% cov
train_cov = cov(train'); % rows are observations while columns are random variables
% eigens:
[eigen_vector,eigen_value] = eig(train_cov);
% sort:
eigen_value = diag(eigen_value);
[eigen_value_sort,I] = sort(eigen_value,'descend');
eigen_vactor_sort = eigen_vector(:,I(1:784)); % flip becasue it is already 
% in ascending order just like it in eigen_value
eigen_vactor_sort2828 = reshape(eigen_vactor_sort,[28,28,784]);

figure;
title('Top 10 Principle Components');
for i = 1:100
    subplot(10,10,i);
    imshow(eigen_vactor_sort2828(:,:,i),[]);
end

figure;
index = 0:1:783;
plot(index,eigen_value_sort);
title('Eigen Values');
xlabel('feature Space Dimension(pixels)');

%% class 5 only:
% compute sample mean:
train_5 = train(:,labelTrain==5);
class_5_mean = mean(train_5,2);
% cov
class_5_cov = cov(train_5.'); % rows are observations while columns are random variables
% eigen:
[eigen_vector,eigen_value] = eig(class_5_cov);
% sort:
eigen_value = diag(eigen_value);
[eigen_value_sort,I] = sort(eigen_value,'descend');
eigen_vactor_sort = eigen_vector(:,I(:));
eigen_vactor_sort2828 = reshape(eigen_vactor_sort,[28,28,784]);
figure;
for i = 1:10
    subplot(2,5,i);
    imshow(eigen_vactor_sort2828(:,:,i),[]);
end
figure;
index = 0:1:783;
plot(index,eigen_value_sort);

%% Part 2:feature space with top k's:
figure;

k_loop_counter = 1;
for k = [5,10,20,30,40,60,90,130,180,250]
train_mean = mean(train,2); % get 784*1 mean of all 5000 imageTrain
% cov
train_cov = cov(train'); % rows are observations while columns are random variables?5000*784
% eigens:
[eigen_vector,eigen_value] = eig(train_cov);

% sort:
eigen_value = diag(eigen_value);
[eigen_value_sort,I] = sort(eigen_value,'descend');

% subspace:
eigen_value_lower = eigen_value_sort(1:k,:);
eigen_vactor_lower = eigen_vector(:,I(1:k));

% project imageTest into lower dimension:
test = test - train_mean;
test_lower = eigen_vactor_lower' * test;
% project imageTrain into lower dimension:
train = train - train_mean;
train_lower = eigen_vactor_lower' * train;

% calculate cov:
%cov_lower = cov(train_lower'); %*************************************
% Apply BDR:
classmean_lower = zeros(k,10);
sum_lower = zeros(k,10);
for i = 1:10
    index_each_class_lower = find(labelTrain==i-1);
    for j = 1:length(index_each_class_lower)
        sum_lower(:,i) = sum_lower(:,i) + train_lower(:,index_each_class_lower(j));
    end
    classmean_lower(:,i) = sum_lower(:,i)./(length(index_each_class_lower));
end
diff = zeros(k,10);
i_x = zeros(10,1);
prediction = zeros(500,1);
for i = 1:500
    for j = 1:10
        index_each_class_lower = find(labelTrain==j-1);
        each_class_lower = train_lower(:,index_each_class_lower);
        cov_each_class_lower = cov(each_class_lower');
        
        diff(:,j) = test_lower(:,i) - classmean_lower(:,j);
        %i_x(j,1) = (-1/2)*(diff(:,j)' * diff(:,j)); % with cov = 1
        %i_x(j,1) = (-1/2)*(diff(:,j)'/cov_lower * diff(:,j));% with train-cov
        %i_x(j,1) = (-1/2)*(diff(:,j)'/cov_each_class_lower * diff(:,j));% with class-train-cov
        i_x(j,1) = mvnpdf(test_lower(:,i),classmean_lower(:,j),cov_each_class_lower);
    end
    [a,prediction(i,1)] = max(i_x); % return the max value and its index
end
for i = 1:500
     prediction(i,1) = prediction(i,1)-1;
end
% Calculate total errors:
total_error_num = length(find(prediction~=labelTest));
total_error_rate(k_loop_counter) = total_error_num/500;
% Calculate errors for each class:
error_num_for_each_class = zeros(10,1);
num_for_each_class = zeros(10,1);
for i = 1:10
    num_for_each_class(i) = length(find(labelTest==i-1));
    error_num_for_each_class(i) = length(find(prediction(labelTest==i-1) ~= i-1));
end
error_rate_for_each_class = error_num_for_each_class./num_for_each_class;
% plots:
index = (0:1:9);
subplot(3,4,k_loop_counter);
bar(index,error_rate_for_each_class);
ylim([0 1]);
xlabel('Class');
ylabel('Error Rate');


k_loop_counter = k_loop_counter + 1;
end
%% total error plot for part 2:
figure;
plot([5,10,20,30,40,60,90,130,180,250],total_error_rate);
title('Principle Component Analysis');
ylabel('Error rate');
xlabel('Dimensions');

%% part 3 least like 5:
% class 5 only:
class_5_only = train(:,labelTrain==5);
% class 5 mean:
class_5_only_mean = mean(class_5_only,2);
% class 5 conv:
class_5_conv = cov(class_5_only');
%eigen value and eigen vectors:
[eigen_vector,eigen_value] = eig(class_5_conv);
eigen_value = diag(eigen_value);
% sort:
[eigen_value_sort,I] = sort(eigen_value,'descend');
eigen_vector_sort = eigen_vector(:,I(:));
% the last (784-40 = 744)744 non-principal components:
eigen_vector_sort_non_principal = eigen_vector_sort(:,41:784);
projected = eigen_vector_sort_non_principal' * (test-class_5_only_mean);

% norm of each image:
%[a,index] = max(vecnorm(projected));
energy_norms = zeros(1,500);
for i = 1:500
    energy_norms(1,i) = norm(projected(:,i));
end
[norm_value,norm_I] = sort(energy_norms,'descend');

images_least_like_5 = reshape(test(:,norm_I(1:10)),[28,28,10]);

figure;
for i = 1:10
    subplot(2,5,i);
    imshow(images_least_like_5(:,:,i),[]);
end





%%

% table:
Class = {'0';'1';'2';'3';'4';'5';'6';'7';'8';'9'};
Correctly_classified = num_for_each_class - error_num_for_each_class;
Incorrectly_classified = error_num_for_each_class;
Total_num_each_class = num_for_each_class;
Error_rate = error_rate_for_each_class;
T = table(Class,Total_num_each_class,Correctly_classified,Incorrectly_classified,Error_rate);




%%
load('data.mat');
load('label.mat');
prediction = labelTest;
prediction(6) = 4;
prediction(67) = 4;
prediction(345) = 3;
prediction(99) = 9;
prediction(4) = 2;
prediction(63) = 1;
prediction(456) = 7;
figure;
plotconfusion(labelTest,prediction);



