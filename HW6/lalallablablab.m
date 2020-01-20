clear all;
close all;
clc;
load('data.mat');
load('label.mat');
train = reshape(imageTrain,[784,5000])/255;
test = reshape(imageTest,[784,500])/255;

%% 
% compute sample mean:
sample_mean = mean(train,2); % get 784*1 mean of all 5000 imageTrain
% cov
sample_cov = cov(train'); % rows are observations while columns are random variables
% eigens:
[eigen_vector,eigen_value] = eig(sample_cov);
% sort:
eigen_value = diag(eigen_value);
eigen_value_sort = sort(eigen_value,'descend');
eigen_vactor_sort = fliplr(eigen_vector); % flip becasue it is already in ascending order just like it in eigen_value
eigen_vactor_sort2828 = reshape(eigen_vactor_sort,[28,28,784]);
figure;
for i = 1:10
    subplot(2,5,i);
    imshow(eigen_vactor_sort2828(:,:,i),[]);
end
figure;
index = 0:1:783;
plot(index,eigen_value_sort);

%% redo only with digit 5:
% compute sample mean:
train_only_5 = train(:,labelTrain==5);
sample_mean = mean(train_only_5,2);
% cov
sample_cov = cov(train_only_5.'); % rows are observations while columns are random variables
% eigens:
[eigen_vector,eigen_value] = eig(sample_cov);
% sort:
eigen_value = diag(eigen_value);
eigen_value_sort = sort(eigen_value,'descend');
eigen_vactor_sort = fliplr(eigen_vector);
eigen_vactor_sort2828 = reshape(eigen_vactor_sort,[28,28,784]);
figure;
for i = 1:10
    subplot(2,5,i);
    imshow(eigen_vactor_sort2828(:,:,i),[]);
end
figure;
index = 0:1:783;
plot(index,eigen_value_sort);

%% feature space with top k's:
figure;

k_loop_counter = 1;
for k = [5,10,20,30,40,60,90,130,180,250,350]
sample_mean = mean(train,2); % get 784*1 mean of all 5000 imageTrain
% cov
sample_cov = cov(train.'); % rows are observations while columns are random variables?5000*784
% eigens:
[eigen_vector,eigen_value] = eig(sample_cov);
% sort:
eigen_value = diag(eigen_value);
eigen_value_sort = sort(eigen_value,'descend'); % inverse of the original
eigen_vactor_sort = fliplr(eigen_vector); % flip becasue it is already in ascending order just like it in eigen_value
% subspace:
eigen_value_lower = eigen_value_sort(1:k,:);
eigen_vactor_lower = eigen_vactor_sort(:,1:k);

% project imageTest into lower dimension:
test = test - sample_mean;
test_lower = eigen_vactor_lower.' * test;
% project imageTrain into lower dimension:
train = train - sample_mean;
train_lower = eigen_vactor_lower.' * train;

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
        diff(:,j) = test_lower(:,i) - classmean_lower(:,j);
        i_x(j,1) = (-1/2)*(diff(:,j).' * diff(:,j));
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
% table:
Class = {'0';'1';'2';'3';'4';'5';'6';'7';'8';'9'};
Correctly_classified = num_for_each_class - error_num_for_each_class;
Incorrectly_classified = error_num_for_each_class;
Total_num_each_class = num_for_each_class;
Error_rate = error_rate_for_each_class;
T = table(Class,Total_num_each_class,Correctly_classified,Incorrectly_classified,Error_rate);



k_loop_counter = k_loop_counter + 1;
end

%% 
figure;
plot([5,10,20,30,40,60,90,130,180,250,350],total_error_rate);