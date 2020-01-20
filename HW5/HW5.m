clear all;
clc;
load('data.mat');
load('label.mat');
%% Part 1
k_mean = zeros(28,28,10);
for i = 1:10
    k_mean(:,:,i) = randi([0 225],28,28);
end

class_addition = zeros(28,28,10); 
class_length = zeros(10,1); % number of images in a class
label = zeros(5000,1);
label_old = zeros(5000,1);
change_rate = 1;

while(change_rate>0.002)
    for i = 1:5000
        for j = 1:10
            diff = imageTrain(:,:,i)-k_mean(:,:,j);
            arg = sum(diff(:).^2);
            if j == 1
                arg_min = arg;
                label(i) = 1;
                continue
            end
            if arg_min>arg
                arg_min = arg;
                label(i) = j;
            end
        end
    end
    change_rate = length(find(label_old(:)~=label(:)))/5000
    label_old = label;
    
    % update mean:
    for i = 1:5000
        class_addition(:,:,label(i)) = class_addition(:,:,label(i)) + imageTrain(:,:,i); % images with the same class added together
        class_length(label(i),1) = class_length(label(i),1) + 1;
    end
    for i = 1:10
        k_mean(:,:,i) = class_addition(:,:,i) ./ class_length(i,1);
    end
end

for i = 1:10
    subplot(2,5,i);
    imshow(k_mean(:,:,i),[0 255]);
end



%% Part 2
initial_index = zeros(10,1);
k_mean = zeros(28,28,10);
for i = 1:10
    initial_index(i,1) = randi([1 5000],1,1);
    k_mean(:,:,i) = imageTrain(:,:,initial_index(i,1)); 
end
class_addition = zeros(28,28,10);
class_length = zeros(10,1); % number of images in a class
label = zeros(5000,1);
label_old = zeros(5000,1);
change_rate = 1;
while(change_rate>0.002)
    for i = 1:5000
        for j = 1:10
            diff = imageTrain(:,:,i)-k_mean(:,:,j);
            arg = sum(diff(:).^2);
            if j == 1
                arg_min = arg;
                label(i) = 1;
                continue
            end
            if arg_min>arg
                arg_min = arg;
                label(i) = j;
            end
        end
    end
    change_rate = length(find(label_old(:)~=label(:)))/5000
    label_old = label;
    
    % update mean:
    for i = 1:5000
        class_addition(:,:,label(i)) = class_addition(:,:,label(i)) + imageTrain(:,:,i); % images with the same class added together
        class_length(label(i),1) = class_length(label(i),1) + 1;
    end
    for i = 1:10
        k_mean(:,:,i) = class_addition(:,:,i) ./ class_length(i,1);
    end
end
figure;
for i = 1:10
    subplot(2,5,i);
    imshow(k_mean(:,:,i),[]);
end

figure;
for i = 1:10
    subplot(2,5,i);
    imshow(imageTrain(:,:,initial_index(i,1)),[]);
end


%% Manually-Labeling for part 3:
k_mean_sorted = zeros(28,28,10);

k_mean_sorted(:,:,1) = k_mean(:,:,7);%0
k_mean_sorted(:,:,2) = k_mean(:,:,10);%1
k_mean_sorted(:,:,3) = k_mean(:,:,1);%2
k_mean_sorted(:,:,4) = k_mean(:,:,4);%3
k_mean_sorted(:,:,5) = k_mean(:,:,5);%4
k_mean_sorted(:,:,6) = k_mean(:,:,8);%5
k_mean_sorted(:,:,7) = k_mean(:,:,3);%6
k_mean_sorted(:,:,8) = k_mean(:,:,6);%7
k_mean_sorted(:,:,9) = k_mean(:,:,9);%8
k_mean_sorted(:,:,10) = k_mean(:,:,2);%9

%% Part 3
samplemean = reshape(k_mean_sorted,[784,10]);
diff = zeros(784,10);
diff_trans = zeros(10,784);
i_x = zeros(10,1);

sampletest = reshape(imageTest,[784,500]);
prediction = zeros(500,1);

 for i = 1:500
    for j = 1:10
        diff(:,j) = sampletest(:,i) - samplemean(:,j);
        diff_trans(j,:) = diff(:,j).';
        i_x(j,1) = (-1/2)*(diff_trans(j,:) * diff(:,j));
    end
    [a,prediction(i,1)] = max(i_x);
 end
 
 for i = 1:500
     prediction(i,1) = prediction(i,1)-1;
 end
 
[total_error_num,col] = size(find(prediction~=labelTest));
total_error_rate = total_error_num/500;

error_num_for_each_class = zeros(10,1);
num_for_each_class = zeros(10,1);
for i = 1:10
    [num_for_each_class(i),dummyy] = size(find(labelTest==i-1));
    [error_num_for_each_class(i),dummy] = size(find(prediction(labelTest==i-1) ~= i-1));
end
error_rate = error_num_for_each_class./num_for_each_class;


%Error_plot:
figure;
index = (0:1:9);
bar(index,error_rate);
xlabel('Class');
ylabel('Error Rate');


%Error_Table:
Class = {'0';'1';'2';'3';'4';'5';'6';'7';'8';'9'};
Correctly_classified = num_for_each_class - error_num_for_each_class;
Incorrectly_classified = error_num_for_each_class;
Total_num_each_class = num_for_each_class;
Error_rate = error_rate;
T = table(Class,Total_num_each_class,Correctly_classified,Incorrectly_classified,Error_rate);


