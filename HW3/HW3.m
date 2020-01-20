clear all;
load('label.mat');
load('data.mat');
% data = load 'data.mat';
% train = data.imageTrain;

classmean = zeros(28,28,10);
summation = zeros(28,28,10);
for i = 1:10
    class = find(labelTrain==i-1);
    for j = 1:size(class,1)
        summation(:,:,i) = summation(:,:,i) + imageTrain(:,:,class(j));
    end
    classmean(:,:,i) = summation(:,:,i)./(size(class,1));
end

figure;
for i = 1:10
    subplot(2,5,i);
    imshow(classmean(:,:,i),[]);
end

% part2:
samplemean = reshape(classmean,[784,10]);
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


%% Covariance:
adding = zeros(28,28,10);
for i = 1:10
    sorted = imageTrain(:,:,labelTrain==i-1);
    [a,b,num_each_class] = size(sorted);
    for j = 1:num_each_class
        adding(:,:,i) = adding(:,:,i) + sorted(:,:,j);
    end
end

reshape_adding = zeros(784,1,10);
for k = 1:10
    reshape_adding(:,:,k) = reshape(adding(:,:,k),[784 1]);
end

huge = zeros(784,784,10);
huge_cov = zeros(784,784,10);
figure;
for k = 1:10
    huge(:,:,k) = reshape_adding(:,:,k) * reshape_adding(:,:,k).';
    huge_cov(:,:,k) = cov(huge(:,:,k));
    subplot(2,5,k);
    imshow(huge_cov(:,:,k),[]);
end
 