%% (1)
clear all;
sampletrain = double(imread('sampletrain.png'));
sampletest = double(imread('sampletest.png'));


sampletest = sampletest(:);
sampletrain = sampletrain(:);

% Sample Amplifier:
a = (sampletrain.' * sampletest)/(sampletrain.' * sampletrain);

%% (2) classification using least square distance metric:
load 'data.mat';
%normalization:
imageTestNew = imageTestNew./a;

% Euclidean distance
EDistance = zeros(500,5000);
smallest = zeros(500,4);
for i = 1:500
    for j = 1:5000
        difference = (imageTestNew(:,:,i) - imageTrain(:,:,j)).^2;
        summation = sum(difference(:));
        EDistance(i,j) = sqrt(summation);
        if j==1
            smallest(i,1) = EDistance(i,1);
            smallest(i,2) = i;
            smallest(i,3) = j;
        end
        if EDistance(i,j)<smallest(i,1)
            smallest(i,1) = EDistance(i,j);
            smallest(i,2) = i;
            smallest(i,3) = j;
        end
    end
end
smallest(:,4) = labelTrain(smallest(:,3)); % columns in 'smallest': shortest_distance, test_idex, train_index, train(predict)_label

% total_error:
total_error = find(smallest(:,4)~=labelTestNew);
total_error_rate = size(total_error,1)/500;

error_num = zeros(10,1);
class_num = zeros(10,1);
 for i = 1:10
     error_num(i) = size(find(smallest(labelTestNew(:)==i-1,4)~=i-1),1);
     class_num(i) = size(find(labelTestNew(:)==i-1),1);
 end
error_rate_each_class = error_num./class_num;

% Plot:
index = (0:1:9);
bar(index,error_rate_each_class(:,1));
xlabel('Class');
ylabel('Error Rate');
%Error_Table:
Class = {'0';'1';'2';'3';'4';'5';'6';'7';'8';'9'};
Correctly_classified = class_num - error_num;
Incorrectly_classified = error_num;
Total_num_of_images = class_num;
Error_rate = error_rate_each_class(:,1);
T = table(Class,Total_num_of_images,Correctly_classified,Incorrectly_classified,Error_rate);




%% (3) NN classifier without MLE
clear all;
load 'data.mat';

%normalization:
%imageTestNew = imageTestNew./a;

% Euclidean distance
EDistance = zeros(500,5000);
smallest = zeros(500,4);

for i = 1:500
    for j = 1:5000
        difference = (imageTestNew(:,:,i) - imageTrain(:,:,j)).^2;
        summation = sum(difference(:));
        EDistance(i,j) = sqrt(summation);
        if j==1
            smallest(i,1) = EDistance(i,1);
            smallest(i,2) = i;
            smallest(i,3) = j;
        end
        if EDistance(i,j)<smallest(i,1)
            smallest(i,1) = EDistance(i,j);
            smallest(i,2) = i;
            smallest(i,3) = j;
        end
    end
end
smallest(:,4) = labelTrain(smallest(:,3)); % columns in 'smallest': shortest_distance, test_idex, train_index, train(predict)_label

% total_error:
total_error = find(smallest(:,4)~=labelTestNew);
total_error_rate = size(total_error,1)/500;


error_num = zeros(10,1);
class_num = zeros(10,1);
 for i = 1:10
     error_num(i) = size(find(smallest(labelTestNew(:)==i-1,4)~=i-1),1);
     class_num(i) = size(find(labelTestNew(:)==i-1),1);
 end
error_rate_each_class = error_num./class_num;

% Plot:
index = (0:1:9);
bar(index,error_rate_each_class(:,1));
xlabel('Class');
ylabel('Error Rate');

%Error_Table:
Class = {'0';'1';'2';'3';'4';'5';'6';'7';'8';'9'};
Correctly_classified = class_num - error_num;
Incorrectly_classified = error_num;
Total_num_of_images = class_num;
Error_rate = error_rate_each_class(:,1);
T = table(Class,Total_num_of_images,Correctly_classified,Incorrectly_classified,Error_rate);
