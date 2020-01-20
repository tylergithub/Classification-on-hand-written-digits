clear all;
close all;
clc
load 'data.mat';
load 'label.mat';

% distance between TestData and TrainData:
EDistance = zeros(500,5003);
small = zeros(1,2); % value and index
r_rate = zeros(500, 10); % for error_rate calculation 
for i = 1:500
    for j = 1:5000
        difference = (imageTrain(:,:,j) - imageTest(:,:,i)).^2;
        summation = sum(difference(:));
        EDistance(i,j) = sqrt(summation);
        if j == 1
            small(1) = EDistance(i,1);
            small(2) = 1;
        end
        if EDistance(i,j) < small(1)
            small(1) = EDistance(i,j);
            small(2) = j;
        end
    end
    EDistance(i,5001) = small(1); % predicted value
    EDistance(i,5002) = small(2); % predicted index
    EDistance(i,5003) = labelTrain(small(2)); % predicted class
    
    for k = 1:9
        if labelTest(i) == k
            if EDistance(i,5003) == labelTest(i)
                r_rate(i,k) = 1;
            end
        end
    end
    if labelTest(i) == 0
        if EDistance(i,5003) == labelTest(i)
            r_rate(i,10) = 1;
        end
    end
end
error_rate = zeros(10,3); %error rate, class_total#, and class_correct

% Error_rate per class:
for l = 1:9
    error_rate(l+1,1) = 1 - ((sum(r_rate(:,l) == 1)) / sum(labelTest(:) == l));
    error_rate(l+1,3) = sum(labelTest(:) == l);
    error_rate(l+1,2) = sum(r_rate(:,l) == 1);
end
error_rate(1,1) = 1 - ((sum(r_rate(:,10) == 1)) / sum(labelTest(:) == 0));
error_rate(1,3) = sum(labelTest(:) == 0);
error_rate(1,2) = sum(r_rate(:,10) == 1);

% Plot:
index = (0:1:9);
bar(index,error_rate(:,1));
xlabel('Class');
ylabel('Error Rate');

%Error_Table:
Class = {'0';'1';'2';'3';'4';'5';'6';'7';'8';'9'};
Correctly_classified = error_rate(:,2);
Incorrectly_classified = error_rate(:,3) - error_rate(:,2);
Total_num_of_images = error_rate(:,3);
Error_rate = error_rate(:,1);
T = table(Class,Total_num_of_images,Correctly_classified,Incorrectly_classified,Error_rate);

% Total Error_rate:
Total_error_rate = (sum(EDistance(:,5003) ~= labelTest(:)))/500;

% 5 misclassified images:
% Compare the classes between the 5003th columns in EDistance and classes
% in labelTest. Yield the 5th, 7th, 17th, 25th, and 44th images are
% different:

% 5th:
figure;
imshow(imageTrain(:,:,1733));
figure;
imshow(imageTest(:,:,5));
% 7th:
figure;
imshow(imageTrain(:,:,55));
figure;
imshow(imageTest(:,:,7));
% 17th:
figure;
imshow(imageTrain(:,:,2053));
figure;
imshow(imageTest(:,:,17));
% 25th:
figure;
imshow(imageTrain(:,:,4207));
figure;
imshow(imageTest(:,:,25));
% 44th:
figure;
imshow(imageTrain(:,:,3593));
figure;
imshow(imageTest(:,:,44));




