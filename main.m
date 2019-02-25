% The following code snippet shows how data is generated for all
% combination of |theta1| and |theta2| values and saved into a matrix to be
% used as training data. The reason for saving the data in two matrices is
% explained in the following section.

l1 = 10; % length of first arm
l2 = 7; % length of second arm
l3 = 4;

nr_p = 50;

theta1 = linspace(0, pi/4, nr_p); % all possible theta1 values
theta2 = linspace(0, pi/2, nr_p); % all possible theta2 values
% theta3 = linspace(0, pi/4, nr_p); % all possible theta3 values

[THETA1,THETA2] = meshgrid(theta1,theta2);

% X = l1 * cos(THETA1) + l2 * cos(THETA1 + THETA2) + l3 * cos(THETA1 + THETA2 + THETA3); % compute x coordinates
% Y = l1 * sin(THETA1) + l2 * sin(THETA1 + THETA2) + l3 * sin(THETA1 + THETA2 + THETA3); % compute y coordinates

X = l1 * cos(THETA1) + l2 * cos(THETA1 + THETA2);
Y = l1 * sin(THETA1) + l2 * sin(THETA1 + THETA2);

data1 = [X(:) Y(:) THETA1(:)]; % create x-y-theta1 dataset
data2 = [X(:) Y(:) THETA2(:)]; % create x-y-theta2 dataset
% data3 = [X(:) Y(:) THETA3(:)]; % create x-y-theta3 dataset

train_data1 = data1(1:2:end,:);
train_data2 = data2(1:2:end,:);
% train_data3 = data3(1:2:end,:);
val_data1 = data1(2:2:end,:);
val_data2 = data2(2:2:end,:);
% val_data3 = data3(2:2:end,:);

%%
% The following plot shows all the X-Y data points generated by cycling
% through different combinations of |theta1| and |theta2| and deducing x
% and y coordinates for each. The plot can be generated by using the
% code-snippet shown below. The plot is illustrated further for easier
% understanding.
%
plot(X(:),Y(:),'r.'); 
axis equal;
xlabel('X','fontsize',10)
ylabel('Y','fontsize',10)
title('X-Y coordinates generated for all theta1 and theta2 combinations using forward kinematics formula','fontsize',10)
%
% <<../invkine_grid.png>>
%
%%
opt = anfisOptions;
opt.InitialFIS = 7;
opt.EpochNumber = 150;
epoch = 1:150;
opt.DisplayANFISInformation = 0;
opt.DisplayErrorValues = 0;
opt.DisplayStepSize = 0;
opt.DisplayFinalResults = 0;

%%
% Train an ANFIS system using the first set of training data, |data1|.
disp('--> Training first ANFIS network.')

opt.ValidationData = val_data1;

[anfis1,trnErr,ss,anfis12,chkErr] = anfis(train_data1,opt);

figure
plot(epoch,trnErr,'o',epoch,chkErr,'x')
hold on;
plot(epoch,[trnErr chkErr])
hold off;

%%
% Change the number of input membership functions and train an ANFIS system
% using the second set of training data, |data2|.
disp('--> Training second ANFIS network.')
opt.InitialFIS = 6;
opt.ValidationData = val_data2;

[anfis2,trnErr,ss,anfis22,chkErr] = anfis(train_data2,opt);

figure
plot(epoch,trnErr,'o',epoch,chkErr,'x')
hold on;
plot(epoch,[trnErr chkErr])
hold off;

%%
% Change the number of input membership functions and train an ANFIS system
% using the second set of training data, |data2|.
disp('--> Training third ANFIS network.')
opt.InitialFIS = 5;
opt.ValidationData = val_data3;

[anfis3,trnErr,ss,anfis32,chkErr] = anfis(train_data3,opt);

figure
plot(epoch,trnErr,'o',epoch,chkErr,'x')
hold on;
plot(epoch,[trnErr chkErr])
hold off;

%%
% <matlab:edit('traininv') Click here for unvectorized code>
%%
% extract validation data as set of XY
XY = val_data1(:,1:2);
THETA1P = evalfis(XY,anfis1); % theta1 predicted by anfis1
THETA2P = evalfis(XY,anfis2); % theta2 predicted by anfis2
% THETA3P = evalfis(XY,anfis3); % theta3 predicted by anfis3

%% calculate overall RMSE on angles
RMSE1 = sqrt(mean((THETA1P-val_data1(:,3)).^2))
RMSE2 = sqrt(mean((THETA2P-val_data2(:,3)).^2))
%RMSE2 = sqrt(mean((THETA3P-val_data3(:,3)).^2))

%% calculate overall RMSE on position
% Xp = l1 * cos(THETA1P) + l2 * cos(THETA1P + THETA2P) + l3 * cos(THETA1P + THETA2P + THETA3P); % compute x coordinates
% Yp = l1 * sin(THETA1P) + l2 * sin(THETA1P + THETA2P) + l3 * sin(THETA1P + THETA2P + THETA3P); % compute y coordinates

Xp = l1 * cos(THETA1P) + l2 * cos(THETA1P + THETA2P);
Yp = l1 * sin(THETA1P) + l2 * sin(THETA1P + THETA2P);

% euclidian distance
ed = sqrt((Xp-XY(:,1)).^2 + (Yp-XY(:,2)).^2);

% RMSE based on euclidian distance
RMSE = sqrt(mean(ed.^2))