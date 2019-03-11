% prepare data

l1 = 10; % length of first arm
l2 = 7; % length of second arm

nr_p = 50;

theta1 = linspace(0, pi/4, nr_p); % all possible theta1 values
theta2 = linspace(0, pi/2, nr_p); % all possible theta2 values

[THETA1,THETA2] = meshgrid(theta1,theta2);

X = l1 * cos(THETA1) + l2 * cos(THETA1 + THETA2);
Y = l1 * sin(THETA1) + l2 * sin(THETA1 + THETA2);

data1 = [X(:) Y(:) THETA1(:)]; % create x-y-theta1 dataset

train_data1 = data1(1:2:end,:);
val_data1 = data1(2:2:end,:);
%% train first
opt = genfisOptions('SubtractiveClustering',...
                    'ClusterInfluenceRange',0.1);

% opt = genfisOptions('GridPartition');
% opt.NumMembershipFunctions = 5;

fismat=genfis(train_data1(:,1:2),train_data1(:,3),opt);

opt = anfisOptions;
opt.InitialFIS = fismat;
opt.EpochNumber = 150;
epoch = 1:150;
opt.DisplayANFISInformation = 0;
opt.DisplayErrorValues = 0;
opt.DisplayStepSize = 0;
opt.DisplayFinalResults = 0;

% Train an ANFIS system using the first set of training data, |data1|.
disp('--> Training first ANFIS network.')

opt.ValidationData = val_data1;

[anfis11,trnErr1,ss1,anfis12,chkErr1] = anfis(train_data1,opt);
disp('--> Finished training first ANFIS network.')
%% train second
opt2 = genfisOptions('GridPartition');
opt2.NumMembershipFunctions = 5;

fismat=genfis(train_data1(:,1:2),train_data1(:,3),opt2);

opt2 = anfisOptions;
opt2.InitialFIS = fismat;
opt2.EpochNumber = 150;
opt2.DisplayANFISInformation = 0;
opt2.DisplayErrorValues = 0;
opt2.DisplayStepSize = 0;
opt2.DisplayFinalResults = 0;

% Train an ANFIS system using the first set of training data, |data1|.
disp('--> Training second ANFIS network.')

opt2.ValidationData = val_data1;

[anfis2,trnErr2,ss2,anfis22,chkErr2] = anfis(train_data1,opt2);
disp('--> Finished training second ANFIS network.')
%% displaying
figure
subplot(2,1,1);
plot(epoch,trnErr1,'o-b',epoch,chkErr1,'x-r')
title('SubtractiveClustering');
subplot(2,1,2);
plot(epoch,trnErr2,'o-b',epoch,chkErr2,'x-r')
title('GridPartition');