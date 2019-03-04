% The following code snippet shows how data is generated for all
% combination of |theta1| and |theta2| values and saved into a matrix to be
% used as training data. The reason for saving the data in two matrices is
% explained in the following section.

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

opt = genfisOptions('SubtractiveClustering',...
                    'ClusterInfluenceRange',0.5);

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

[anfis1,trnErr,ss,anfis12,chkErr] = anfis(train_data1,opt);

figure
plot(epoch,trnErr,'o-b',epoch,chkErr,'x-r')

