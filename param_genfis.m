l1 = 10; % length of first arm
l2 = 7; % length of second arm
% l3 = 4;

nr_p = 50;

theta1 = linspace(0, pi/4, nr_p); % all possible theta1 values
theta2 = linspace(0, pi/2, nr_p); % all possible theta2 values
% theta3 = linspace(0, pi/4, nr_p); % all possible theta3 values

[THETA1,THETA2] = meshgrid(theta1,theta2);
% [THETA1,THETA2,THETA3] = meshgrid(theta1,theta2,theta3);

% X = l1 * cos(THETA1) + l2 * cos(THETA1 + THETA2) + l3 * cos(THETA1 + THETA2 + THETA3); % compute x coordinates
% Y = l1 * sin(THETA1) + l2 * sin(THETA1 + THETA2) + l3 * sin(THETA1 + THETA2 + THETA3); % compute y coordinates

X = l1 * cos(THETA1) + l2 * cos(THETA1 + THETA2);
Y = l1 * sin(THETA1) + l2 * sin(THETA1 + THETA2);

data1 = [X(:) Y(:) THETA1(:)]; % create x-y-theta1 dataset
data2 = [X(:) Y(:) THETA2(:)]; % create x-y-theta2 dataset
% data3 = [X(:) Y(:) THETA3(:)]; % create x-y-theta3 dataset
%%
train_data1 = data1(1:2:end,:);
train_data2 = data2(1:2:end,:);
% train_data3 = data3(1:2:end,:);
val_data1 = data1(2:2:end,:);
val_data2 = data2(2:2:end,:);
% val_data3 = data3(2:2:end,:);
%%
TRAIN_P = 80;
PARTS = 25;
GROUP_SIZE = length(data1)/PARTS;

all_points = 1:length(data1);

parts = reshape(all_points, [PARTS, GROUP_SIZE]);

train_points = [];
val_points = [];
test_points = [];
for i = 1:PARTS
    t = datasample(parts(i,:),floor(TRAIN_P/100*GROUP_SIZE), 'Replace', false);
    rest = setdiff(parts(i,:), t);
    
    v = datasample(rest,floor(size(rest,2)/2), 'Replace', false);
    tp = setdiff(rest, v);
    
    train_points = [train_points t];
    val_points = [val_points v];
    test_points = [test_points tp];
end

% load the points into data sets
train_data1 = data1(train_points, :);
train_data2 = data2(train_points, :);
% train_data3 = data3(train_points, :);

val_data1 = data1(val_points, :);
val_data2 = data2(val_points, :);
% val_data3 = data3(val_points, :);

test_data1 = data1(test_points, :);
test_data2 = data2(test_points, :);
% test_data3 = data3(test_points, :);
%%
err = [];
time = [];
overfit = [];
start = 0.1;
fin = 0.5;
for i = start:0.05:fin
    fprintf('Training with %f.\n', i);
    tic;
    opt = genfisOptions('SubtractiveClustering',...
                    'ClusterInfluenceRange',i);
    fismat=genfis(train_data1(:,1:2),train_data1(:,3),opt);
    opt = anfisOptions;
    opt.InitialFIS = fismat;
    opt.EpochNumber = 150;
    epoch = 1:150;
    opt.DisplayANFISInformation = 0;
    opt.DisplayErrorValues = 0;
    opt.DisplayStepSize = 0;
    opt.DisplayFinalResults = 0;
    opt.ValidationData = val_data1;
    [anfis1,trnErr,ss,anfis12,chkErr] = anfis(train_data1,opt);
    
    t = toc;
%     figure;
%     plot(1:epochs,chkErr,'r');
%     hold on;
%     plot(1:epochs,trnErr,'b');
%     hold off;
%     pause(0.1);
    if (abs(trnErr(end) - chkErr(end)) > mean(chkErr))
        overfit = [overfit 1];
    else
        overfit = [overfit 0];
    end
    err = [err chkErr(end)];
    time = [time t];
end
%%
figure
subplot(4,1,1);
x = start:0.05:fin;
plot(x,err);
ylabel('Error'); 
xlabel('ClusterInfluenceRange');
subplot(4,1,2);
plot(x,time);
ylabel('Time');
xlabel('ClusterInfluenceRange');
subplot(4,1,3);
p_err = rescale(err,0,1);
p_time = rescale(time,0,1);
plot(x,p_err,'r');
hold on;
plot(x,p_time,'b');
hold off;
ylabel('Normalized err and time');
xlabel('ClusterInfluenceRange');

subplot(4,1,4);
score = [];
for i = 1:length(overfit)
    if overfit(i) == 0
        score = [score 1/(p_time(i)+p_err(i))*10];
    else
        score = [score 0];
    end
end
plot(x,score);
axis([start fin 0 max(score)+1]);
ylabel('Fitness');
xlabel('ClusterInfluenceRange');
